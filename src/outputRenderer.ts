import { IRenderMimeRegistry, MimeModel } from '@jupyterlab/rendermime';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { Signal } from '@lumino/signaling';
import { Panel, Widget } from '@lumino/widgets';

import { RANK_MIME_TYPE } from './constants';
import { executionId } from './rankUpdates';

export const MIME_TYPE = RANK_MIME_TYPE;
const MIN_RANK_TAB_WIDTH = 72;

type OutputType =
  | 'stream'
  | 'execute_result'
  | 'display_data'
  | 'error';

interface RankOutput {
  output_type: OutputType;
  name?: 'stdout' | 'stderr';
  text?: string | string[];
  data?: IRenderMime.IMimeModel['data'];
  metadata?: IRenderMime.IMimeModel['metadata'];
  ename?: string;
  evalue?: string;
  traceback?: string[];
}

interface RankRecord {
  rank: number;
  outputs: RankOutput[];
  error: boolean;
}

/** Stores the selected rank independently for each logical cell execution. */
export class RankSelectionModel {
  readonly changed = new Signal<this, { executionId: string; rank: number }>(
    this
  );

  get(executionId: string | null): number {
    return executionId ? (this._values.get(executionId) ?? 0) : 0;
  }

  has(executionId: string | null): boolean {
    return executionId !== null && this._values.has(executionId);
  }

  set(executionId: string | null, rank: number): void {
    if (
      !executionId ||
      (this._values.has(executionId) && this.get(executionId) === rank)
    ) {
      return;
    }
    this._values.set(executionId, rank);
    this.changed.emit({ executionId, rank });
  }

  private _values = new Map<string, number>();
}

export class RankOutputRenderer extends Panel implements IRenderMime.IRenderer {
  constructor(
    options: IRenderMime.IRendererOptions,
    rendermime: IRenderMimeRegistry,
    selections: RankSelectionModel
  ) {
    super();
    this.addClass('jp-JupyterDistributedRankOutput');
    this._mimeType = options.mimeType;
    this._rendermime = rendermime;
    this._selections = selections;
    selections.changed.connect(this._onSelectionChanged, this);
    this._resizeObserver = new ResizeObserver(() => {
      this._updateNavigationMode();
    });
    this._resizeObserver.observe(this.node);
  }

  async renderModel(model: IRenderMime.IMimeModel): Promise<void> {
    this._executionId = executionId(model.data[this._mimeType]);
    await this._queueRender(model.data[this._mimeType], model.trusted);
  }

  private async _queueRender(
    payload: unknown,
    trusted: boolean
  ): Promise<void> {
    this._pendingRender = { payload, trusted };
    if (this._rendering) {
      return;
    }
    this._rendering = true;
    try {
      while (this._pendingRender !== null && !this.isDisposed) {
        const pending = this._pendingRender;
        this._pendingRender = null;
        await this._renderPayload(pending.payload, pending.trusted);
      }
    } finally {
      this._rendering = false;
    }
  }

  private async _renderPayload(
    payload: unknown,
    trusted: boolean
  ): Promise<void> {
    this._clear();
    const ranks = Private.normalizePayload(payload);
    if (ranks.length === 0) {
      this.addWidget(
        Private.textWidget(
          'No rank output was provided.',
          'jp-JupyterDistributedRankOutput-empty'
        )
      );
      return;
    }
    if (ranks.every(rank => rank.outputs.length === 0)) {
      return;
    }

    this._ranks = ranks.map(item => item.rank);
    this._executionId = executionId(payload);
    const firstOutputRank = ranks.find(rank => rank.outputs.length > 0)?.rank;
    const remembered = this._selections.has(this._executionId)
      ? this._selections.get(this._executionId)
      : (firstOutputRank ?? this._selectedRank);
    this._selectedRank = this._ranks.includes(remembered)
      ? remembered
      : this._ranks[0];

    if (ranks.length > 1) {
      const tabs = new Widget({ node: document.createElement('div') });
      tabs.addClass('jp-JupyterDistributedRankOutput-tabs');
      tabs.node.setAttribute('role', 'tablist');
      this._tabs = tabs;
      this.addWidget(tabs);

      const picker = new Widget({ node: Private.createRankPickerNode() });
      picker.addClass('jp-JupyterDistributedRankOutput-picker');
      this._rankSelect = picker.node.querySelector('select')!;
      this._rankSelect.addEventListener('change', this._onDropdownChange);
      this.addWidget(picker);
    }

    for (const rank of ranks) {
      if (this._tabs) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = `Rank ${rank.rank}`;
        button.dataset.rank = String(rank.rank);
        button.className = 'jp-JupyterDistributedRankOutput-tab';
        button.setAttribute('role', 'tab');
        if (rank.error) {
          button.classList.add('jp-mod-error');
          button.title = `Rank ${rank.rank} produced an error`;
        }
        button.addEventListener('click', () => {
          this._selections.set(this._executionId, rank.rank);
          this._select(rank.rank);
        });
        this._tabs.node.appendChild(button);
      }

      if (this._rankSelect) {
        const option = document.createElement('option');
        option.value = String(rank.rank);
        option.textContent = rank.error
          ? `Rank ${rank.rank} — error`
          : `Rank ${rank.rank}`;
        this._rankSelect.appendChild(option);
      }

      const content = new Panel();
      content.addClass('jp-JupyterDistributedRankOutput-rank');
      content.node.dataset.rank = String(rank.rank);
      content.node.setAttribute('role', 'tabpanel');
      this.addWidget(content);
      this._content.set(rank.rank, content);

      for (const output of rank.outputs) {
        await this._renderOutput(content, output, trusted);
      }
    }

    this._select(this._selectedRank);
    this._updateNavigationMode();
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this._selections.changed.disconnect(this._onSelectionChanged, this);
    this._resizeObserver.disconnect();
    this._clear();
    super.dispose();
  }

  private _clear(): void {
    this._rankSelect?.removeEventListener('change', this._onDropdownChange);
    while (this.widgets.length > 0) {
      this.widgets[0].dispose();
    }
    this._content.clear();
    this._ranks = [];
    this._tabs = null;
    this._rankSelect = null;
  }

  private async _renderOutput(
    parent: Panel,
    output: RankOutput,
    trusted: boolean
  ): Promise<void> {
    if (output.output_type === 'stream') {
      const text = Private.asText(output.text);
      parent.addWidget(
        Private.textWidget(
          text,
          output.name === 'stderr'
            ? 'jp-JupyterDistributedRankOutput-stderr'
            : 'jp-JupyterDistributedRankOutput-stdout'
        )
      );
      return;
    }

    if (output.output_type === 'error') {
      const traceback = output.traceback?.join('\n');
      const summary = [output.ename, output.evalue].filter(Boolean).join(': ');
      parent.addWidget(
        Private.textWidget(
          traceback || summary || 'Unknown error',
          'jp-JupyterDistributedRankOutput-error'
        )
      );
      return;
    }

    const data = output.data ?? {};
    const mimeType = this._rendermime.preferredMimeType(
      data,
      trusted ? 'any' : 'ensure'
    );
    if (!mimeType) {
      parent.addWidget(
        Private.textWidget(
          JSON.stringify(data, null, 2),
          'jp-JupyterDistributedRankOutput-plain'
        )
      );
      return;
    }

    const renderer = this._rendermime.createRenderer(mimeType);
    parent.addWidget(renderer);
    await renderer.renderModel(
      new MimeModel({
        data,
        metadata: output.metadata ?? {},
        trusted
      })
    );
  }

  private _select(rank: number): void {
    if (!this._ranks.includes(rank)) {
      return;
    }
    this._selectedRank = rank;
    this._tabs?.node
      .querySelectorAll<HTMLButtonElement>('[data-rank]')
      .forEach(button => {
        const selected = Number(button.dataset.rank) === rank;
        button.classList.toggle('jp-mod-selected', selected);
        button.setAttribute('aria-selected', String(selected));
      });
    if (this._rankSelect) {
      this._rankSelect.value = String(rank);
    }
    for (const [candidate, content] of this._content) {
      content.setHidden(candidate !== rank);
    }
  }

  private _onSelectionChanged(
    sender: RankSelectionModel,
    change: { executionId: string; rank: number }
  ): void {
    if (change.executionId === this._executionId) {
      this._select(change.rank);
    }
  }

  private _onDropdownChange = (): void => {
    if (!this._rankSelect) {
      return;
    }
    const rank = Number(this._rankSelect.value);
    this._selections.set(this._executionId, rank);
    this._select(rank);
  };

  private _updateNavigationMode(): void {
    const width = this.node.getBoundingClientRect().width;
    const useDropdown =
      width > 0 && this._ranks.length * MIN_RANK_TAB_WIDTH > width;
    this.toggleClass('jp-mod-rankDropdown', useDropdown);
  }

  private _content = new Map<number, Panel>();
  private _executionId: string | null = null;
  private _mimeType: string;
  private _pendingRender: { payload: unknown; trusted: boolean } | null = null;
  private _rankSelect: HTMLSelectElement | null = null;
  private _ranks: number[] = [];
  private _rendermime: IRenderMimeRegistry;
  private _rendering = false;
  private _resizeObserver: ResizeObserver;
  private _selectedRank = 0;
  private _selections: RankSelectionModel;
  private _tabs: Widget | null = null;
}

namespace Private {
  export function createRankPickerNode(): HTMLElement {
    const label = document.createElement('label');
    const text = document.createElement('span');
    const select = document.createElement('select');
    text.textContent = 'Output:';
    select.setAttribute('aria-label', 'Displayed process rank');
    label.append(text, select);
    return label;
  }

  export function normalizePayload(value: unknown): RankRecord[] {
    if (Array.isArray(value)) {
      return value
        .map(item => normalizeRank(item))
        .filter((item): item is RankRecord => item !== null)
        .sort((a, b) => a.rank - b.rank);
    }
    if (!value || typeof value !== 'object') {
      return [];
    }

    const payload = value as Record<string, unknown>;
    if (Array.isArray(payload.ranks)) {
      return payload.ranks
        .map(item => normalizeRank(item))
        .filter((item): item is RankRecord => item !== null)
        .sort((a, b) => a.rank - b.rank);
    }

    const outputs = payload.rank_outputs ?? payload.outputs;
    if (outputs && typeof outputs === 'object' && !Array.isArray(outputs)) {
      return Object.entries(outputs as Record<string, unknown>)
        .map(([rank, item]) => {
          const record =
            item && typeof item === 'object' && !Array.isArray(item)
              ? (item as Record<string, unknown>)
              : null;
          return normalizeRank({
            ...(record ?? {}),
            rank: Number(rank),
            outputs: record?.outputs ?? (Array.isArray(item) ? item : [item])
          });
        })
        .filter((item): item is RankRecord => item !== null)
        .sort((a, b) => a.rank - b.rank);
    }

    return [];
  }

  function normalizeRank(value: unknown): RankRecord | null {
    if (!value || typeof value !== 'object') {
      return null;
    }
    const candidate = value as Record<string, unknown>;
    const rank = Number(candidate.rank);
    if (!Number.isInteger(rank) || rank < 0) {
      return null;
    }
    const rawOutputs = Array.isArray(candidate.outputs)
      ? candidate.outputs
      : candidate.output
        ? [candidate.output]
        : [];
    const outputs = rawOutputs
      .map(normalizeOutput)
      .filter((item): item is RankOutput => item !== null);
    return {
      rank,
      outputs,
      error:
        candidate.error === true ||
        candidate.status === 'error' ||
        outputs.some(output => output.output_type === 'error')
    };
  }

  function normalizeOutput(value: unknown): RankOutput | null {
    if (!value || typeof value !== 'object') {
      return null;
    }
    const record = value as Record<string, unknown>;
    const header =
      record.header && typeof record.header === 'object'
        ? (record.header as Record<string, unknown>)
        : undefined;
    const discriminator =
      record.output_type ?? record.type ?? record.msg_type ?? header?.msg_type;
    const outputType =
      discriminator === 'stdout' || discriminator === 'stderr'
        ? 'stream'
        : discriminator;
    const recognized =
      outputType === 'stream' ||
      outputType === 'execute_result' ||
      outputType === 'display_data' ||
      outputType === 'error';
    if (!recognized) {
      return null;
    }
    const content =
      record.content && typeof record.content === 'object'
        ? (record.content as Record<string, unknown>)
        : record;
    return {
      ...content,
      output_type: outputType,
      ...(discriminator === 'stdout' || discriminator === 'stderr'
        ? { name: discriminator }
        : {})
    } as unknown as RankOutput;
  }

  export function asText(value: string | string[] | undefined): string {
    return Array.isArray(value) ? value.join('') : (value ?? '');
  }

  export function textWidget(text: string, className: string): Widget {
    const node = document.createElement('pre');
    node.textContent = text;
    const widget = new Widget({ node });
    widget.addClass('jp-JupyterDistributedRankOutput-text');
    widget.addClass(className);
    return widget;
  }
}
