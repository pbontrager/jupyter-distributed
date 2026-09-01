import type * as nbformat from '@jupyterlab/nbformat';
import { OutputArea, OutputAreaModel } from '@jupyterlab/outputarea';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { JSONExt } from '@lumino/coreutils';
import { Signal } from '@lumino/signaling';
import { Panel, TabBar, Widget } from '@lumino/widgets';

import { RANK_MIME_TYPE } from './constants';

export const MIME_TYPE = RANK_MIME_TYPE;
const MIN_RANK_TAB_WIDTH = 72;

function executionId(value: unknown): string | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const id = (value as Record<string, unknown>).execution_id;
  return typeof id === 'string' && id ? id : null;
}

interface RankRecord {
  rank: number;
  outputs: nbformat.IOutput[];
  error: boolean;
}

interface RankView {
  area: OutputArea;
  model: OutputAreaModel;
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
    const normalizedRanks = Private.normalizePayload(payload);
    const targetRank = Private.targetRank(payload);
    const ranks =
      targetRank === null
        ? normalizedRanks
        : normalizedRanks.filter(rank => rank.rank === targetRank);
    if (ranks.length === 0) {
      this._clear();
      this.addWidget(
        Private.textWidget(
          'No rank output was provided.',
          'jp-JupyterDistributedRankOutput-empty'
        )
      );
      return;
    }
    if (ranks.every(rank => rank.outputs.length === 0)) {
      this._clear();
      return;
    }

    const nextExecutionId = executionId(payload);
    const nextRanks = ranks.map(item => item.rank);
    const nextLayoutKey = JSON.stringify({
      executionId: nextExecutionId,
      ranks: nextRanks,
      targetRank
    });
    if (nextLayoutKey !== this._layoutKey) {
      this._clear();
      this._layoutKey = nextLayoutKey;
      this._executionId = nextExecutionId;
      this._ranks = nextRanks;
      const firstOutputRank = ranks.find(rank => rank.outputs.length > 0)?.rank;
      const remembered = this._selections.has(this._executionId)
        ? this._selections.get(this._executionId)
        : (firstOutputRank ?? this._selectedRank);
      this._selectedRank = this._ranks.includes(remembered)
        ? remembered
        : this._ranks[0];
      this._createNavigation(ranks.length);
      for (const rank of ranks) {
        this._createRankView(rank.rank, trusted);
      }
    }

    for (const rank of ranks) {
      this._updateRankStatus(rank);
      const view = this._rankViews.get(rank.rank);
      if (view) {
        view.model.trusted = trusted;
        Private.syncOutputs(view.model, rank.outputs);
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
    for (const view of this._rankViews.values()) {
      view.model.dispose();
    }
    this._rankViews.clear();
    this._executionId = null;
    this._ranks = [];
    this._tabs = null;
    this._rankSelect = null;
    this._layoutKey = null;
  }

  private _createNavigation(rankCount: number): void {
    if (rankCount <= 1) {
      return;
    }
    const tabs = new TabBar<Widget>({
      allowDeselect: false,
      insertBehavior: 'none',
      removeBehavior: 'select-previous-tab',
      tabsMovable: false
    });
    tabs.addClass('jp-JupyterDistributedRankOutput-tabs');
    tabs.currentChanged.connect(this._onTabChanged, this);
    this._tabs = tabs;
    this.addWidget(tabs);

    const picker = new Widget({ node: Private.createRankPickerNode() });
    picker.addClass('jp-JupyterDistributedRankOutput-picker');
    this._rankSelect = picker.node.querySelector('select')!;
    this._rankSelect.addEventListener('change', this._onDropdownChange);
    this.addWidget(picker);

    for (const rank of this._ranks) {
      const option = document.createElement('option');
      option.value = String(rank);
      this._rankSelect.appendChild(option);
    }
  }

  private _createRankView(rank: number, trusted: boolean): void {
    // Keep JupyterLab's native output model and widget alive for the duration
    // of an execution. Snapshot updates mutate the model instead of replacing
    // renderers, preserving widget views and third-party MIME behavior.
    const model = new OutputAreaModel({ trusted });
    const area = new OutputArea({
      model,
      rendermime: this._rendermime,
      promptOverlay: false,
      showInputPlaceholder: false
    });
    area.addClass('jp-JupyterDistributedRankOutput-rank');
    area.node.dataset.rank = String(rank);
    area.node.setAttribute('role', 'tabpanel');
    area.title.label = `Rank ${rank}`;
    area.title.caption = `Output from rank ${rank}`;
    area.title.closable = false;
    this._tabs?.addTab(area.title);
    this.addWidget(area);
    this._rankViews.set(rank, { area, model });
  }

  private _updateRankStatus(rank: RankRecord): void {
    const view = this._rankViews.get(rank.rank);
    if (view) {
      view.area.title.className = rank.error ? 'jp-mod-error' : '';
      view.area.title.caption = rank.error
        ? `Rank ${rank.rank} produced an error`
        : `Output from rank ${rank.rank}`;
    }
    const option = this._rankSelect?.querySelector<HTMLOptionElement>(
      `option[value="${rank.rank}"]`
    );
    if (option) {
      option.textContent = rank.error
        ? `Rank ${rank.rank} — error`
        : `Rank ${rank.rank}`;
    }
  }

  private _select(rank: number): void {
    if (!this._ranks.includes(rank)) {
      return;
    }
    this._selectedRank = rank;
    const selectedView = this._rankViews.get(rank);
    if (this._tabs && selectedView) {
      this._tabs.currentTitle = selectedView.area.title;
    }
    if (this._rankSelect) {
      this._rankSelect.value = String(rank);
    }
    for (const [candidate, view] of this._rankViews) {
      const selected = candidate === rank;
      view.area.setHidden(!selected);
      view.area.node.setAttribute('aria-hidden', String(!selected));
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

  private _onTabChanged(
    sender: TabBar<Widget>,
    change: TabBar.ICurrentChangedArgs<Widget>
  ): void {
    const rank = Number(change.currentTitle?.owner.node.dataset.rank);
    if (Number.isInteger(rank) && this._ranks.includes(rank)) {
      this._selections.set(this._executionId, rank);
      this._select(rank);
    }
  }

  private _updateNavigationMode(): void {
    const width = this.node.getBoundingClientRect().width;
    const useDropdown =
      width > 0 && this._ranks.length * MIN_RANK_TAB_WIDTH > width;
    this.toggleClass('jp-mod-rankDropdown', useDropdown);
  }

  private _executionId: string | null = null;
  private _layoutKey: string | null = null;
  private _mimeType: string;
  private _pendingRender: { payload: unknown; trusted: boolean } | null = null;
  private _rankSelect: HTMLSelectElement | null = null;
  private _rankViews = new Map<number, RankView>();
  private _ranks: number[] = [];
  private _rendermime: IRenderMimeRegistry;
  private _rendering = false;
  private _resizeObserver: ResizeObserver;
  private _selectedRank = 0;
  private _selections: RankSelectionModel;
  private _tabs: TabBar<Widget> | null = null;
}

namespace Private {
  export function targetRank(value: unknown): number | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return null;
    }
    const target = Number((value as Record<string, unknown>).target_rank);
    return Number.isInteger(target) && target >= 0 ? target : null;
  }

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
      .filter((item): item is nbformat.IOutput => item !== null);
    return {
      rank,
      outputs,
      error:
        candidate.error === true ||
        candidate.status === 'error' ||
        outputs.some(output => output.output_type === 'error')
    };
  }

  function normalizeOutput(value: unknown): nbformat.IOutput | null {
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
    const output = {
      ...content,
      output_type: outputType,
      ...(discriminator === 'stdout' || discriminator === 'stderr'
        ? { name: discriminator }
        : {})
    } as unknown as nbformat.IOutput;
    if (output.output_type === 'stream' && Array.isArray(output.text)) {
      output.text = output.text.join('');
    }
    delete (output as unknown as Record<string, unknown>).transient;
    return output;
  }

  export function syncOutputs(
    model: OutputAreaModel,
    outputs: nbformat.IOutput[]
  ): void {
    while (model.length > outputs.length) {
      model.remove(model.length - 1);
    }
    const commonLength = Math.min(model.length, outputs.length);
    for (let index = 0; index < commonLength; index++) {
      const currentModel = model.get(index);
      const current = currentModel.toJSON();
      const next = outputs[index];
      if (JSONExt.deepEqual(current, next)) {
        continue;
      }
      if (
        current.output_type === 'stream' &&
        next.output_type === 'stream' &&
        current.name === next.name &&
        index === model.length - 1
      ) {
        const currentText = currentModel.streamText?.text ?? '';
        const nextText = asText(next.text);
        if (nextText.startsWith(currentText)) {
          model.appendStreamOutput(nextText.slice(currentText.length));
        } else {
          // Snapshot transport has already applied carriage returns and
          // backspaces. Mutate the existing stream model so JupyterLab keeps
          // exactly one renderer and no private output-list cursor is changed.
          model.removeStreamOutput(currentText.length);
          model.appendStreamOutput(nextText);
        }
      } else {
        model.set(index, next);
      }
    }
    for (let index = model.length; index < outputs.length; index++) {
      model.add(outputs[index]);
    }
  }

  export function textWidget(text: string, className: string): Widget {
    const node = document.createElement('pre');
    node.textContent = text;
    const widget = new Widget({ node });
    widget.addClass('jp-JupyterDistributedRankOutput-text');
    widget.addClass(className);
    return widget;
  }

  function asText(value: unknown): string {
    return Array.isArray(value) ? value.join('') : String(value ?? '');
  }
}
