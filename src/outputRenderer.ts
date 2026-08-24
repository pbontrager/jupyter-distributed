import { NotebookPanel } from '@jupyterlab/notebook';
import { IRenderMimeRegistry, MimeModel } from '@jupyterlab/rendermime';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { Signal } from '@lumino/signaling';
import { Panel, Widget } from '@lumino/widgets';

export const MIME_TYPE = 'application/vnd.spmd-jupyter.rank+json';

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

/** Stores and broadcasts the selected output rank for each notebook panel. */
export class RankSelectionModel {
  readonly changed = new Signal<this, { panel: NotebookPanel; rank: number }>(
    this
  );

  get(panel: NotebookPanel | null): number {
    return panel ? (this._values.get(panel) ?? 0) : 0;
  }

  set(panel: NotebookPanel | null, rank: number): void {
    if (!panel || this.get(panel) === rank) {
      return;
    }
    this._values.set(panel, rank);
    this.changed.emit({ panel, rank });
  }

  private _values = new WeakMap<NotebookPanel, number>();
}

export class RankOutputRenderer extends Panel implements IRenderMime.IRenderer {
  constructor(
    options: IRenderMime.IRendererOptions,
    rendermime: IRenderMimeRegistry,
    panel: NotebookPanel | null,
    selections: RankSelectionModel
  ) {
    super();
    this.addClass('jp-SpmdRankOutput');
    this._mimeType = options.mimeType;
    this._rendermime = rendermime;
    this._panel = panel;
    this._selections = selections;
    selections.changed.connect(this._onSelectionChanged, this);
  }

  async renderModel(model: IRenderMime.IMimeModel): Promise<void> {
    this._clear();
    const ranks = Private.normalizePayload(model.data[this._mimeType]);
    if (ranks.length === 0) {
      this.addWidget(
        Private.textWidget('No rank output was provided.', 'jp-SpmdRankOutput-empty')
      );
      return;
    }

    this._ranks = ranks.map(item => item.rank);
    const remembered = this._selections.get(this._panel);
    this._selectedRank = this._ranks.includes(remembered)
      ? remembered
      : this._ranks[0];

    const tabs = new Widget({ node: document.createElement('div') });
    tabs.addClass('jp-SpmdRankOutput-tabs');
    tabs.node.setAttribute('role', 'tablist');
    this.addWidget(tabs);

    for (const rank of ranks) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `Rank ${rank.rank}`;
      button.dataset.rank = String(rank.rank);
      button.className = 'jp-SpmdRankOutput-tab';
      button.setAttribute('role', 'tab');
      if (rank.error) {
        button.classList.add('jp-mod-error');
        button.title = `Rank ${rank.rank} produced an error`;
      }
      button.addEventListener('click', () => {
        this._selections.set(this._panel, rank.rank);
        this._select(rank.rank);
      });
      tabs.node.appendChild(button);

      const content = new Panel();
      content.addClass('jp-SpmdRankOutput-rank');
      content.node.dataset.rank = String(rank.rank);
      content.node.setAttribute('role', 'tabpanel');
      this.addWidget(content);
      this._content.set(rank.rank, content);

      for (const output of rank.outputs) {
        await this._renderOutput(content, output, model.trusted);
      }
    }

    this._select(this._selectedRank);
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this._selections.changed.disconnect(this._onSelectionChanged, this);
    this._clear();
    super.dispose();
  }

  private _clear(): void {
    while (this.widgets.length > 0) {
      this.widgets[0].dispose();
    }
    this._content.clear();
    this._ranks = [];
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
            ? 'jp-SpmdRankOutput-stderr'
            : 'jp-SpmdRankOutput-stdout'
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
          'jp-SpmdRankOutput-error'
        )
      );
      return;
    }

    const data = output.data ?? {};
    const mimeType = this._rendermime.preferredMimeType(data, 'ensure');
    if (!mimeType) {
      parent.addWidget(
        Private.textWidget(
          JSON.stringify(data, null, 2),
          'jp-SpmdRankOutput-plain'
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
    const tabs = this.widgets[0]?.node;
    tabs?.querySelectorAll<HTMLButtonElement>('[data-rank]').forEach(button => {
      const selected = Number(button.dataset.rank) === rank;
      button.classList.toggle('jp-mod-selected', selected);
      button.setAttribute('aria-selected', String(selected));
    });
    for (const [candidate, content] of this._content) {
      content.setHidden(candidate !== rank);
    }
  }

  private _onSelectionChanged(
    sender: RankSelectionModel,
    change: { panel: NotebookPanel; rank: number }
  ): void {
    if (change.panel === this._panel) {
      this._select(change.rank);
    }
  }

  private _content = new Map<number, Panel>();
  private _mimeType: string;
  private _panel: NotebookPanel | null;
  private _ranks: number[] = [];
  private _rendermime: IRenderMimeRegistry;
  private _selectedRank = 0;
  private _selections: RankSelectionModel;
}

namespace Private {
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
    widget.addClass('jp-SpmdRankOutput-text');
    widget.addClass(className);
    return widget;
  }
}
