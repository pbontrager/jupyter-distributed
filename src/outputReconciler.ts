import { isCodeCellModel } from '@jupyterlab/cells';
import type { CellList, NotebookPanel } from '@jupyterlab/notebook';
import type { IOutputAreaModel } from '@jupyterlab/outputarea';
import type { IDisposable } from '@lumino/disposable';

import { RANK_MIME_TYPE } from './constants';

/**
 * Reconciles cumulative distributed snapshots that were appended as separate
 * notebook outputs instead of applied as display updates.
 *
 * Some server-side execution providers currently persist `update_display_data`
 * messages as new `display_data` outputs. Keeping the first output model alive
 * and updating it with the newest snapshot preserves JupyterLab renderer state
 * while ensuring the notebook contains one output per distributed execution.
 */
export class DistributedOutputReconciler implements IDisposable {
  constructor(panel: NotebookPanel) {
    this._panel = panel;
    panel.disposed.connect(this.dispose, this);
    void panel.context.ready.then(() => {
      if (this._isDisposed) {
        return;
      }
      const cells = panel.content.model?.cells;
      if (!cells) {
        return;
      }
      this._cells = cells;
      cells.changed.connect(this._onCellsChanged, this);
      this._watchCells();
    });
  }

  get isDisposed(): boolean {
    return this._isDisposed;
  }

  dispose(): void {
    if (this._isDisposed) {
      return;
    }
    this._isDisposed = true;
    this._panel.disposed.disconnect(this.dispose, this);
    this._cells?.changed.disconnect(this._onCellsChanged, this);
    for (const outputs of this._outputs) {
      outputs.changed.disconnect(this._onOutputsChanged, this);
    }
    this._outputs.clear();
    this._cells = null;
  }

  private _watchCells(): void {
    if (!this._cells) {
      return;
    }
    const current = new Set<IOutputAreaModel>();
    for (let index = 0; index < this._cells.length; index++) {
      const cell = this._cells.get(index);
      if (isCodeCellModel(cell)) {
        current.add(cell.outputs);
      }
    }
    for (const outputs of this._outputs) {
      if (!current.has(outputs)) {
        outputs.changed.disconnect(this._onOutputsChanged, this);
        this._outputs.delete(outputs);
      }
    }
    for (const outputs of current) {
      if (!this._outputs.has(outputs)) {
        this._outputs.add(outputs);
        outputs.changed.connect(this._onOutputsChanged, this);
      }
      this._reconcile(outputs);
    }
  }

  private _reconcile(outputs: IOutputAreaModel): void {
    if (this._reconciling.has(outputs)) {
      return;
    }

    const canonical = new Map<string, number>();
    const duplicates = new Map<number, number>();
    for (let index = 0; index < outputs.length; index++) {
      const id = Private.executionId(outputs.get(index).data[RANK_MIME_TYPE]);
      if (!id) {
        continue;
      }
      const existing = canonical.get(id);
      if (existing === undefined) {
        canonical.set(id, index);
      } else {
        duplicates.set(index, existing);
      }
    }
    if (duplicates.size === 0) {
      return;
    }

    this._reconciling.add(outputs);
    try {
      // Each cumulative snapshot supersedes the preceding one. Apply the
      // newest data to the original model so its renderer remains mounted.
      for (const [duplicateIndex, canonicalIndex] of duplicates) {
        const latest = outputs.get(duplicateIndex);
        outputs.get(canonicalIndex).setData({
          data: latest.data,
          metadata: latest.metadata
        });
      }
      const indices = [...duplicates.keys()].sort((a, b) => b - a);
      for (const index of indices) {
        outputs.remove(index);
      }
    } finally {
      this._reconciling.delete(outputs);
    }
  }

  private _onCellsChanged(): void {
    this._watchCells();
  }

  private _onOutputsChanged(outputs: IOutputAreaModel): void {
    this._reconcile(outputs);
  }

  private _cells: CellList | null = null;
  private _isDisposed = false;
  private _outputs = new Set<IOutputAreaModel>();
  private _panel: NotebookPanel;
  private _reconciling = new Set<IOutputAreaModel>();
}

namespace Private {
  export function executionId(value: unknown): string | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return null;
    }
    const id = (value as Record<string, unknown>).execution_id;
    return typeof id === 'string' && id ? id : null;
  }
}
