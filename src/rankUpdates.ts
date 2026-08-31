import { ICodeCellModel } from '@jupyterlab/cells';
import { NotebookPanel } from '@jupyterlab/notebook';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { IDisposable } from '@lumino/disposable';

import { RANK_MIME_TYPE, RANK_UPDATE_COMM_TARGET } from './constants';

export interface RankUpdateSnapshot {
  execution_id: string;
  cell_id?: string | null;
  final: boolean;
  data: IRenderMime.IMimeModel['data'];
  metadata: IRenderMime.IMimeModel['metadata'];
}

interface SnapshotWaiter {
  cellId: string;
  previousExecutionId: string | null;
  resolve: (snapshot: RankUpdateSnapshot | null) => void;
  timer: number;
}

export function executionId(value: unknown): string | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const id = (value as Record<string, unknown>).execution_id;
  return typeof id === 'string' && id ? id : null;
}

/** Owns rank-output state and the update comm for one notebook panel. */
export class RankOutputController implements IDisposable {
  constructor(panel: NotebookPanel) {
    this._panel = panel;
    panel.sessionContext.kernelChanged.connect(this._onKernelChanged, this);
    panel.sessionContext.connectionStatusChanged.connect(
      this._onConnectionStatusChanged,
      this
    );
    void panel.context.ready.then(() => {
      if (this._disposed) {
        return;
      }
      panel.content.model?.cells.changed.connect(this._onCellsChanged, this);
      this._watchCellOutputs();
      void this.ensureConnected();
    });
  }

  get isDisposed(): boolean {
    return this._disposed;
  }

  latestExecutionId(cellId: string): string | null {
    return this._cellExecutions.get(cellId)?.execution_id ?? null;
  }

  async ensureConnected(): Promise<boolean> {
    if (this._disposed) {
      return false;
    }
    const kernel = this._panel.sessionContext.session?.kernel ?? null;
    if (
      kernel &&
      kernel === this._kernel &&
      this._comm &&
      !this._comm.isDisposed
    ) {
      return true;
    }
    if (this._connecting) {
      return this._connecting;
    }
    const connecting = this._connect();
    this._connecting = connecting;
    try {
      return await connecting;
    } finally {
      if (this._connecting === connecting) {
        this._connecting = null;
      }
    }
  }

  async reconnect(): Promise<boolean> {
    this._invalidateConnection();
    const deadline = Date.now() + 10000;
    do {
      if (await this.ensureConnected()) {
        return true;
      }
      await Private.delay(100);
    } while (!this._disposed && Date.now() < deadline);
    return false;
  }

  waitForFinal(
    cellId: string,
    previousExecutionId: string | null,
    timeout = 5000
  ): Promise<RankUpdateSnapshot | null> {
    const current = this._cellExecutions.get(cellId);
    if (current?.final && current.execution_id !== previousExecutionId) {
      return Promise.resolve(current);
    }
    return new Promise(resolve => {
      const waiter: SnapshotWaiter = {
        cellId,
        previousExecutionId,
        resolve,
        timer: window.setTimeout(() => {
          this._waiters.delete(waiter);
          resolve(null);
        }, timeout)
      };
      this._waiters.add(waiter);
    });
  }

  dispose(): void {
    if (this._disposed) {
      return;
    }
    this._disposed = true;
    if (this._reconnectTimer !== null) {
      window.clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this._panel.sessionContext.kernelChanged.disconnect(
      this._onKernelChanged,
      this
    );
    this._panel.sessionContext.connectionStatusChanged.disconnect(
      this._onConnectionStatusChanged,
      this
    );
    this._panel.content.model?.cells.changed.disconnect(this._onCellsChanged, this);
    for (const outputs of this._watchedOutputs) {
      outputs.changed.disconnect(this._onOutputsChanged, this);
    }
    this._watchedOutputs.clear();
    this._invalidateConnection();
    for (const waiter of this._waiters) {
      window.clearTimeout(waiter.timer);
      waiter.resolve(null);
    }
    this._waiters.clear();
  }

  private async _connect(): Promise<boolean> {
    const kernel = this._panel.sessionContext.session?.kernel ?? null;
    if (!kernel) {
      return false;
    }
    const generation = this._generation;
    try {
      const infoReply = await kernel.requestKernelInfo();
      const implementation =
        infoReply && 'implementation' in infoReply.content
          ? infoReply.content.implementation
          : null;
      if (
        this._disposed ||
        generation !== this._generation ||
        kernel !== this._panel.sessionContext.session?.kernel ||
        implementation !== 'jupyter_distributed'
      ) {
        return false;
      }

      this._closeComm();
      this._kernel = kernel;
      const comm = kernel.createComm(RANK_UPDATE_COMM_TARGET);
      let settleReady: (ready: boolean) => void = () => undefined;
      const ready = new Promise<boolean>(resolve => {
        const timer = window.setTimeout(() => resolve(false), 5000);
        settleReady = value => {
          window.clearTimeout(timer);
          resolve(value);
        };
      });
      comm.onMsg = message => {
        this._onMessage(message);
        const data = message.content.data as Record<string, unknown>;
        if (data.method === 'snapshots') {
          settleReady(true);
        }
      };
      comm.onClose = () => {
        settleReady(false);
        if (this._comm === comm) {
          this._comm = null;
          this._kernel = null;
          this._scheduleReconnect();
        }
      };
      this._comm = comm;
      void comm.open({}).done.catch(error => {
        settleReady(false);
        if (this._comm === comm) {
          this._comm = null;
          this._kernel = null;
          console.warn('Unable to open distributed rank update comm', error);
          this._scheduleReconnect();
        }
      });
      const connected = await ready;
      if (
        !connected ||
        this._disposed ||
        generation !== this._generation ||
        this._comm !== comm
      ) {
        if (this._comm === comm) {
          this._closeComm();
          this._scheduleReconnect();
        }
        return false;
      }
      return true;
    } catch (error) {
      console.warn('Unable to connect distributed rank updates', error);
      this._scheduleReconnect();
      return false;
    }
  }

  private _disconnect(): void {
    this._invalidateConnection();
  }

  private _invalidateConnection(): void {
    this._generation += 1;
    this._connecting = null;
    this._closeComm();
  }

  private _closeComm(): void {
    const comm = this._comm;
    this._comm = null;
    this._kernel = null;
    if (!comm || comm.isDisposed) {
      return;
    }
    try {
      void comm.close().done.catch(() => undefined);
    } catch {
      comm.dispose();
    }
  }

  private _record(value: unknown): void {
    const snapshot = Private.normalizeSnapshot(value);
    if (!snapshot) {
      return;
    }
    if (snapshot.cell_id) {
      this._cellExecutions.set(snapshot.cell_id, snapshot);
    }
    if (!this._apply(snapshot) && snapshot.cell_id) {
      this._pending.set(snapshot.execution_id, snapshot);
    }
    if (snapshot.final && snapshot.cell_id) {
      for (const waiter of [...this._waiters]) {
        if (
          waiter.cellId === snapshot.cell_id &&
          waiter.previousExecutionId !== snapshot.execution_id
        ) {
          window.clearTimeout(waiter.timer);
          this._waiters.delete(waiter);
          waiter.resolve(snapshot);
        }
      }
    }
  }

  private _apply(snapshot: RankUpdateSnapshot): boolean {
    if (!snapshot.cell_id) {
      return false;
    }
    const cells = this._panel.content.model?.cells;
    if (!cells) {
      return false;
    }
    let cell: ICodeCellModel | null = null;
    for (let index = 0; index < cells.length; index++) {
      const candidate = cells.get(index);
      if (candidate.id === snapshot.cell_id && candidate.type === 'code') {
        cell = candidate as ICodeCellModel;
        break;
      }
    }
    if (!cell) {
      return false;
    }
    for (let index = 0; index < cell.outputs.length; index++) {
      const output = cell.outputs.get(index);
      if (executionId(output.data[RANK_MIME_TYPE]) === snapshot.execution_id) {
        this._pending.delete(snapshot.execution_id);
        output.setData({ data: snapshot.data, metadata: snapshot.metadata });
        return true;
      }
    }
    return false;
  }

  private _applyPending(): void {
    for (const snapshot of [...this._pending.values()]) {
      this._apply(snapshot);
    }
  }

  private _watchCellOutputs(): void {
    const cells = this._panel.content.model?.cells;
    const current = new Set<ICodeCellModel['outputs']>();
    if (cells) {
      for (let index = 0; index < cells.length; index++) {
        const cell = cells.get(index);
        if (cell.type === 'code') {
          current.add((cell as ICodeCellModel).outputs);
        }
      }
    }
    for (const outputs of this._watchedOutputs) {
      if (!current.has(outputs)) {
        outputs.changed.disconnect(this._onOutputsChanged, this);
      }
    }
    for (const outputs of current) {
      if (!this._watchedOutputs.has(outputs)) {
        outputs.changed.connect(this._onOutputsChanged, this);
      }
    }
    this._watchedOutputs = current;
    this._applyPending();
  }

  private _scheduleReconnect(): void {
    if (
      this._disposed ||
      this._reconnectTimer !== null ||
      this._panel.sessionContext.session?.kernel?.connectionStatus !==
        'connected'
    ) {
      return;
    }
    this._reconnectTimer = window.setTimeout(() => {
      this._reconnectTimer = null;
      void this.ensureConnected();
    }, 250);
  }

  private _onKernelChanged = (): void => {
    this._disconnect();
    void this.ensureConnected();
  };

  private _onConnectionStatusChanged = (): void => {
    const status =
      this._panel.sessionContext.session?.kernel?.connectionStatus;
    if (status === 'connected') {
      void this.ensureConnected();
    } else {
      this._disconnect();
    }
  };

  private _onCellsChanged = (): void => {
    this._watchCellOutputs();
  };

  private _onOutputsChanged = (): void => {
    this._applyPending();
  };

  private _onMessage = (message: KernelMessage.ICommMsgMsg): void => {
    const data = message.content.data as Record<string, unknown>;
    if (data.method === 'update') {
      this._record(data.snapshot);
      return;
    }
    if (data.method === 'snapshots' && Array.isArray(data.snapshots)) {
      for (const snapshot of data.snapshots) {
        this._record(snapshot);
      }
    }
  };

  private _cellExecutions = new Map<string, RankUpdateSnapshot>();
  private _comm: Kernel.IComm | null = null;
  private _connecting: Promise<boolean> | null = null;
  private _disposed = false;
  private _generation = 0;
  private _kernel: Kernel.IKernelConnection | null = null;
  private _panel: NotebookPanel;
  private _pending = new Map<string, RankUpdateSnapshot>();
  private _reconnectTimer: number | null = null;
  private _watchedOutputs = new Set<ICodeCellModel['outputs']>();
  private _waiters = new Set<SnapshotWaiter>();
}

namespace Private {
  export function delay(milliseconds: number): Promise<void> {
    return new Promise(resolve => window.setTimeout(resolve, milliseconds));
  }

  export function normalizeSnapshot(value: unknown): RankUpdateSnapshot | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return null;
    }
    const candidate = value as Record<string, unknown>;
    if (
      typeof candidate.execution_id !== 'string' ||
      !candidate.execution_id ||
      typeof candidate.final !== 'boolean' ||
      !candidate.data ||
      typeof candidate.data !== 'object' ||
      Array.isArray(candidate.data)
    ) {
      return null;
    }
    const metadata =
      candidate.metadata &&
      typeof candidate.metadata === 'object' &&
      !Array.isArray(candidate.metadata)
        ? candidate.metadata
        : {};
    return {
      execution_id: candidate.execution_id,
      cell_id:
        typeof candidate.cell_id === 'string' ? candidate.cell_id : null,
      final: candidate.final,
      data: candidate.data as IRenderMime.IMimeModel['data'],
      metadata: metadata as IRenderMime.IMimeModel['metadata']
    };
  }
}
