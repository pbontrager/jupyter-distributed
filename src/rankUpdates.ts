import { NotebookPanel } from '@jupyterlab/notebook';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { IDisposable } from '@lumino/disposable';
import { Signal } from '@lumino/signaling';

import { RANK_UPDATE_COMM_TARGET } from './constants';

export interface RankUpdateSnapshot {
  execution_id: string;
  final: boolean;
  data: IRenderMime.IMimeModel['data'];
  metadata: IRenderMime.IMimeModel['metadata'];
}

/** Stores transient snapshots independently of the notebook document model. */
export class RankUpdateModel {
  readonly changed = new Signal<
    this,
    { executionId: string; snapshot: RankUpdateSnapshot }
  >(this);

  get(executionId: string | null): RankUpdateSnapshot | undefined {
    return executionId ? this._snapshots.get(executionId) : undefined;
  }

  update(value: unknown): void {
    const snapshot = Private.normalizeSnapshot(value);
    if (!snapshot) {
      return;
    }
    this._snapshots.set(snapshot.execution_id, snapshot);
    while (this._snapshots.size > 16) {
      const oldest = this._snapshots.keys().next().value;
      if (oldest === undefined) {
        break;
      }
      this._snapshots.delete(oldest);
    }
    this.changed.emit({ executionId: snapshot.execution_id, snapshot });
  }

  private _snapshots = new Map<string, RankUpdateSnapshot>();
}

/** Owns the transient rank-update comm for one notebook connection. */
export class RankUpdateComm implements IDisposable {
  constructor(panel: NotebookPanel, updates: RankUpdateModel) {
    this._panel = panel;
    this._updates = updates;
    panel.disposed.connect(this.dispose, this);
    panel.sessionContext.kernelChanged.connect(this._onKernelChanged, this);
    panel.sessionContext.connectionStatusChanged.connect(
      this._onConnectionStatusChanged,
      this
    );
  }

  get isDisposed(): boolean {
    return this._disposed;
  }

  async ensureConnected(): Promise<boolean> {
    if (this._disposed) {
      return false;
    }
    this._enabled = true;
    const kernel = this._connectedKernel();
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
    this._enabled = true;
    this._invalidateConnection();
    const deadline = Date.now() + 10000;
    do {
      if (await this.ensureConnected()) {
        return true;
      }
      await Private.delay(100);
    } while (!this._disposed && Date.now() < deadline);
    console.warn('Unable to connect distributed rank updates');
    return false;
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
    this._panel.disposed.disconnect(this.dispose, this);
    this._panel.sessionContext.kernelChanged.disconnect(
      this._onKernelChanged,
      this
    );
    this._panel.sessionContext.connectionStatusChanged.disconnect(
      this._onConnectionStatusChanged,
      this
    );
    this._invalidateConnection();
  }

  private _connectedKernel(): Kernel.IKernelConnection | null {
    const kernel = this._panel.sessionContext.session?.kernel ?? null;
    return kernel?.connectionStatus === 'connected' ? kernel : null;
  }

  private async _connect(): Promise<boolean> {
    const kernel = this._connectedKernel();
    if (!kernel) {
      return false;
    }
    const generation = this._generation;
    const comm = kernel.createComm(RANK_UPDATE_COMM_TARGET);
    let settle: (connected: boolean) => void = () => undefined;
    const acknowledged = new Promise<boolean>(resolve => {
      const timer = window.setTimeout(() => resolve(false), 2000);
      settle = connected => {
        window.clearTimeout(timer);
        resolve(connected);
      };
    });
    comm.onMsg = message => {
      this._onMessage(message);
      const data = message.content.data as Record<string, unknown>;
      if (data.method === 'snapshots') {
        settle(true);
      }
    };
    comm.onClose = () => {
      settle(false);
      if (this._comm === comm) {
        this._comm = null;
        this._kernel = null;
        this._scheduleReconnect();
      }
    };
    this._kernel = kernel;
    this._comm = comm;
    try {
      void comm.open({}).done.catch(() => settle(false));
      const connected = await acknowledged;
      if (
        connected &&
        !this._disposed &&
        generation === this._generation &&
        kernel === this._connectedKernel() &&
        this._comm === comm
      ) {
        return true;
      }
    } catch {
      // A restart can invalidate the comm while it is opening.
    }
    if (this._comm === comm) {
      this._closeComm();
    }
    return false;
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

  private _scheduleReconnect(): void {
    if (
      !this._enabled ||
      this._disposed ||
      this._reconnectTimer !== null ||
      !this._connectedKernel()
    ) {
      return;
    }
    this._reconnectTimer = window.setTimeout(() => {
      this._reconnectTimer = null;
      void this.reconnect();
    }, 250);
  }

  private _onKernelChanged = (): void => {
    this._invalidateConnection();
    if (this._enabled) {
      void this.ensureConnected();
    }
  };

  private _onConnectionStatusChanged = (): void => {
    if (this._connectedKernel()) {
      if (this._enabled) {
        void this.ensureConnected();
      }
    } else {
      this._invalidateConnection();
    }
  };

  private _onMessage = (message: KernelMessage.ICommMsgMsg): void => {
    const data = message.content.data as Record<string, unknown>;
    if (data.method === 'update') {
      this._updates.update(data.snapshot);
      return;
    }
    if (data.method === 'snapshots' && Array.isArray(data.snapshots)) {
      for (const snapshot of data.snapshots) {
        this._updates.update(snapshot);
      }
    }
  };

  private _comm: Kernel.IComm | null = null;
  private _connecting: Promise<boolean> | null = null;
  private _disposed = false;
  private _enabled = false;
  private _generation = 0;
  private _kernel: Kernel.IKernelConnection | null = null;
  private _panel: NotebookPanel;
  private _reconnectTimer: number | null = null;
  private _updates: RankUpdateModel;
}

export function executionId(value: unknown): string | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const id = (value as Record<string, unknown>).execution_id;
  return typeof id === 'string' && id ? id : null;
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
      final: candidate.final,
      data: candidate.data as IRenderMime.IMimeModel['data'],
      metadata: metadata as IRenderMime.IMimeModel['metadata']
    };
  }
}
