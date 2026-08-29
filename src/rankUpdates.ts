import { NotebookPanel } from '@jupyterlab/notebook';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { IDisposable } from '@lumino/disposable';
import { Signal } from '@lumino/signaling';

export const RANK_UPDATE_COMM_TARGET = 'jupyter.distributed.rank_updates';
const MAX_RECENT_SNAPSHOTS = 64;

export interface RankUpdateSnapshot {
  execution_id: string;
  cell_id?: string | null;
  final: boolean;
  data: IRenderMime.IMimeModel['data'];
  metadata: IRenderMime.IMimeModel['metadata'];
}

export function executionId(value: unknown): string | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const id = (value as Record<string, unknown>).execution_id;
  return typeof id === 'string' && id ? id : null;
}

/** Stores the newest live snapshot for each distributed cell execution. */
export class RankUpdateModel {
  readonly changed = new Signal<
    this,
    { executionId: string; snapshot: RankUpdateSnapshot }
  >(this);

  get(id: string | null): RankUpdateSnapshot | undefined {
    return id ? this._snapshots.get(id) : undefined;
  }

  update(value: unknown): void {
    const snapshot = Private.normalizeSnapshot(value);
    if (!snapshot) {
      return;
    }
    this._snapshots.delete(snapshot.execution_id);
    this._snapshots.set(snapshot.execution_id, snapshot);
    while (this._snapshots.size > MAX_RECENT_SNAPSHOTS) {
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

/** Maintains the frontend-created comm used for executor-independent updates. */
export class RankUpdateComm implements IDisposable {
  constructor(panel: NotebookPanel, updates: RankUpdateModel) {
    this._panel = panel;
    this._updates = updates;
    panel.sessionContext.kernelChanged.connect(this._onKernelChanged, this);
    panel.sessionContext.connectionStatusChanged.connect(
      this._onConnectionStatusChanged,
      this
    );
    void this._connect();
  }

  get isDisposed(): boolean {
    return this._disposed;
  }

  dispose(): void {
    if (this._disposed) {
      return;
    }
    this._disposed = true;
    this._request += 1;
    this._panel.sessionContext.kernelChanged.disconnect(
      this._onKernelChanged,
      this
    );
    this._panel.sessionContext.connectionStatusChanged.disconnect(
      this._onConnectionStatusChanged,
      this
    );
    this._disconnect();
  }

  private async _connect(force = false): Promise<void> {
    if (this._disposed) {
      return;
    }
    const kernel = this._panel.sessionContext.session?.kernel ?? null;
    if (
      !force &&
      kernel === this._kernel &&
      this._comm &&
      !this._comm.isDisposed
    ) {
      return;
    }
    const request = ++this._request;
    this._disconnect();
    if (!kernel) {
      return;
    }

    try {
      // The logical kernel ID is retained when process count changes, while
      // `kernel.info` may still contain the pre-restart implementation.
      const infoReply = await kernel.requestKernelInfo();
      const implementation =
        infoReply && 'implementation' in infoReply.content
          ? infoReply.content.implementation
          : null;
      if (
        request !== this._request ||
        this._disposed ||
        kernel !== this._panel.sessionContext.session?.kernel ||
        implementation !== 'jupyter_distributed'
      ) {
        return;
      }
      this._kernel = kernel;
      const comm = kernel.createComm(RANK_UPDATE_COMM_TARGET);
      comm.onMsg = this._onMessage;
      comm.onClose = () => {
        if (this._comm === comm) {
          this._comm = null;
        }
      };
      this._comm = comm;
      void comm.open({}).done.catch(error => {
        console.warn('Unable to open distributed rank update comm', error);
      });
    } catch (error) {
      console.warn('Unable to connect distributed rank updates', error);
    }
  }

  private _disconnect(): void {
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

  private _onKernelChanged = (): void => {
    void this._connect(true);
  };

  private _onConnectionStatusChanged = (): void => {
    if (
      this._panel.sessionContext.session?.kernel?.connectionStatus ===
      'connected'
    ) {
      void this._connect(true);
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
  private _disposed = false;
  private _kernel: Kernel.IKernelConnection | null = null;
  private _panel: NotebookPanel;
  private _request = 0;
  private _updates: RankUpdateModel;
}

namespace Private {
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
