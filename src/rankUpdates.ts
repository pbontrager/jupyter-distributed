import { NotebookPanel } from '@jupyterlab/notebook';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { JSONObject } from '@lumino/coreutils';
import { IDisposable } from '@lumino/disposable';
import { Signal } from '@lumino/signaling';

import { RANK_UPDATE_COMM_TARGET } from './constants';

export interface RankUpdateChange {
  executionId: string;
  payload: Record<string, unknown>;
  changedRanks: number[];
  rankUpdates: RankPatchUpdate[] | null;
}

export interface RankPatchUpdate {
  rank: number;
  patches: Record<string, unknown>[];
}

interface ExecutionState {
  sequence: number;
  payload: Record<string, unknown>;
}

interface ApplyResult {
  needsSnapshot: boolean;
}

/** Reduces ordered rank patches into recoverable execution snapshots. */
export class RankUpdateModel {
  readonly changed = new Signal<this, RankUpdateChange>(this);

  seed(value: unknown): Record<string, unknown> | null {
    const payload = Private.normalizePayload(value);
    if (!payload) {
      return null;
    }
    const id = executionId(payload)!;
    const existing = this._executions.get(id);
    if (!existing || payload.status !== 'busy') {
      this._store(id, { sequence: existing?.sequence ?? 0, payload });
      return payload;
    }
    return existing.payload;
  }

  apply(sequence: number, kind: unknown, value: unknown): ApplyResult {
    if (!Number.isSafeInteger(sequence) || sequence < 0) {
      return { needsSnapshot: true };
    }
    if (kind === 'snapshot') {
      const payload = Private.normalizePayload(value);
      if (!payload) {
        return { needsSnapshot: true };
      }
      const id = executionId(payload)!;
      const existing = this._executions.get(id);
      if (existing && sequence <= existing.sequence) {
        return { needsSnapshot: false };
      }
      this._store(id, { sequence, payload });
      this.changed.emit({
        executionId: id,
        payload,
        changedRanks: Private.rankNumbers(payload),
        rankUpdates: null
      });
      return { needsSnapshot: false };
    }
    if (kind !== 'patch') {
      return { needsSnapshot: true };
    }

    const patch = Private.normalizePatch(value);
    if (!patch) {
      return { needsSnapshot: true };
    }
    const id = executionId(patch)!;
    const existing = this._executions.get(id);
    if (existing && sequence <= existing.sequence) {
      return { needsSnapshot: false };
    }
    if (!existing || sequence !== existing.sequence + 1) {
      return { needsSnapshot: true };
    }
    const applied = Private.applyPatch(existing.payload, patch);
    if (!applied) {
      return { needsSnapshot: true };
    }
    this._store(id, { sequence, payload: applied.payload });
    this.changed.emit({
      executionId: id,
      payload: applied.payload,
      changedRanks: applied.changedRanks,
      rankUpdates: applied.rankUpdates
    });
    return { needsSnapshot: false };
  }

  private _store(executionId: string, state: ExecutionState): void {
    this._executions.delete(executionId);
    this._executions.set(executionId, state);
    while (this._executions.size > 16) {
      const oldest = this._executions.keys().next().value;
      if (oldest === undefined) {
        break;
      }
      this._executions.delete(oldest);
    }
  }

  private _executions = new Map<string, ExecutionState>();
}

/** Owns the sequenced rank-update comm for one notebook connection. */
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
      this._onMessage(comm, message);
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

  private _send(comm: Kernel.IComm, data: JSONObject): void {
    if (comm.isDisposed) {
      return;
    }
    try {
      void comm.send(data).done.catch(() => undefined);
    } catch {
      // Reconnection will recover from the kernel's latest snapshot.
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

  private _onMessage(
    comm: Kernel.IComm,
    message: KernelMessage.ICommMsgMsg
  ): void {
    const data = message.content.data as Record<string, unknown>;
    if (data.method === 'update') {
      const sequence = Number(data.sequence);
      const result = this._updates.apply(sequence, data.kind, data.payload);
      if (result.needsSnapshot) {
        this._send(comm, { method: 'request_snapshots' });
      }
      return;
    }
    if (data.method === 'snapshots' && Array.isArray(data.snapshots)) {
      for (const snapshot of data.snapshots) {
        if (!Private.isRecord(snapshot)) {
          continue;
        }
        this._updates.apply(
          Number(snapshot.sequence),
          'snapshot',
          snapshot.payload
        );
      }
    }
  }

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
  if (!Private.isRecord(value)) {
    return null;
  }
  const id = value.execution_id;
  return typeof id === 'string' && id ? id : null;
}

namespace Private {
  interface AppliedPatch {
    payload: Record<string, unknown>;
    changedRanks: number[];
    rankUpdates: RankPatchUpdate[];
  }

  export function delay(milliseconds: number): Promise<void> {
    return new Promise(resolve => window.setTimeout(resolve, milliseconds));
  }

  export function isRecord(value: unknown): value is Record<string, unknown> {
    return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
  }

  export function normalizePayload(
    value: unknown
  ): Record<string, unknown> | null {
    if (!isRecord(value) || !executionId(value) || !Array.isArray(value.ranks)) {
      return null;
    }
    return { ...value, ranks: value.ranks.map(rank => cloneRecord(rank)) };
  }

  export function normalizePatch(
    value: unknown
  ): Record<string, unknown> | null {
    if (
      !isRecord(value) ||
      !executionId(value) ||
      !Array.isArray(value.rank_updates)
    ) {
      return null;
    }
    return value;
  }

  export function rankNumbers(payload: Record<string, unknown>): number[] {
    return Array.isArray(payload.ranks)
      ? payload.ranks
          .map(rank => (isRecord(rank) ? Number(rank.rank) : NaN))
          .filter(rank => Number.isSafeInteger(rank) && rank >= 0)
      : [];
  }

  export function applyPatch(
    previous: Record<string, unknown>,
    patch: Record<string, unknown>
  ): AppliedPatch | null {
    if (executionId(previous) !== executionId(patch)) {
      return null;
    }
    const ranks = new Map<number, Record<string, unknown>>();
    if (!Array.isArray(previous.ranks)) {
      return null;
    }
    for (const value of previous.ranks) {
      if (!isRecord(value) || !Number.isSafeInteger(Number(value.rank))) {
        return null;
      }
      ranks.set(Number(value.rank), cloneRank(value));
    }

    const changedRanks: number[] = [];
    const rankUpdates: RankPatchUpdate[] = [];
    for (const value of patch.rank_updates as unknown[]) {
      if (!isRecord(value) || !Number.isSafeInteger(Number(value.rank))) {
        return null;
      }
      const rankNumber = Number(value.rank);
      const rank = ranks.get(rankNumber);
      if (!rank || !Array.isArray(rank.outputs) || !Array.isArray(value.patches)) {
        return null;
      }
      const outputs = [...rank.outputs];
      for (const operation of value.patches) {
        if (!isRecord(operation) || !applyOutputPatch(outputs, operation)) {
          return null;
        }
      }
      ranks.set(rankNumber, { ...rank, outputs });
      changedRanks.push(rankNumber);
      rankUpdates.push({
        rank: rankNumber,
        patches: value.patches.filter(isRecord)
      });
    }

    const payload = {
      ...previous,
      ...Object.fromEntries(
        Object.entries(patch).filter(([key]) => key !== 'rank_updates')
      ),
      ranks: [...ranks.values()].sort(
        (left, right) => Number(left.rank) - Number(right.rank)
      )
    };
    return { payload, changedRanks, rankUpdates };
  }

  function applyOutputPatch(
    outputs: unknown[],
    patch: Record<string, unknown>
  ): boolean {
    if (patch.kind === 'append_output' && isRecord(patch.output)) {
      outputs.push(cloneRecord(patch.output));
      return true;
    }
    const index = Number(patch.index);
    if (patch.kind === 'append_stream') {
      if (
        !Number.isSafeInteger(index) ||
        index < 0 ||
        index >= outputs.length ||
        typeof patch.text !== 'string'
      ) {
        return false;
      }
      const output = outputs[index];
      if (!isRecord(output) || output.type !== 'stream' || !isRecord(output.content)) {
        return false;
      }
      const previousText = output.content.text;
      const nextText = Array.isArray(previousText)
        ? previousText
        : [asText(previousText)];
      nextText.push(patch.text);
      outputs[index] = {
        ...output,
        content: { ...output.content, text: nextText }
      };
      return true;
    }
    if (patch.kind === 'replace_output') {
      if (
        !Number.isSafeInteger(index) ||
        index < 0 ||
        index >= outputs.length ||
        !isRecord(patch.output)
      ) {
        return false;
      }
      outputs[index] = cloneRecord(patch.output);
      return true;
    }
    if (patch.kind === 'truncate') {
      const length = Number(patch.length);
      if (!Number.isSafeInteger(length) || length < 0 || length > outputs.length) {
        return false;
      }
      outputs.splice(length);
      return true;
    }
    return false;
  }

  function cloneRank(value: Record<string, unknown>): Record<string, unknown> {
    return {
      ...value,
      outputs: Array.isArray(value.outputs)
        ? value.outputs.map(output => cloneRecord(output))
        : []
    };
  }

  function cloneRecord(value: unknown): Record<string, unknown> {
    return isRecord(value) ? { ...value } : {};
  }

  function asText(value: unknown): string {
    return Array.isArray(value)
      ? value.map(item => String(item)).join('')
      : String(value ?? '');
  }
}
