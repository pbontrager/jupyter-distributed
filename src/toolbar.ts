import { Dialog, showDialog, showErrorMessage } from '@jupyterlab/apputils';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import { INotebookModel, NotebookPanel } from '@jupyterlab/notebook';
import { DisposableDelegate, IDisposable } from '@lumino/disposable';
import { Widget } from '@lumino/widgets';

const WORLD_SIZES = [1, 2, 4, 8] as const;
const WORLD_SIZE_EXPRESSION = 'jupyter_distributed_world_size';

/** World-size choices are scoped to the logical Jupyter session. */
export class WorldSizeState {
  has(sessionId: string | undefined): boolean {
    return sessionId ? this._values.has(sessionId) : false;
  }

  get(sessionId: string | undefined): number {
    return sessionId ? (this._values.get(sessionId) ?? 1) : 1;
  }

  set(sessionId: string | undefined, value: number): void {
    if (sessionId) {
      this._values.set(sessionId, value);
    }
  }

  private _values = new Map<string, number>();
}

class ProcessSelector extends Widget {
  constructor(panel: NotebookPanel, state: WorldSizeState) {
    super({ node: Private.createNode() });
    this.addClass('jp-JupyterDistributedProcessSelector');
    this._panel = panel;
    this._state = state;
    this._select = this.node.querySelector('select')!;

    this._select.addEventListener('change', this._onChange);
    panel.sessionContext.sessionChanged.connect(this._onKernelChanged, this);
    panel.sessionContext.kernelChanged.connect(this._onKernelChanged, this);
    this._syncLocal();
    void this._syncFromKernel();
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this._select.removeEventListener('change', this._onChange);
    this._panel.sessionContext.sessionChanged.disconnect(
      this._onKernelChanged,
      this
    );
    this._panel.sessionContext.kernelChanged.disconnect(
      this._onKernelChanged,
      this
    );
    super.dispose();
  }

  private _sessionId(): string | undefined {
    return this._panel.sessionContext.session?.id;
  }

  private _syncLocal(): void {
    const isDistributed = this._isDistributedKernel();
    this.setHidden(!isDistributed);
    this._select.disabled = !isDistributed;
    this._select.value = String(this._state.get(this._sessionId()));
  }

  private _isDistributedKernel(): boolean {
    return (
      this._panel.sessionContext.session?.kernel?.name ===
      'jupyter-distributed'
    );
  }

  private _onKernelChanged = (): void => {
    if (this._restoring?.sessionId !== this._sessionId()) {
      this._restoring = null;
    }
    this._syncLocal();
    void this._syncFromKernel();
  };

  private async _syncFromKernel(): Promise<void> {
    const kernel = this._panel.sessionContext.session?.kernel;
    const sessionId = this._sessionId();
    const request = ++this._syncRequest;
    if (!kernel || !sessionId || !this._isDistributedKernel()) {
      return;
    }

    try {
      const future = kernel.requestExecute({
        code: '%spmd_world_size',
        silent: true,
        store_history: false,
        user_expressions: { [WORLD_SIZE_EXPRESSION]: 'None' },
        allow_stdin: false,
        stop_on_error: false
      });
      const reply = await future.done;
      if (
        request !== this._syncRequest ||
        kernel !== this._panel.sessionContext.session?.kernel ||
        sessionId !== this._sessionId() ||
        reply.content.status !== 'ok'
      ) {
        return;
      }

      const expression = reply.content.user_expressions[WORLD_SIZE_EXPRESSION];
      const value = Private.worldSizeFromExpression(expression);
      if (value === null) {
        return;
      }

      if (!this._state.has(sessionId)) {
        this._state.set(sessionId, value);
        this._select.value = String(value);
        this._restoring = null;
        return;
      }

      const remembered = this._state.get(sessionId);
      this._select.value = String(remembered);
      if (value === remembered) {
        this._restoring = null;
        return;
      }

      await this._restoreRememberedWorldSize(kernel, sessionId, remembered);
    } catch {
      // Kernels without the query magic retain the local/default selection.
    }
  }

  private async _restoreRememberedWorldSize(
    kernel: NonNullable<
      NonNullable<NotebookPanel['sessionContext']['session']>['kernel']
    >,
    sessionId: string,
    worldSize: number
  ): Promise<void> {
    if (
      this._restoring?.sessionId === sessionId &&
      this._restoring.worldSize === worldSize
    ) {
      return;
    }

    this._restoring = { sessionId, worldSize };
    try {
      const future = kernel.requestExecute({
        code: `%spmd_world_size ${worldSize}`,
        silent: true,
        store_history: false,
        allow_stdin: false,
        stop_on_error: true
      });
      const reply = await future.done;
      if (reply.content.status !== 'ok') {
        throw new Error('The kernel rejected the remembered process count.');
      }
      if (
        this._restoring?.sessionId === sessionId &&
        this._restoring.worldSize === worldSize
      ) {
        this._restoring = null;
      }
    } catch {
      if (
        this._restoring?.sessionId === sessionId &&
        this._restoring.worldSize === worldSize
      ) {
        this._restoring = null;
      }
    }
  }

  private _onChange = async (): Promise<void> => {
    const sessionId = this._sessionId();
    const previous = this._state.get(sessionId);
    const next = Number(this._select.value);
    this._select.value = String(previous);

    if (!this._isDistributedKernel() || next === previous) {
      return;
    }

    const result = await showDialog({
      title: 'Restart distributed kernel?',
      body:
        `Changing the process count from ${previous} to ${next} will restart ` +
        'this notebook kernel. All in-memory state will be lost.',
      buttons: [
        Dialog.cancelButton(),
        Dialog.warnButton({ label: `Restart with ${next} processes` })
      ]
    });

    if (!result.button.accept) {
      return;
    }

    const kernel = this._panel.sessionContext.session?.kernel;
    if (
      !kernel ||
      kernel.name !== 'jupyter-distributed' ||
      !sessionId ||
      sessionId !== this._sessionId()
    ) {
      await showErrorMessage(
        'Unable to change process count',
        'This notebook is not connected to a Jupyter Distributed kernel.'
      );
      return;
    }

    this._select.disabled = true;
    this._state.set(sessionId, next);
    this._restoring = { sessionId, worldSize: next };
    this._select.value = String(next);
    try {
      const future = kernel.requestExecute({
        code: `%spmd_world_size ${next}`,
        silent: true,
        store_history: false,
        allow_stdin: false,
        stop_on_error: true
      });
      const reply = await future.done;
      if (reply.content.status !== 'ok') {
        const content = reply.content as typeof reply.content & {
          ename?: string;
          evalue?: string;
        };
        throw new Error(
          [content.ename, content.evalue].filter(Boolean).join(': ') ||
            'The kernel rejected the process-count change.'
        );
      }
      if (
        this._restoring?.sessionId === sessionId &&
        this._restoring.worldSize === next
      ) {
        this._restoring = null;
      }
    } catch (error) {
      this._state.set(sessionId, previous);
      this._restoring = null;
      await showErrorMessage(
        'Unable to change process count',
        error instanceof Error ? error : String(error)
      );
      if (sessionId === this._sessionId()) {
        this._select.value = String(previous);
      }
    } finally {
      this._select.disabled = false;
    }
  };

  private _panel: NotebookPanel;
  private _restoring: { sessionId: string; worldSize: number } | null = null;
  private _select: HTMLSelectElement;
  private _state: WorldSizeState;
  private _syncRequest = 0;
}

export class ProcessToolbarExtension
  implements DocumentRegistry.IWidgetExtension<NotebookPanel, INotebookModel>
{
  constructor(state: WorldSizeState) {
    this._state = state;
  }

  createNew(
    panel: NotebookPanel,
    context: DocumentRegistry.IContext<INotebookModel>
  ): IDisposable {
    const selector = new ProcessSelector(panel, this._state);
    if (
      !panel.toolbar.insertAfter(
        'kernelName',
        'jupyter-distributed-processes',
        selector
      )
    ) {
      panel.toolbar.addItem('jupyter-distributed-processes', selector);
    }
    return new DisposableDelegate(() => selector.dispose());
  }

  private _state: WorldSizeState;
}

namespace Private {
  export function worldSizeFromExpression(value: unknown): number | null {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return null;
    }
    const data = (value as { data?: unknown }).data;
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
      return null;
    }
    const text = (data as Record<string, unknown>)['text/plain'];
    if (typeof text !== 'string') {
      return null;
    }
    const parsed = Number(text.replace(/^(['"])(.*)\1$/, '$2'));
    return WORLD_SIZES.some(size => size === parsed) ? parsed : null;
  }

  export function createNode(): HTMLElement {
    const wrapper = document.createElement('label');
    const text = document.createElement('span');
    const select = document.createElement('select');

    text.textContent = 'Processes:';
    text.className = 'jp-JupyterDistributedProcessSelector-label';
    select.className = 'jp-JupyterDistributedProcessSelector-select';
    select.setAttribute('aria-label', 'Kernel process count');
    for (const size of WORLD_SIZES) {
      const option = document.createElement('option');
      option.value = String(size);
      option.textContent = String(size);
      select.appendChild(option);
    }
    wrapper.append(text, select);
    return wrapper;
  }
}
