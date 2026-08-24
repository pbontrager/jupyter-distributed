import { Dialog, showDialog, showErrorMessage } from '@jupyterlab/apputils';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import { INotebookModel, NotebookPanel } from '@jupyterlab/notebook';
import { DisposableDelegate, IDisposable } from '@lumino/disposable';
import { Widget } from '@lumino/widgets';

const WORLD_SIZES = [1, 2, 4, 8] as const;

/** World-size choices are scoped to the logical Jupyter session. */
export class WorldSizeState {
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
    this.addClass('jp-SpmdProcessSelector');
    this._panel = panel;
    this._state = state;
    this._select = this.node.querySelector('select')!;

    this._select.addEventListener('change', this._onChange);
    panel.sessionContext.sessionChanged.connect(this._sync, this);
    this._sync();
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this._select.removeEventListener('change', this._onChange);
    this._panel.sessionContext.sessionChanged.disconnect(this._sync, this);
    super.dispose();
  }

  private _sessionId(): string | undefined {
    return this._panel.sessionContext.session?.id;
  }

  private _sync(): void {
    this._select.value = String(this._state.get(this._sessionId()));
  }

  private _onChange = async (): Promise<void> => {
    const previous = this._state.get(this._sessionId());
    const next = Number(this._select.value);
    this._select.value = String(previous);

    if (next === previous) {
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
    if (!kernel) {
      await showErrorMessage(
        'Unable to change process count',
        'This notebook is not connected to a kernel.'
      );
      return;
    }

    this._select.disabled = true;
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
      this._state.set(this._sessionId(), next);
      this._select.value = String(next);
    } catch (error) {
      await showErrorMessage(
        'Unable to change process count',
        error instanceof Error ? error : String(error)
      );
      this._select.value = String(previous);
    } finally {
      this._select.disabled = false;
    }
  };

  private _panel: NotebookPanel;
  private _select: HTMLSelectElement;
  private _state: WorldSizeState;
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
    if (!panel.toolbar.insertAfter('kernelName', 'spmd-processes', selector)) {
      panel.toolbar.addItem('spmd-processes', selector);
    }
    return new DisposableDelegate(() => selector.dispose());
  }

  private _state: WorldSizeState;
}

namespace Private {
  export function createNode(): HTMLElement {
    const wrapper = document.createElement('label');
    const text = document.createElement('span');
    const select = document.createElement('select');

    text.textContent = 'Processes:';
    text.className = 'jp-SpmdProcessSelector-label';
    select.className = 'jp-SpmdProcessSelector-select';
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
