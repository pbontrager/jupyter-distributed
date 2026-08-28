import { Dialog, showDialog, showErrorMessage } from '@jupyterlab/apputils';
import { URLExt } from '@jupyterlab/coreutils';
import { DocumentRegistry } from '@jupyterlab/docregistry';
import { INotebookModel, NotebookPanel } from '@jupyterlab/notebook';
import { ServerConnection } from '@jupyterlab/services';
import { DisposableDelegate, IDisposable } from '@lumino/disposable';
import { Widget } from '@lumino/widgets';

const METADATA_KEY = 'jupyter_distributed';
const SINGLE_PROCESS_RANK_MAGIC_BOOTSTRAP = `
from jupyter_distributed.rank_magic import register_single_process_rank_magic as _jd_register_rank_magic
_jd_register_rank_magic(get_ipython())
del _jd_register_rank_magic
`.trim();

interface DistributedKernelModel {
  kernel_id: string;
  kernel_name: string;
  world_size: number;
  distributed: boolean;
}

class ProcessSelector extends Widget {
  constructor(panel: NotebookPanel) {
    super({ node: Private.createNode() });
    this.addClass('jp-JupyterDistributedProcessSelector');
    this._panel = panel;
    this._input = this.node.querySelector('input')!;

    this._input.addEventListener('change', this._onChange);
    this._input.addEventListener('keydown', this._onKeyDown);
    panel.sessionContext.sessionChanged.connect(this._onKernelChanged, this);
    panel.sessionContext.kernelChanged.connect(this._onKernelChanged, this);
    this._syncVisibility();
    void this._initialize();
  }

  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this._input.removeEventListener('change', this._onChange);
    this._input.removeEventListener('keydown', this._onKeyDown);
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

  private _kernelId(): string | undefined {
    return this._panel.sessionContext.session?.kernel?.id;
  }

  private _syncVisibility(): void {
    const connected = this._kernelId() !== undefined;
    this.setHidden(!connected);
    this._input.disabled = !connected || this._pending;
  }

  private _onKernelChanged = (): void => {
    this._syncVisibility();
    if (this._ready && !this._pending) {
      void this._syncFromServer();
    }
  };

  private async _initialize(): Promise<void> {
    await Promise.all([
      this._panel.context.ready,
      this._panel.sessionContext.ready
    ]);
    if (this.isDisposed) {
      return;
    }
    this._ready = true;
    this._syncVisibility();
    await this._syncFromServer();
  }

  private async _syncFromServer(): Promise<void> {
    const kernelId = this._kernelId();
    const request = ++this._syncRequest;
    let restoring = false;
    if (!kernelId) {
      return;
    }

    try {
      const model = await this._request(kernelId, 'GET');
      if (request !== this._syncRequest || kernelId !== this._kernelId()) {
        return;
      }
      const savedWorldSize = this._savedWorldSize();
      if (
        savedWorldSize === undefined ||
        savedWorldSize === model.world_size
      ) {
        this._setWorldSize(model.world_size);
        await this._ensureSingleProcessRankMagic(model.world_size);
        return;
      }

      this._pending = true;
      restoring = true;
      this._syncVisibility();
      const restored = await this._request(kernelId, 'POST', {
        world_size: savedWorldSize
      });
      if (kernelId === this._kernelId()) {
        this._setWorldSize(restored.world_size);
        await this._ensureSingleProcessRankMagic(restored.world_size);
      }
    } catch (error) {
      // Keep the default value if the server extension is unavailable.
      if (restoring) {
        await showErrorMessage(
          'Unable to restore process count',
          error instanceof Error ? error : String(error)
        );
      }
    } finally {
      if (restoring) {
        this._pending = false;
        this._syncVisibility();
        if (kernelId !== this._kernelId()) {
          void this._syncFromServer();
        }
      }
    }
  }

  private _onChange = async (): Promise<void> => {
    const kernelId = this._kernelId();
    const previous = Number(this._input.dataset.worldSize ?? '1');
    const rawValue = this._input.value.trim();
    const next = Number(rawValue);
    this._input.value = String(previous);

    if (!/^[1-9]\d*$/.test(rawValue) || !Number.isSafeInteger(next)) {
      await showErrorMessage(
        'Invalid process count',
        'Processes must be a positive integer.'
      );
      return;
    }

    if (!kernelId) {
      return;
    }
    if (next === previous) {
      this._saveWorldSize(next);
      await this._ensureSingleProcessRankMagic(next);
      return;
    }

    const result = await showDialog({
      title: 'Restart kernel processes?',
      body:
        `Changing the process count from ${previous} to ${next} will restart ` +
        'the selected kernel. All in-memory state will be lost.',
      buttons: [
        Dialog.cancelButton(),
        Dialog.warnButton({ label: `Restart with ${next} processes` })
      ]
    });

    if (!result.button.accept) {
      return;
    }

    this._pending = true;
    this._syncVisibility();
    try {
      const model = await this._request(kernelId, 'POST', {
        world_size: next
      });
      if (kernelId === this._kernelId()) {
        this._setWorldSize(model.world_size);
        this._saveWorldSize(model.world_size);
        await this._ensureSingleProcessRankMagic(model.world_size);
      }
    } catch (error) {
      await showErrorMessage(
        'Unable to change process count',
        error instanceof Error ? error : String(error)
      );
      this._setWorldSize(previous);
    } finally {
      this._pending = false;
      this._syncVisibility();
      if (kernelId !== this._kernelId()) {
        void this._syncFromServer();
      }
    }
  };

  private _onKeyDown = (event: KeyboardEvent): void => {
    if (event.key === 'Enter') {
      event.preventDefault();
      this._input.blur();
    }
  };

  private _setWorldSize(worldSize: number): void {
    const value =
      Number.isSafeInteger(worldSize) && worldSize > 0 ? worldSize : 1;
    this._input.value = String(value);
    this._input.dataset.worldSize = String(value);
  }

  private _savedWorldSize(): number | undefined {
    const metadata = this._panel.context.model.getMetadata(METADATA_KEY);
    if (Private.isRecord(metadata)) {
      const worldSize = metadata.world_size;
      if (Number.isSafeInteger(worldSize) && Number(worldSize) > 0) {
        return Number(worldSize);
      }
    }
    return undefined;
  }

  private _saveWorldSize(worldSize: number): void {
    const model = this._panel.context.model;
    if (worldSize === 1) {
      model.deleteMetadata(METADATA_KEY);
    } else {
      model.setMetadata(METADATA_KEY, { world_size: worldSize });
    }
  }

  private async _ensureSingleProcessRankMagic(worldSize: number): Promise<void> {
    if (worldSize !== 1) {
      return;
    }
    const kernel = this._panel.sessionContext.session?.kernel;
    if (!kernel) {
      return;
    }
    try {
      const info = await kernel.info;
      if (
        kernel !== this._panel.sessionContext.session?.kernel ||
        !info.language_info.name.toLowerCase().startsWith('python')
      ) {
        return;
      }
      const future = kernel.requestExecute({
        code: SINGLE_PROCESS_RANK_MAGIC_BOOTSTRAP,
        silent: true,
        store_history: false,
        allow_stdin: false,
        stop_on_error: false
      });
      await future.done;
    } catch {
      // The compatibility magic is optional for non-IPython Python kernels.
    }
  }

  private async _request(
    kernelId: string,
    method: 'GET' | 'POST',
    body?: { world_size: number }
  ): Promise<DistributedKernelModel> {
    const settings = ServerConnection.makeSettings();
    const url = URLExt.join(
      settings.baseUrl,
      'jupyter-distributed',
      'kernels',
      encodeURIComponent(kernelId)
    );
    const response = await ServerConnection.makeRequest(
      url,
      {
        method,
        headers: body ? { 'Content-Type': 'application/json' } : undefined,
        body: body ? JSON.stringify(body) : undefined
      },
      settings
    );
    if (!response.ok) {
      const message = (await response.text()) || response.statusText;
      throw new Error(message);
    }
    return (await response.json()) as DistributedKernelModel;
  }

  private _panel: NotebookPanel;
  private _pending = false;
  private _ready = false;
  private _input: HTMLInputElement;
  private _syncRequest = 0;
}

export class ProcessToolbarExtension
  implements DocumentRegistry.IWidgetExtension<NotebookPanel, INotebookModel>
{
  createNew(
    panel: NotebookPanel,
    context: DocumentRegistry.IContext<INotebookModel>
  ): IDisposable {
    const selector = new ProcessSelector(panel);
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
}

namespace Private {
  export function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
  }

  export function createNode(): HTMLElement {
    const wrapper = document.createElement('label');
    const text = document.createElement('span');
    const input = document.createElement('input');

    text.textContent = 'Processes:';
    text.className = 'jp-JupyterDistributedProcessSelector-label';
    input.className = 'jp-JupyterDistributedProcessSelector-input';
    input.type = 'text';
    input.inputMode = 'numeric';
    input.pattern = '[1-9][0-9]*';
    input.value = '1';
    input.dataset.worldSize = '1';
    input.setAttribute('aria-label', 'Kernel process count');
    input.setAttribute('autocomplete', 'off');
    wrapper.append(text, input);
    return wrapper;
  }
}
