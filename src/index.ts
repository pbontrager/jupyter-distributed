import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import {
  IDebugger,
  IDebuggerSidebar
} from '@jupyterlab/debugger';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';

import {
  MIME_TYPE,
  RankOutputRenderer,
  RankSelectionModel
} from './outputRenderer';
import { DebuggerRankSelector } from './debuggerRank';
import { ProcessToolbarExtension } from './toolbar';

const notebookPlugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter-distributed:notebook',
  description: 'Jupyter Distributed process controls and rank-aware output',
  autoStart: true,
  requires: [IRenderMimeRegistry, INotebookTracker],
  activate: (
    app: JupyterFrontEnd,
    rendermime: IRenderMimeRegistry,
    notebooks: INotebookTracker
  ): void => {
    const rankSelections = new RankSelectionModel();

    app.docRegistry.addWidgetExtension(
      'Notebook',
      new ProcessToolbarExtension()
    );

    rendermime.addFactory(
      Private.rendererFactory(rendermime, rankSelections),
      0
    );

    const registerNotebookRenderer = (panel: NotebookPanel): void => {
      const contextual = panel.content.rendermime;
      contextual.removeMimeType(MIME_TYPE);
      contextual.addFactory(
        Private.rendererFactory(contextual, rankSelections),
        0
      );
    };
    notebooks.forEach(registerNotebookRenderer);
    notebooks.widgetAdded.connect((_sender, panel) => {
      registerNotebookRenderer(panel);
    });
  }
};

const debuggerPlugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter-distributed:debugger',
  description: 'Jupyter Distributed rank selection for the JupyterLab debugger',
  autoStart: true,
  optional: [IDebugger, IDebuggerSidebar],
  activate: (
    app: JupyterFrontEnd,
    debuggerService: IDebugger | null,
    debuggerSidebar: IDebugger.ISidebar | null
  ): void => {
    if (
      app.name !== 'JupyterLab' ||
      debuggerService === null ||
      debuggerSidebar === null
    ) {
      return;
    }
    debuggerSidebar.insertWidget(
      0,
      new DebuggerRankSelector(debuggerService)
    );
  }
};

export default [notebookPlugin, debuggerPlugin];

namespace Private {
  export function rendererFactory(
    rendermime: IRenderMimeRegistry,
    selections: RankSelectionModel
  ): IRenderMime.IRendererFactory {
    return {
      mimeTypes: [MIME_TYPE],
      safe: true,
      createRenderer: options =>
        new RankOutputRenderer(options, rendermime, selections)
    };
  }
}
