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

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter-distributed:plugin',
  description: 'Jupyter Distributed process controls and rank-aware output',
  autoStart: true,
  requires: [
    IRenderMimeRegistry,
    IDebugger,
    IDebuggerSidebar,
    INotebookTracker
  ],
  activate: (
    app: JupyterFrontEnd,
    rendermime: IRenderMimeRegistry,
    debuggerService: IDebugger,
    debuggerSidebar: IDebugger.ISidebar,
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

    debuggerSidebar.insertWidget(
      0,
      new DebuggerRankSelector(debuggerService)
    );
  }
};

export default plugin;

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
