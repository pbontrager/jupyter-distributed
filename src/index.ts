import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';

import {
  MIME_TYPE,
  RankOutputRenderer,
  RankSelectionModel
} from './outputRenderer';
import { ProcessToolbarExtension, WorldSizeState } from './toolbar';

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'spmd-jupyter:plugin',
  description: 'SPMD process controls and rank-aware notebook output',
  autoStart: true,
  requires: [INotebookTracker, IRenderMimeRegistry],
  activate: (
    app: JupyterFrontEnd,
    notebooks: INotebookTracker,
    rendermime: IRenderMimeRegistry
  ): void => {
    const worldSizes = new WorldSizeState();
    const rankSelections = new RankSelectionModel();

    app.docRegistry.addWidgetExtension(
      'Notebook',
      new ProcessToolbarExtension(worldSizes)
    );

    const rendererFactory: IRenderMime.IRendererFactory = {
      mimeTypes: [MIME_TYPE],
      safe: true,
      createRenderer: options =>
        new RankOutputRenderer(
          options,
          rendermime,
          notebooks.currentWidget,
          rankSelections
        )
    };
    rendermime.addFactory(rendererFactory, 0);
  }
};

export default plugin;
