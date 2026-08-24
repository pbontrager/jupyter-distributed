import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';

import {
  MIME_TYPE,
  RankOutputRenderer,
  RankSelectionModel
} from './outputRenderer';
import { ProcessToolbarExtension } from './toolbar';

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter-distributed:plugin',
  description: 'Jupyter Distributed process controls and rank-aware output',
  autoStart: true,
  requires: [IRenderMimeRegistry],
  activate: (
    app: JupyterFrontEnd,
    rendermime: IRenderMimeRegistry
  ): void => {
    const rankSelections = new RankSelectionModel();

    app.docRegistry.addWidgetExtension(
      'Notebook',
      new ProcessToolbarExtension()
    );

    const rendererFactory: IRenderMime.IRendererFactory = {
      mimeTypes: [MIME_TYPE],
      safe: true,
      createRenderer: options =>
        new RankOutputRenderer(options, rendermime, rankSelections)
    };
    rendermime.addFactory(rendererFactory, 0);
  }
};

export default plugin;
