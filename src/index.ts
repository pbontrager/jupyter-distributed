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
import { RankUpdateComm, RankUpdateModel } from './rankUpdates';
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
    const rankUpdates = new RankUpdateModel();
    const updateComms = new WeakMap<NotebookPanel, RankUpdateComm>();
    const processControls = new ProcessToolbarExtension();

    app.docRegistry.addWidgetExtension(
      'Notebook',
      processControls
    );

    app.commands.addCommand('jupyter-distributed:set-processes', {
      label: 'Set Distributed Processes',
      caption: 'Set the process count and restart the selected notebook kernel',
      describedBy: {
        args: {
          type: 'object',
          properties: {
            processes: {
              type: 'integer',
              minimum: 1,
              description: 'Number of persistent notebook processes'
            },
            notebookPath: {
              type: 'string',
              description: 'Notebook path relative to the Jupyter server root'
            }
          },
          required: ['processes']
        }
      },
      execute: async args => {
        const processes = Number(args.processes);
        if (!Number.isSafeInteger(processes) || processes < 1) {
          throw new Error('Processes must be a positive integer.');
        }
        const requestedPath =
          typeof args.notebookPath === 'string'
            ? args.notebookPath.replace(/^\/+/, '')
            : null;
        const panel = requestedPath
          ? (notebooks.find(
              candidate => candidate.context.path === requestedPath
            ) ?? null)
          : notebooks.currentWidget;
        if (!panel) {
          throw new Error(
            requestedPath
              ? `Notebook is not open in JupyterLab: ${requestedPath}`
              : 'No active notebook was found.'
          );
        }
        return processControls.setWorldSize(panel, processes);
      }
    });

    rendermime.addFactory(
      Private.rendererFactory(rendermime, rankSelections, rankUpdates),
      0
    );

    const registerNotebookRenderer = (panel: NotebookPanel): void => {
      const contextual = panel.content.rendermime;
      contextual.removeMimeType(MIME_TYPE);
      contextual.addFactory(
        Private.rendererFactory(contextual, rankSelections, rankUpdates),
        0
      );
      if (!updateComms.has(panel)) {
        const comm = new RankUpdateComm(panel, rankUpdates);
        updateComms.set(panel, comm);
        panel.disposed.connect(() => comm.dispose());
      }
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
    const rankSelector = new DebuggerRankSelector(debuggerService);
    debuggerSidebar.insertWidget(0, rankSelector);
    app.commands.addCommand('jupyter-distributed:select-debug-rank', {
      label: 'Select Distributed Debugger Rank',
      caption: 'Select which stopped rank supplies the JupyterLab debugger views',
      describedBy: {
        args: {
          type: 'object',
          properties: {
            rank: {
              type: 'integer',
              minimum: 0,
              description: 'Stopped distributed rank to inspect'
            }
          },
          required: ['rank']
        }
      },
      execute: async args => rankSelector.selectRank(Number(args.rank))
    });
  }
};

export default [notebookPlugin, debuggerPlugin];

namespace Private {
  export function rendererFactory(
    rendermime: IRenderMimeRegistry,
    selections: RankSelectionModel,
    updates: RankUpdateModel
  ): IRenderMime.IRendererFactory {
    return {
      mimeTypes: [MIME_TYPE],
      safe: true,
      createRenderer: options =>
        new RankOutputRenderer(options, rendermime, selections, updates)
    };
  }
}
