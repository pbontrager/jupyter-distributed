import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { IToolbarWidgetRegistry } from '@jupyterlab/apputils';
import {
  IDebugger,
  IDebuggerSidebar
} from '@jupyterlab/debugger';
import { isCodeCellModel } from '@jupyterlab/cells';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { IRenderMimeRegistry } from '@jupyterlab/rendermime';
import { IRenderMime } from '@jupyterlab/rendermime-interfaces';

import {
  MIME_TYPE,
  RankOutputRenderer,
  RankSelectionModel
} from './outputRenderer';
import { RANK_MIME_TYPE } from './constants';
import { DebuggerRankSelector } from './debuggerRank';
import { DistributedOutputReconciler } from './outputReconciler';
import { ProcessToolbarController } from './toolbar';

const notebookPlugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter-distributed:notebook',
  description: 'Jupyter Distributed process controls and rank-aware output',
  autoStart: true,
  requires: [IRenderMimeRegistry, INotebookTracker, IToolbarWidgetRegistry],
  activate: (
    app: JupyterFrontEnd,
    rendermime: IRenderMimeRegistry,
    notebooks: INotebookTracker,
    toolbarRegistry: IToolbarWidgetRegistry
  ): void => {
    const rankSelections = new RankSelectionModel();
    const processControls = new ProcessToolbarController();

    toolbarRegistry.addFactory<NotebookPanel>(
      'Notebook',
      'jupyter-distributed-processes',
      panel => processControls.createSelector(panel)
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
        const panel = Private.findNotebook(
          notebooks,
          typeof args.notebookPath === 'string' ? args.notebookPath : null
        );
        return processControls.setWorldSize(panel, processes);
      }
    });

    app.commands.addCommand('jupyter-distributed:run-cell', {
      label: 'Run Distributed Cell',
      caption: 'Run a cell and return its complete rank output',
      describedBy: {
        args: {
          type: 'object',
          properties: {
            cellId: {
              type: 'string',
              description: 'Notebook cell ID to execute'
            },
            notebookPath: {
              type: 'string',
              description: 'Notebook path relative to the Jupyter server root'
            }
          },
          required: ['cellId']
        }
      },
      execute: async args => {
        const cellId = typeof args.cellId === 'string' ? args.cellId : null;
        if (!cellId) {
          throw new Error('cellId is required.');
        }
        const panel = Private.findNotebook(
          notebooks,
          typeof args.notebookPath === 'string' ? args.notebookPath : null
        );
        if (!app.commands.hasCommand('jupyterlab-ai-commands:run-cell')) {
          throw new Error('The Jupyter notebook command tools are not installed.');
        }
        const execution = await app.commands.execute(
          'jupyterlab-ai-commands:run-cell',
          {
            notebookPath: panel.context.path,
            cellId
          }
        );
        return {
          execution,
          distributed_execution: Private.distributedExecution(panel, cellId)
        };
      }
    });

    rendermime.addFactory(Private.rendererFactory(rendermime, rankSelections), 0);

    const reconciledPanels = new WeakSet<NotebookPanel>();
    const registerNotebookRenderer = (panel: NotebookPanel): void => {
      const contextual = panel.content.rendermime;
      contextual.removeMimeType(MIME_TYPE);
      contextual.addFactory(
        Private.rendererFactory(contextual, rankSelections),
        0
      );
      if (!reconciledPanels.has(panel)) {
        reconciledPanels.add(panel);
        new DistributedOutputReconciler(panel);
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
    selections: RankSelectionModel
  ): IRenderMime.IRendererFactory {
    return {
      mimeTypes: [MIME_TYPE],
      safe: true,
      createRenderer: options =>
        new RankOutputRenderer(options, rendermime, selections)
    };
  }

  export function findNotebook(
    notebooks: INotebookTracker,
    requestedPath: string | null
  ): NotebookPanel {
    const path = requestedPath?.replace(/^\/+/, '') ?? null;
    const panel = path
      ? (notebooks.find(candidate => candidate.context.path === path) ?? null)
      : notebooks.currentWidget;
    if (!panel) {
      throw new Error(
        path
          ? `Notebook is not open in JupyterLab: ${path}`
          : 'No active notebook was found.'
      );
    }
    return panel;
  }

  export function distributedExecution(
    panel: NotebookPanel,
    cellId: string
  ): unknown | null {
    const cells = panel.content.model?.cells;
    if (!cells) {
      return null;
    }
    for (let index = 0; index < cells.length; index++) {
      const cell = cells.get(index);
      if (cell.id !== cellId) {
        continue;
      }
      if (!isCodeCellModel(cell)) {
        return null;
      }
      const outputs = cell.outputs.toJSON();
      for (let outputIndex = outputs.length - 1; outputIndex >= 0; outputIndex--) {
        const output = outputs[outputIndex] as unknown as Record<string, unknown>;
        const data = output.data;
        if (
          data &&
          typeof data === 'object' &&
          !Array.isArray(data) &&
          RANK_MIME_TYPE in data
        ) {
          return (data as Record<string, unknown>)[RANK_MIME_TYPE] ?? null;
        }
      }
      return null;
    }
    throw new Error(`Notebook cell not found: ${cellId}`);
  }
}
