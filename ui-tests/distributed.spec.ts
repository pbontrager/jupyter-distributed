import {
  expect,
  test,
  type IJupyterLabPageFixture
} from '@jupyterlab/galata';
import type { Locator } from '@playwright/test';

// These tests exercise real kernel restarts. Galata's in-memory kernel tracker
// rewrites successful kernel POST responses to 201, but Jupyter's restart API
// returns (and @jupyterlab/services requires) 200.
test.use({ kernels: null });

async function setProcesses(
  page: IJupyterLabPageFixture,
  processes: number
): Promise<void> {
  const input = page.locator('.jp-JupyterDistributedProcessSelector-input');
  await expect(input).toBeVisible({ timeout: 30000 });
  await expect(input).toBeEnabled({ timeout: 30000 });
  await input.fill(String(processes));
  await input.press('Enter');
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await dialog
    .getByRole('button', { name: `Restart with ${processes} processes` })
    .click();
  await expect(input).toHaveValue(String(processes), { timeout: 30000 });
  await expect(input).toBeEnabled({ timeout: 30000 });
}

async function createDistributedNotebook(
  page: IJupyterLabPageFixture,
  name: string
): Promise<void> {
  expect(await page.notebook.createNew(name, { kernel: 'python3' })).toBe(name);
  await setProcesses(page, 2);
  expect(await page.notebook.save()).toBe(true);
}

async function addCodeCell(
  page: IJupyterLabPageFixture,
  source: string
): Promise<number> {
  const index = await page.notebook.getCellCount();
  expect(index).toBeGreaterThanOrEqual(0);
  expect(await page.notebook.addCell('code', '')).toBe(true);
  await setCodeCell(page, index, source);
  return index;
}

async function setCodeCell(
  page: IJupyterLabPageFixture,
  cellIndex: number,
  source: string
): Promise<void> {
  expect(
    await page.evaluate(
      ({ index, value }) => {
        const galata = window.galata as typeof window.galata & {
          setNotebookCell(
            cellIndex: number,
            cellType: 'code',
            source: string
          ): boolean;
        };
        return galata.setNotebookCell(index, 'code', value);
      },
      { index: cellIndex, value: source }
    )
  ).toBe(true);
}

async function cellOutput(
  page: IJupyterLabPageFixture,
  cellIndex: number
): Promise<Locator> {
  const cell = await page.notebook.getCellLocator(cellIndex);
  if (!cell) {
    throw new Error(`Notebook cell ${cellIndex} was not found.`);
  }
  return cell.locator('.jp-JupyterDistributedRankOutput');
}

async function selectOutputRank(
  output: Locator,
  rank: number
): Promise<void> {
  await output
    .locator('.lm-TabBar-tab')
    .filter({ hasText: `Rank ${rank}` })
    .click();
}

async function waitForDebuggerStarted(
  page: IJupyterLabPageFixture
): Promise<void> {
  await expect
    .poll(
      async () =>
        page.evaluate(async () => {
          const panel = window.galata.app.shell.currentWidget as unknown as {
            context?: {
              sessionContext?: {
                session?: {
                  kernel?: {
                    requestDebug(request: {
                      type: 'request';
                      seq: number;
                      command: 'debugInfo';
                    }): {
                      done: Promise<{
                        content: { body?: { isStarted?: boolean } };
                      }>;
                    };
                  } | null;
                } | null;
              };
            };
          };
          const kernel = panel.context?.sessionContext?.session?.kernel;
          if (!kernel) {
            return false;
          }
          const reply = await kernel.requestDebug({
            type: 'request',
            seq: Date.now(),
            command: 'debugInfo'
          }).done;
          return reply.content.body?.isStarted === true;
        }),
      { timeout: 30000 }
    )
    .toBe(true);
}

test.describe('distributed notebook rendering', () => {
  test('streams terminal output, isolates ranks, and reconnects', async ({
    page
  }) => {
    await createDistributedNotebook(page, 'distributed.ipynb');
    const streamDirectory = await page.filebrowser.getCurrentDirectory();
    const streamGatePath = [streamDirectory, 'stream-output-update.gate']
      .filter(Boolean)
      .join('/');

    await setCodeCell(
      page,
      0,
      [
        'import os, sys, time',
        'from pathlib import Path',
        'rank = int(os.environ["RANK"])',
        'world_size = int(os.environ["WORLD_SIZE"])',
        'print(f"env-{rank}-{world_size}", flush=True)',
        'sys.stderr.write(f"progress-{rank}-0")',
        'sys.stderr.flush()',
        'while not Path("stream-output-update.gate").exists():',
        '    time.sleep(0.05)',
        'sys.stderr.write(f"\\rprogress-{rank}-1")',
        'sys.stderr.flush()',
        'time.sleep(0.2)',
        'sys.stderr.write(f"\\rprogress-{rank}-2\\n")',
        'sys.stderr.flush()',
        'print(f"end-{rank}", flush=True)'
      ].join('\n')
    );
    await page.notebook.runCell(0, { inplace: true, wait: false });

    const output = page.locator('.jp-JupyterDistributedRankOutput');
    await expect(output).toContainText('env-0-2', { timeout: 30000 });
    const rankZeroOutput = output.locator(
      '.jp-JupyterDistributedRankOutput-rank[data-rank="0"]'
    );
    const rankOneOutput = output.locator(
      '.jp-JupyterDistributedRankOutput-rank[data-rank="1"]'
    );
    await expect(rankZeroOutput).toContainText('progress-0-0');
    await expect(rankOneOutput).toContainText('progress-1-0');
    await expect(output).not.toContainText('end-0');
    expect(await page.contents.uploadContent('', 'text', streamGatePath)).toBe(
      true
    );
    await page.notebook.waitForRun(0);
    await expect(output).toContainText('end-0');
    await expect(output).toContainText('progress-0-2');
    await expect(output).not.toContainText('progress-0-0');
    await expect(output.locator('.lm-TabBar-tab')).toHaveCount(2);
    await selectOutputRank(output, 1);
    await expect(rankOneOutput).toBeVisible();
    await expect(rankOneOutput).toContainText('env-1-2');
    await expect(rankOneOutput).toContainText('progress-1-2');
    await expect(rankOneOutput).toContainText('end-1');

    const rankOnlyCell = await addCodeCell(
      page,
      [
        '%%rank 1',
        'import os',
        'rank_only = int(os.environ["RANK"]) + 10',
        'print(f"rank-only-{rank_only}")'
      ].join('\n')
    );
    await page.notebook.runCell(rankOnlyCell, { inplace: true });
    const rankOnlyOutput = await cellOutput(page, rankOnlyCell);
    await expect(rankOnlyOutput).toContainText('rank-only-11');
    await expect(rankOnlyOutput).not.toContainText('rank-only-10');
    await expect(rankOnlyOutput.locator('.lm-TabBar-tab')).toHaveCount(0);

    const failedCell = await addCodeCell(
      page,
      [
        'import os',
        'rank = int(os.environ["RANK"])',
        'print(f"state-{rank}-{globals().get(\'rank_only\', \'missing\')}", flush=True)',
        'if rank == 1:',
        '    raise RuntimeError("failure-rank-1")'
      ].join('\n')
    );
    await page.notebook.runCell(failedCell, { inplace: true });
    const failedOutput = await cellOutput(page, failedCell);
    await expect(failedOutput.locator('.lm-TabBar-tab')).toHaveCount(2);
    await selectOutputRank(failedOutput, 0);
    await expect(
      failedOutput.locator(
        '.jp-JupyterDistributedRankOutput-rank[data-rank="0"]'
      )
    ).toContainText('state-0-missing');
    const failedRankTab = failedOutput
      .locator('.lm-TabBar-tab')
      .filter({ hasText: 'Rank 1' });
    await expect(failedRankTab).toHaveClass(/jp-mod-error/);
    await failedRankTab.click();
    const failedRankOutput = failedOutput.locator(
      '.jp-JupyterDistributedRankOutput-rank[data-rank="1"]'
    );
    await expect(failedRankOutput).toContainText('state-1-11');
    await expect(failedRankOutput).toContainText('failure-rank-1');

    await page.notebook.clickToolbarItem('restart');
    await page
      .getByRole('dialog')
      .getByRole('button', { name: 'Confirm Kernel Restart', exact: true })
      .click();
    await expect(
      page.locator('.jp-JupyterDistributedProcessSelector-input')
    ).toBeEnabled({ timeout: 30000 });
    expect(await page.notebook.save()).toBe(true);

    const directory = await page.filebrowser.getCurrentDirectory();
    const directoryParts = directory.split('/').filter(Boolean);
    const notebookPath = [...directoryParts, 'distributed.ipynb'].join('/');
    const directoryPath = directoryParts.map(encodeURIComponent).join('/');
    await page.goto(`tree/${directoryPath}?reset`);
    expect(await page.notebook.openByPath(notebookPath)).toBe(true);
    await expect(
      page
        .getByRole('main')
        .getByRole('tabpanel', { name: 'distributed.ipynb' })
    ).toBeVisible({ timeout: 30000 });
    const processInput = page.locator(
      '.jp-JupyterDistributedProcessSelector-input'
    );
    await expect(processInput).toBeVisible({ timeout: 30000 });
    await expect(processInput).toHaveValue('2', { timeout: 30000 });
    await expect(processInput).toBeEnabled({ timeout: 30000 });

    const reconnectedCell = await addCodeCell(
      page,
      'import os; print(os.environ["RANK"])'
    );
    await page.notebook.runCell(reconnectedCell, { inplace: true });
    await expect(await cellOutput(page, reconnectedCell)).toContainText('0');
  });

  test('renders live rich display updates independently by rank', async ({
    page
  }) => {
    await createDistributedNotebook(page, 'rich-output.ipynb');
    const directory = await page.filebrowser.getCurrentDirectory();
    const gatePath = [directory, 'rich-output-update.gate']
      .filter(Boolean)
      .join('/');
    await setCodeCell(
      page,
      0,
      [
        'import os, time',
        'from pathlib import Path',
        'from IPython.display import HTML, display',
        'rank = int(os.environ["RANK"])',
        'handle = display(',
        '    HTML(f\'<strong class="rank-rich">initial-{rank}</strong>\'),',
        '    display_id=True,',
        ')',
        'while not Path("rich-output-update.gate").exists():',
        '    time.sleep(0.05)',
        'handle.update(HTML(f\'<strong class="rank-rich">updated-{rank}</strong>\'))'
      ].join('\n')
    );
    await page.notebook.runCell(0, { inplace: true, wait: false });

    const output = page.locator('.jp-JupyterDistributedRankOutput');
    const rankZeroRichOutput = output.locator(
      '.jp-JupyterDistributedRankOutput-rank[data-rank="0"] .rank-rich'
    );
    const rankOneRichOutput = output.locator(
      '.jp-JupyterDistributedRankOutput-rank[data-rank="1"] .rank-rich'
    );
    await expect(rankZeroRichOutput).toHaveText('initial-0', {
      timeout: 30000
    });
    await selectOutputRank(output, 1);
    await expect(rankOneRichOutput).toBeVisible();
    await expect(rankOneRichOutput).toHaveText('initial-1');
    expect(await page.contents.uploadContent('', 'text', gatePath)).toBe(true);
    await page.notebook.waitForRun(0);
    await expect(rankOneRichOutput).toHaveText('updated-1');
    await expect(output).not.toContainText('initial-1');
    await selectOutputRank(output, 0);
    await expect(rankZeroRichOutput).toHaveText('updated-0');
    await expect(output).not.toContainText('initial-0');
  });

  test('routes debugger evaluation to the selected rank', async ({ page }) => {
    test.setTimeout(120000);
    await createDistributedNotebook(page, 'debugger.ipynb');
    const toolbar = await page.notebook.getToolbarLocator('debugger.ipynb');
    expect(toolbar).not.toBeNull();
    const debuggerButton = toolbar!.locator('.jp-DebuggerBugButton');
    await expect(debuggerButton).toBeEnabled({ timeout: 30000 });
    await debuggerButton.click();
    await expect(debuggerButton).toHaveAttribute('aria-pressed', 'true', {
      timeout: 30000
    });
    await waitForDebuggerStarted(page);
    await page.sidebar.openTab('jp-debugger-sidebar');

    const debugCell = await addCodeCell(
      page,
      [
        'import os',
        'rank = int(os.environ["RANK"])',
        'a = rank',
        'breakpoint()',
        'print(a)'
      ].join('\n')
    );
    await page.notebook.runCell(debugCell, { inplace: true, wait: false });

    const rankSelector = page.getByRole('combobox', {
      name: 'Distributed debugger rank'
    });
    await expect(rankSelector).toBeVisible({ timeout: 30000 });
    await expect(rankSelector.locator('option')).toHaveCount(2);
    await rankSelector.selectOption('1');
    await expect(rankSelector).toHaveValue('1');
    await expect(rankSelector).toBeEnabled({ timeout: 30000 });

    await page.evaluate(async () => {
      await window.galata.app.commands.execute('debugger:evaluate');
    });
    const debugPrompt = page.locator(
      '.jp-DebugConsole .jp-CodeConsole-promptCell .cm-content'
    );
    await expect(debugPrompt).toBeVisible();
    await debugPrompt.fill('a += 1');
    await debugPrompt.press('Shift+Enter');
    await expect(debugPrompt).toHaveText('', { timeout: 30000 });

    const continueButton = page
      .getByRole('toolbar', { name: 'Callstack panel toolbar' })
      .getByRole('button', { name: /^Continue(?: \(F9\))?$/ });
    await expect(continueButton).toBeEnabled({ timeout: 30000 });
    await continueButton.click();
    await page.notebook.activate('debugger.ipynb');
    await page.notebook.waitForRun(debugCell);

    const output = await cellOutput(page, debugCell);
    await expect(output.locator('.lm-TabBar-tab')).toHaveCount(2);
    await selectOutputRank(output, 0);
    await expect(
      output.locator(
        '.jp-JupyterDistributedRankOutput-rank[data-rank="0"]'
      )
    ).toContainText('0');
    await selectOutputRank(output, 1);
    await expect(
      output.locator(
        '.jp-JupyterDistributedRankOutput-rank[data-rank="1"]'
      )
    ).toContainText('2');
  });
});
