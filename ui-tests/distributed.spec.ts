import {
  expect,
  test,
  type IJupyterLabPageFixture
} from '@jupyterlab/galata';

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
  expect(await page.notebook.createNew(name)).toBe(name);
  await setProcesses(page, 2);
}

async function selectOutputRank(
  page: IJupyterLabPageFixture,
  rank: number
): Promise<void> {
  const output = page.locator('.jp-JupyterDistributedRankOutput');
  await output
    .locator('.lm-TabBar-tab')
    .filter({ hasText: `Rank ${rank}` })
    .click();
}

test.describe('distributed notebook rendering', () => {
  test('streams terminal output, isolates ranks, and reconnects', async ({
    page
  }) => {
    await createDistributedNotebook(page, 'distributed.ipynb');

    await page.notebook.setCell(
      0,
      'code',
      [
        'import os, sys, time',
        'rank = int(os.environ["RANK"])',
        'world_size = int(os.environ["WORLD_SIZE"])',
        'print(f"env-{rank}-{world_size}", flush=True)',
        'sys.stderr.write(f"progress-{rank}-0")',
        'sys.stderr.flush()',
        'time.sleep(2)',
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
    await expect(output).toContainText('progress-0-0');
    await expect(output).not.toContainText('end-0');
    await page.notebook.waitForRun(0);
    await expect(output).toContainText('end-0');
    await expect(output).toContainText('progress-0-2');
    await expect(output).not.toContainText('progress-0-0');
    await expect(output.locator('.lm-TabBar-tab')).toHaveCount(2);
    await selectOutputRank(page, 1);
    const rankOneOutput = output.locator(
      '.jp-JupyterDistributedRankOutput-rank[data-rank="1"]'
    );
    await expect(rankOneOutput).toBeVisible();
    await expect(rankOneOutput).toContainText('env-1-2');
    await expect(rankOneOutput).toContainText('progress-1-2');
    await expect(rankOneOutput).toContainText('end-1');

    await page.notebook.setCell(
      0,
      'code',
      [
        '%%rank 1',
        'import os',
        'rank_only = int(os.environ["RANK"]) + 10',
        'print(f"rank-only-{rank_only}")'
      ].join('\n')
    );
    await page.notebook.runCell(0, { inplace: true });
    await expect(output).toContainText('rank-only-11');
    await expect(output).not.toContainText('rank-only-10');
    await expect(output.locator('.lm-TabBar-tab')).toHaveCount(0);

    await page.notebook.setCell(
      0,
      'code',
      [
        'import os',
        'rank = int(os.environ["RANK"])',
        'print(f"state-{rank}-{globals().get(\'rank_only\', \'missing\')}", flush=True)',
        'if rank == 1:',
        '    raise RuntimeError("failure-rank-1")'
      ].join('\n')
    );
    await page.notebook.runCell(0, { inplace: true });
    await expect(output.locator('.lm-TabBar-tab')).toHaveCount(2);
    await selectOutputRank(page, 0);
    await expect(
      output.locator(
        '.jp-JupyterDistributedRankOutput-rank[data-rank="0"]'
      )
    ).toContainText('state-0-missing');
    const failedRankTab = output
      .locator('.lm-TabBar-tab')
      .filter({ hasText: 'Rank 1' });
    await expect(failedRankTab).toHaveClass(/jp-mod-error/);
    await failedRankTab.click();
    await expect(rankOneOutput).toContainText('state-1-11');
    await expect(rankOneOutput).toContainText('failure-rank-1');

    await page.notebook.clickToolbarItem('restart');
    await page
      .getByRole('dialog')
      .getByRole('button', { name: 'Confirm Kernel Restart', exact: true })
      .click();
    await expect(
      page.locator('.jp-JupyterDistributedProcessSelector-input')
    ).toBeEnabled({ timeout: 30000 });
    expect(await page.notebook.save()).toBe(true);

    await page.reload({ waitForIsReady: false });
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

    await page.notebook.setCell(0, 'code', 'import os; print(os.environ["RANK"])');
    await page.notebook.runCell(0, { inplace: true });
    await expect(page.locator('.jp-JupyterDistributedRankOutput')).toContainText(
      '0'
    );
  });

  test('renders live rich display updates independently by rank', async ({
    page
  }) => {
    await createDistributedNotebook(page, 'rich-output.ipynb');
    await page.notebook.setCell(
      0,
      'code',
      [
        'import os, time',
        'from IPython.display import HTML, display',
        'rank = int(os.environ["RANK"])',
        'handle = display(',
        '    HTML(f\'<strong class="rank-rich">initial-{rank}</strong>\'),',
        '    display_id=True,',
        ')',
        'time.sleep(2)',
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
    await page.notebook.waitForRun(0);
    await expect(rankZeroRichOutput).toHaveText('updated-0');
    await expect(output).not.toContainText('initial-0');
    await selectOutputRank(page, 1);
    await expect(rankOneRichOutput).toBeVisible();
    await expect(rankOneRichOutput).toHaveText('updated-1');
    await expect(output).not.toContainText('initial-1');
  });

  test('routes debugger evaluation to the selected rank', async ({ page }) => {
    test.setTimeout(120000);
    await createDistributedNotebook(page, 'debugger.ipynb');
    await page.debugger.switchOn('debugger.ipynb');
    await page.sidebar.openTab('jp-debugger-sidebar');

    await page.notebook.setCell(
      0,
      'code',
      [
        'import os',
        'rank = int(os.environ["RANK"])',
        'a = rank',
        'breakpoint()',
        'print(a)'
      ].join('\n')
    );
    await page.notebook.runCell(0, { inplace: true, wait: false });

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

    await page
      .getByRole('toolbar', { name: 'Callstack panel toolbar' })
      .getByRole('button', { name: 'Continue', exact: true })
      .click();
    await page.notebook.activate('debugger.ipynb');
    await page.notebook.waitForRun(0);

    const output = page.locator('.jp-JupyterDistributedRankOutput');
    await expect(output.locator('.lm-TabBar-tab')).toHaveCount(2);
    await selectOutputRank(page, 0);
    await expect(
      output.locator(
        '.jp-JupyterDistributedRankOutput-rank[data-rank="0"]'
      )
    ).toContainText('0');
    await selectOutputRank(page, 1);
    await expect(
      output.locator(
        '.jp-JupyterDistributedRankOutput-rank[data-rank="1"]'
      )
    ).toContainText('2');
  });
});
