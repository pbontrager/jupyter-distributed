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
  await expect(input).toBeEnabled({ timeout: 30000 });
  await input.fill(String(processes));
  await input.press('Enter');
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: `Restart with ${processes} processes` }).click();
  await expect(input).toHaveValue(String(processes), { timeout: 30000 });
  await expect(input).toBeEnabled({ timeout: 30000 });
}

test.describe('distributed notebook rendering', () => {
  test('streams rank output and survives restart and reconnect', async ({
    page
  }) => {
    await page.notebook.createNew('distributed.ipynb');
    await setProcesses(page, 2);

    await page.notebook.setCell(
      0,
      'code',
      [
        'import os, time',
        'rank = os.environ["RANK"]',
        'print(f"start-{rank}", flush=True)',
        'time.sleep(0.5)',
        'print(f"end-{rank}", flush=True)'
      ].join('\n')
    );
    await page.notebook.runCell(0, { inplace: true, wait: false });

    const output = page.locator('.jp-JupyterDistributedRankOutput');
    await expect(output).toContainText('start-0', { timeout: 30000 });
    await page.notebook.waitForRun(0);
    await expect(output).toContainText('end-0');
    await expect(output.locator('.lm-TabBar-tab')).toHaveCount(2);
    await output.locator('.lm-TabBar-tab').filter({ hasText: 'Rank 1' }).click();
    await expect(output).toContainText('end-1');

    await page.notebook.clickToolbarItem('restart');
    await page.getByRole('dialog').getByRole('button', { name: 'Restart' }).click();
    await expect(
      page.locator('.jp-JupyterDistributedProcessSelector-input')
    ).toBeEnabled({ timeout: 30000 });

    await page.reload();
    await page.waitForCondition(
      async () => await page.notebook.isOpen('distributed.ipynb')
    );
    const processInput = page.locator(
      '.jp-JupyterDistributedProcessSelector-input'
    );
    await expect(processInput).toHaveValue('2', { timeout: 30000 });
    await expect(processInput).toBeEnabled({ timeout: 30000 });

    await page.notebook.setCell(0, 'code', 'import os; print(os.environ["RANK"])');
    await page.notebook.runCell(0, { inplace: true });
    await expect(page.locator('.jp-JupyterDistributedRankOutput')).toContainText(
      '0'
    );
  });
});
