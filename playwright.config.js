const base = require('@jupyterlab/galata/lib/playwright-config');

module.exports = {
  ...base,
  testDir: 'ui-tests',
  retries: process.env.CI ? 1 : 0,
  webServer: {
    command:
      'uv run jupyter lab --config=ui-tests/jupyter_server_test_config.py',
    url: 'http://127.0.0.1:9988/lab',
    reuseExistingServer: false,
    timeout: 120000
  },
  use: {
    ...base.use,
    baseURL: 'http://127.0.0.1:9988',
    trace: 'retain-on-failure',
    launchOptions: process.env.PLAYWRIGHT_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH }
      : undefined
  }
};
