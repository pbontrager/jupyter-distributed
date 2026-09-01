# Contributing

## Set up

Install the Python development environment and notebook output filter:

```bash
uv sync --extra mcp --group dev
uv run nbstripout --install
uv run pre-commit install
```

For frontend changes, also install the JavaScript dependencies:

```bash
uv run --with nodeenv python -m nodeenv --node=20.19.0 .nodeenv
source .nodeenv/bin/activate
corepack enable
yarn install --frozen-lockfile
```

Keep `.nodeenv` activated while running the frontend commands below. In a new
shell, reactivate it with `source .nodeenv/bin/activate`.

Build and link the source extension, then launch the development server:

```bash
yarn develop
uv run jupyter lab
```

This server is for interactive development. The browser test command below
builds and links the extension itself, then starts and stops an isolated
JupyterLab server on port `9988`; do not start a server manually first.

## Test

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
yarn test:browser
```

GPU demos run when at least two CUDA devices are available and otherwise skip.

## Commit

Run all commit checks, review, and commit only the intended changes. The same
checks run automatically after installing the pre-commit hook:

```bash
uv run pre-commit run --all-files
git diff
git add <files>
git commit -m "brief description"
```
