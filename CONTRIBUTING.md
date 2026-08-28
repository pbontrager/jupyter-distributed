# Contributing

## Set up

Install the Python development environment and notebook output filter:

```bash
uv sync --extra mcp --group dev
uv run nbstripout --install
```

For frontend changes, also install the JavaScript dependencies:

```bash
uv run --with nodeenv python -m nodeenv --node=20.19.0 .nodeenv
PATH="$PWD/.nodeenv/bin:$PATH" corepack yarn install --frozen-lockfile
```

Launch the development environment with `uv run jupyter lab`.

## Test

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
PATH="$PWD/.nodeenv/bin:$PATH" npm run build:lib
JUPYTERLAB_CORE="$(uv run python -c 'import pathlib, jupyterlab; print(pathlib.Path(jupyterlab.__file__).parent / "staging")')"
PATH="$PWD/.nodeenv/bin:$PATH" npx --no-install build-labextension . --core-path "$JUPYTERLAB_CORE"
```

GPU demos run when at least two CUDA devices are available and otherwise skip.

## Commit

Format, review, and commit only the intended changes:

```bash
uv run ruff format .
git diff
git add <files>
git commit -m "brief description"
```
