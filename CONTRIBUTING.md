# Contributing

## Set up

Install the Python development environment and notebook output filter:

```bash
uv sync --group dev
uv run nbstripout --install
```

For frontend changes, also install the JavaScript dependencies:

```bash
uv run --with nodeenv python -m nodeenv --node=20.19.0 .nodeenv
PATH="$PWD/.nodeenv/bin:$PATH" uv run jlpm install --frozen-lockfile
```

Launch the development environment with `uv run jupyter lab`.

## Test

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
PATH="$PWD/.nodeenv/bin:$PATH" uv run jlpm build
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
