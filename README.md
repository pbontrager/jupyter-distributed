# Jupyter Distributed

Jupyter Distributed makes one notebook behave like one persistent,
multi-process kernel. Select your normal Python, Julia, or other Jupyter kernel,
choose a process count, and ordinary cells execute concurrently on every rank.
Each rank keeps its own state between cells while JupyterLab presents the group
as one logical kernel with rank-aware output.

This repository is an early single-node MVP for JupyterLab 4 and PyTorch
distributed. It is intended for interactive DeviceMesh/DTensor, tensor
parallel, FSDP, expert-parallel, and custom SPMD experiments; it does not choose
a parallelism strategy for the user.

## Install and run with uv

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are required. From a
source checkout:

```bash
uv sync
uv run jupyter lab
```

To create an environment that also includes a default PyTorch build for CPU or
Gloo development, enable the optional dependency set:

```bash
uv sync --extra distributed
```

The wheel installs the server extension and prebuilt JupyterLab extension. It
does not register a kernelspec: existing kernels remain the source of runtime,
environment, language, and startup customization.

For CUDA work, ensure the uv environment resolves the PyTorch build appropriate
for the machine before launching Lab. The project deliberately does not force a
CUDA wheel index or make the large PyTorch wheel part of its small base install.
See PyTorch's current installation guidance for the matching index or preinstall
the desired build into the uv environment.

## Core workflow

1. Start Lab with `uv run jupyter lab` and create a notebook using your normal
   kernel, such as **Python 3**.
2. Leave **Processes: 1** for normal single-process work, or enter any positive
   integer. Changing it restarts the complete kernel group and clears in-memory
   state. Non-default counts are saved under the notebook's namespaced
   `metadata.jupyter_distributed.world_size` field and restored when the
   notebook is reopened. Jupyter installations without this extension safely
   ignore that optional metadata.
3. Execute ordinary cells with Shift+Enter. Every live rank receives the same
   source concurrently.
4. Use the output tabs to inspect stdout, rich results, and exceptions from
   each rank. When the tabs no longer fit, the same control automatically
   becomes a rank dropdown. Prints, logs, progress bars, display updates, and
   output clears are reflected while the cell is still running. The notebook
   retains one logical execution count.
5. Interrupt, restart, and shut down with the standard Jupyter commands; those
   actions apply to the whole group.

A quick persistence check across two separate cells is:

```python
import os
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
rank_local_value = rank * 10
```

```python
rank, world_size, rank_local_value
```

For interactive collectives, prefer a deliberately long process-group timeout:

```python
from jupyter_distributed import init_process_group

init_process_group("nccl", timeout="24h")
```

This changes the PyTorch collective timeout only; it does not disable NCCL
monitoring. Idle time between collectives is safe. A collective for which peers
never arrive can still time out or hang, and a group restart may be required.

## Architecture

```text
JupyterLab extension
  Processes selector + rank-aware output
             |
Jupyter Server extension
  remembers the selected kernelspec and owns group configuration
             |
Internal logical proxy (only for Processes > 1)
             |
selected kernel rank 0 | rank 1 | ... | rank N-1
```

At process count 1, Jupyter launches the selected kernel normally. At larger
process counts, the server restarts the same logical kernel ID through an
internal proxy and launches N copies of the selected kernelspec. The original
kernel name remains visible in the notebook UI. The proxy fans each execution
request out, streams rank output into one updating display, and waits for all
ranks to finish, error, or be interrupted before reporting logical idle. This
notebook-level coordination does not inject a `torch.distributed.barrier()` and
remains separate from the user's NCCL/Gloo process groups. Rank processes
receive the usual torchrun environment:
`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and
`MASTER_PORT`.

The server exposes authenticated lifecycle endpoints beneath
`/jupyter-distributed/kernels/<kernel-id>` and a health endpoint at
`/jupyter-distributed/health`.

## Development

Install the Python and test environment, build the JupyterLab extension, and
run checks with uv-managed commands:

```bash
uv sync --group dev --extra distributed
uv run nbstripout --install --attributes .gitattributes
git config filter.nbstripout.clean \
  "\"$PWD/.venv/bin/python3\" -m nbstripout --keep-id"
git config diff.ipynb.textconv \
  "\"$PWD/.venv/bin/python3\" -m nbstripout -t --keep-id"
git config filter.nbstripout.extrakeys metadata.language_info
uv run --with nodeenv python -m nodeenv --node=20.19.0 .nodeenv
PATH="$PWD/.nodeenv/bin:$PATH" uv run jlpm install
PATH="$PWD/.nodeenv/bin:$PATH" uv run jlpm build
uv run pytest
uv run ruff check .
uv build
```

Use `uv run jupyter lab` from the checkout for manual testing. Frontend changes
may require rebuilding with `uv run jlpm build` and refreshing Lab. The Python
package includes a prebuilt Lab extension directory when one is present, so a
wheel contains the server, internal runtime, and frontend in one
distribution.

CPU/Gloo tests cover the portable runtime path. GPU/NCCL behavior, model
parallelism, cleanup, restart, error aggregation, and output rendering should
also be exercised manually on a two-GPU host before a release.

## Demos

- `demos/01_torchtitan_training.ipynb` builds a deliberately tiny TorchTitan
  Llama model, shards it with PyTorch's composable FSDP, takes a training step,
  inspects local state, and continues training without rebuilding the model.
- `demos/02_transformers_tp_generation.ipynb` uses Accelerate for distributed
  process ergonomics and Transformers native `tp_plan="auto"` to load and reuse
  a small Qwen model across multiple generation cells.

Both notebooks install demo-only dependencies in their first code cell. They
are reference workflows for a compatible two-GPU CUDA machine; they have not
been GPU-validated in this repository, and TorchTitan/Transformers APIs evolve
quickly. Review the pinned ranges and notebook notes before using them for a
long-lived environment.

## MVP limitations

- Single machine only; there is no Slurm, Kubernetes, TorchX, or multi-node
  rendezvous integration.
- World-size changes restart the kernel group. Elastic resizing is not
  supported.
- One notebook maps to one logical group; arbitrary rank-targeted execution is
  not a primary workflow.
- The generic fanout path works with any kernelspec, but language-specific
  multi-process communication remains the user's responsibility.
- Comm-based widgets and interactive debuggers are not forwarded through the
  distributed proxy. Stream output and display updates are supported, but
  bidirectional widget state is outside the SPMD execution model.
- Regular `input()` is not supported by the MVP rank workers; executions reject
  stdin instead of allowing multiple ranks to compete for the notebook channel.
- An ordinary Python error is reported per rank without intentionally killing
  the group, but a failed user collective can leave its process group unusable.
- Distributed `breakpoint()` support is intentionally conservative in the MVP;
  explicit `torch.distributed.breakpoint()` still relies on a healthy user
  process group and consistent participation.
- Notebook files remain standard nbformat documents. Without the Lab extension,
  rank metadata falls back to ordinary stored output rather than the tabbed UI.
- The demos need internet access for installation/model download and suitable
  GPU memory. They are not part of the package's base dependency set.

## License

MIT
