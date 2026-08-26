# Jupyter Distributed

Jupyter Distributed is a notebook extension for running each cell concurrently
across multiple persistent kernel processes. It adds a **Processes** option to
JupyterLab and Jupyter Notebook 7 while letting you continue to select and
customize your normal Jupyter kernels.

This follows the single program, multiple data (SPMD) model: every process runs
the same code but maintains its own independent state. In a notebook, that means
one cell can update several parallel interactive sessions whose variables remain
available in later cells. The model is general, while naturally supporting
distributed training patterns found in ecosystems such as PyTorch and JAX.

![Tensor-parallel generation across eight notebook processes](https://raw.githubusercontent.com/pbontrager/jupyter-distributed/main/tp-chat-generation.png)

## Installation

Install Jupyter Distributed into the same environment as JupyterLab or Jupyter
Notebook 7:

```bash
pip install jupyter-distributed
jupyter lab
```

Use `jupyter notebook` instead of `jupyter lab` to launch Notebook 7. The
Jupyter Server and frontend extensions are enabled automatically. Restart the
server after installing or upgrading the package.

## Using it

1. Open or create a notebook and select its normal kernel, such as **Python 3**.
2. Enter the number of parallel processes in the **Processes** field in the
   notebook toolbar.
3. Confirm the restart. Every subsequent cell is executed concurrently by that
   many persistent processes.
4. Select a rank tab to inspect its output.

Changing the process count stops the current kernel processes, starts a new
group at the requested size, and clears all in-memory state.

With **Processes** greater than `1`, run a cell on only one zero-based rank by
starting it with `%%rank N`. The wrapper can contain ordinary code or another
cell magic:

```python
%%rank 1
activation.max().item()
```

```python
%%rank 0
%%ai
Explain the model defined in this notebook.
```

Only the selected rank runs the cell body or changes state. This is useful for
rank-local inspection and for operations such as `%%ai` that should run once
rather than independently on every process.

Outputs stream while a cell is running. Standard streams, rich display data,
display updates, output clearing, exceptions, and terminal-style progress
updates are kept separate for each rank. Comm-based interactive outputs such as
`ipywidgets` and `tqdm.notebook` also remain independent: each rank owns its
widget state and interactions are routed back to that rank. Widget state is
restored when the browser reconnects to a still-running kernel.

### Debugging Python in JupyterLab

The standard JupyterLab debugger works with distributed IPython kernels. A
notebook breakpoint is installed in every process. Continue, pause, step over,
step in, and step out apply to all paused processes together, while stack frames
and variables are inspected for the selected rank thread.

While paused, use the **Rank** section in the debugger sidebar to choose which
process supplies the Calls and Variables views. JupyterLab's existing debug
console follows the same selected rank. Open it with **Debugger: Evaluate Code**
from the Command Palette and run expressions with `Shift+Enter`.

This is the first-stage debugging interface: ranks appear as threads in
JupyterLab's existing debugger panel rather than in dedicated rank tabs.
Python's built-in `breakpoint()` also works in notebook cells and imported
libraries once the JupyterLab debugger is active. Without an attached debugger,
it prints a warning instead of starting competing `pdb` sessions.

Notebook 7 supports distributed execution and rank-aware output, but the
debugger integration is currently available only in JupyterLab.

### Jupyter AI

When Jupyter AI is installed in the same server environment, Jupyter
Distributed automatically adds MCP tools that let an agent check the live
process count, read every rank's output, and select the rank shown in the
JupyterLab debugger. No additional configuration is required.

## Computation model

Jupyter Distributed follows the single program, multiple data (SPMD) model:

- Every process receives the same cell source.
- Each process has its own interpreter and independent variables.
- Process state persists across cells.
- A cell is considered complete when every process has completed, failed, or
  been interrupted.
- Standard interrupt, restart, and shutdown actions apply to the whole group.

For example, set **Processes** to `2` and run:

```python
import os
import random

rank = int(os.environ["RANK"])
value = random.randint(0, 9)
rank * 10 + value
```

The output contains two rank views with independently generated values. Both
`rank` and `value` remain available in later cells on their respective process.

For complete distributed-model examples, see the [demo notebooks](#demos).

## What it does not do

Jupyter Distributed provides local process lifecycle, SPMD cell execution, and
rank-aware output. It does not:

- choose or configure a distributed-computing framework;
- initialize collectives, shard data or models, or assign devices;
- schedule work across multiple machines, clusters, Slurm, or Kubernetes;
- provide elastic process resizing;
- support Notebook 6 or the legacy classic Notebook frontend;
- support interactive stdin prompts in distributed mode.

The selected runtime remains responsible for communication between processes.
The execution protocol is designed to support any kernelspec, but the current
implementation has only been tested with Python kernels.

## PyTorch and JAX Distributed

Jupyter Distributed is framework agnostic, but PyTorch and JAX distributed
workloads are flagship use cases.

### PyTorch

To make `torch.distributed` convenient, each process receives
torchrun-compatible environment variables:

- `RANK`
- `LOCAL_RANK`
- `WORLD_SIZE`
- `LOCAL_WORLD_SIZE`
- `MASTER_ADDR`
- `MASTER_PORT`

Jupyter Distributed selects a local rendezvous address and available port, but
does not initialize a process group. A notebook uses the ordinary PyTorch API:

```python
import torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group("gloo")

dist.get_rank(), dist.get_world_size()
```

For NCCL, select the appropriate CUDA device from `LOCAL_RANK` before
initializing the process group. The defaults may be overridden in an earlier
cell before `init_process_group()` reads them:

```python
import os

os.environ["MASTER_PORT"] = "29501"
```

PyTorch's process-group timeout applies to outstanding collective operations,
not idle time between notebook cells, so normal interactive pauses do not
require a longer timeout.

### JAX

JAX processes additionally receive:

- `JAX_COORDINATOR_ADDRESS`
- `JAX_PROCESS_ID`
- `JAX_NUM_PROCESSES`

JAX reads these values directly, so initialization is parallel to the PyTorch
example:

```python
import jax

jax.distributed.initialize()

jax.process_index(), jax.process_count()
```

Device visibility and any framework-specific distributed configuration remain
the notebook's responsibility.

## Demos

- [FSDP training with Transformers](demos/fsdp_training.ipynb)
- [Tensor-parallel chat with Transformers](demos/tp_chat.ipynb)
- [Distributed debugging with PyTorch](demos/debugging.ipynb)

The demos require their own model/runtime dependencies and suitable hardware.
They are examples of integrating existing distributed libraries rather than
features implemented by Jupyter Distributed itself.

## License

Licensed under the [MIT License](LICENSE).
