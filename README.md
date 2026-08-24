# Jupyter Distributed

Jupyter Distributed runs each notebook cell across multiple persistent kernel
processes and groups their outputs by rank. It adds a **Processes** control to
JupyterLab while letting you continue to select and customize your normal
Jupyter kernels.

The project is intended for interactive single-machine SPMD development:
distributed training experiments, collective communication, tensor/model
parallelism, and any workflow where every process should execute the same
notebook code while retaining independent state between cells.

## Installation

Install Jupyter Distributed into the same environment as JupyterLab:

```bash
python -m pip install jupyter-distributed
jupyter lab
```

With uv:

```bash
uv add jupyter-distributed
uv run jupyter lab
```

The Jupyter Server and JupyterLab extensions are enabled automatically. Restart
JupyterLab after installing or upgrading the package.

## Using it

1. Open or create a notebook and select its normal kernel, such as **Python 3**.
2. Enter a positive integer in the **Processes** field in the notebook toolbar.
3. Confirm the restart. Every subsequent cell is executed concurrently by that
   many persistent processes.
4. Select a rank tab to inspect its output. When there are too many tabs for the
   available width, the rank control becomes a dropdown.

Changing the process count restarts the complete group and clears in-memory
state. Non-default counts are stored in optional notebook metadata and restored
when the notebook is reopened. Jupyter installations without this extension
ignore that metadata normally.

Outputs stream while a cell is running. Standard streams, rich display data,
display updates, output clearing, exceptions, and terminal-style progress
updates are kept separate for each rank.

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
{"rank": rank, "value": value}
```

The output contains two rank views with independently generated values. Both
`rank` and `value` remain available in later cells on their respective process.

## What it does not do

Jupyter Distributed provides local process lifecycle, SPMD cell execution, and
rank-aware output. It does not:

- choose or configure a distributed-computing framework;
- initialize collectives, shard models, or assign devices;
- schedule work across multiple machines, clusters, Slurm, or Kubernetes;
- provide elastic process resizing;
- support stdin, interactive debugging, or comm-based widgets such as
  `ipywidgets` and `tqdm.notebook` in distributed mode.

The selected runtime remains responsible for communication between processes.
The generic execution path is not limited to Python, although runtime-specific
distributed setup remains the user's responsibility.

## PyTorch Distributed

The package does not require PyTorch. Install the PyTorch build appropriate for
your machine and CUDA environment as part of your existing project. For a basic
default installation, an optional extra is available:

```bash
python -m pip install "jupyter-distributed[distributed]"
```

Each process receives torchrun-compatible environment variables:

- `RANK`
- `LOCAL_RANK`
- `WORLD_SIZE`
- `LOCAL_WORLD_SIZE`
- `MASTER_ADDR`
- `MASTER_PORT`

Jupyter Distributed selects a local rendezvous address and available port, but
does not initialize a process group. A notebook can do that explicitly:

```python
import torch.distributed as dist
from jupyter_distributed import init_process_group

if not dist.is_initialized():
    init_process_group("gloo", timeout="24h")

dist.get_rank(), dist.get_world_size()
```

For NCCL, select the appropriate CUDA device from `LOCAL_RANK` before
initializing the process group. Environment variables may also be changed in an
earlier cell before the distributed framework reads them.

## Demos

- [TorchTitan distributed training](demos/01_torchtitan_training.ipynb)
- [Transformers tensor-parallel generation](demos/02_transformers_tp_generation.ipynb)

The demos require their own model/runtime dependencies and suitable hardware.
They are examples of integrating existing distributed libraries rather than
features implemented by Jupyter Distributed itself.

## License

MIT
