# Jupyter Distributed — Design and Roadmap

This document describes the current technical design and the likely direction
of future development. User installation and usage belong in the
[README](README.md).

## Product model

Jupyter Distributed makes one notebook behave like one persistent
single-program, multiple-data (SPMD) environment:

```text
one notebook
    |
one logical kernel session
    |
rank 0 | rank 1 | ... | rank N-1
```

Every code cell runs concurrently on every rank. Each rank has an independent
interpreter and retains its state between cells. The notebook remains busy until
all ranks complete, fail, or are interrupted.

The extension owns process lifecycle and message coordination. User code owns
the distributed-computing model: collectives, process groups, device placement,
model sharding, and synchronization.

## Design invariants

1. The user selects an ordinary installed kernelspec. There is no public
   Jupyter Distributed kernelspec.
2. A process count of one uses the selected kernel directly.
3. A process count greater than one preserves one logical Jupyter kernel while
   running persistent copies of the selected kernel behind it.
4. Ordinary cells execute on every rank by default.
5. Rank-local state persists between cells.
6. Process-count changes restart the whole group; resizing is not elastic.
7. Interrupt, restart, and shutdown apply to every rank.
8. Notebook coordination never inserts a user-level collective or barrier.
9. The notebook control plane must remain independent of PyTorch, NCCL, Gloo,
   or any other user communication layer.
10. Notebook files remain valid standard nbformat documents.

## Current architecture

The Python distribution contains three cooperating components:

```text
JupyterLab extension
  process control + rank output renderer
                  |
Jupyter Server extension
  selected-kernel lifecycle coordinator
                  |
internal proxy kernel
  request fanout + response aggregation
                  |
selected kernel rank 0 ... selected kernel rank N-1
```

### JupyterLab extension

The frontend adds a positive-integer **Processes** field to each notebook. It
uses the authenticated server endpoint at
`/jupyter-distributed/kernels/<kernel-id>` to read or change the process count.

Non-default counts are stored as optional notebook metadata:

```json
{
  "metadata": {
    "jupyter_distributed": {
      "world_size": 8
    }
  }
}
```

Unknown notebook metadata is ignored by ordinary Jupyter installations. The
frontend restores the saved count after the document and kernel session are
ready.

Distributed output uses
`application/vnd.jupyter-distributed.rank+json`. Ranks are shown as tabs while
there is sufficient width and as a dropdown when the tabs would become too
narrow. Each cell's rank selection is independent and remains stable while its
live output is updated.

### Jupyter Server extension

The server owns process-count transitions. When changing from one process to
multiple processes, it captures the selected kernel's launch configuration,
restarts the same logical kernel ID through the internal proxy, and retains the
selected kernel name in the notebook UI. Returning to one process restores the
original launch configuration.

The coordinator is intentionally independent of the notebook's user-level
distributed state. A broken collective must not prevent the user from
interrupting, restarting, or shutting down the kernel group.

### Internal proxy kernel

The proxy is an implementation detail. It creates a `DistributedKernelGroup`
containing one persistent child kernel per rank and forwards ordinary execute,
completion, inspection, interrupt, restart, and shutdown behavior.

The first rank supplies language and kernel metadata. Debugger support is
reported as unavailable, and unexpected debugger probes receive a valid
unsupported response rather than raising in the control channel.

### Rank kernels

Each rank is launched from the user's selected kernelspec. The current launcher
is local and assigns:

```text
RANK
LOCAL_RANK
WORLD_SIZE
LOCAL_WORLD_SIZE
MASTER_ADDR
MASTER_PORT
JAX_COORDINATOR_ADDRESS
JAX_PROCESS_ID
JAX_NUM_PROCESSES
```

The first six variables are torchrun compatible. The JAX variables expose the
same local rendezvous and rank information for `jax.distributed.initialize`.
They do not initialize a process group or impose a framework. All ranks
currently reside on one machine, so global and local rank values are identical.

## Execution and output semantics

For each code cell, the proxy:

1. assigns one logical execution count;
2. sends the same source to every rank concurrently;
3. creates one rank-aware notebook output;
4. streams output updates while ranks are running;
5. waits for every rank's execute reply and idle state;
6. reports a logical error if any rank failed.

Per-rank output state follows Jupyter output-area behavior:

- adjacent stdout or stderr messages are coalesced;
- carriage returns and backspaces update terminal-style progress text;
- `display_data` is retained as rich MIME data;
- `update_display_data` replaces matching display IDs;
- `clear_output`, including delayed clearing, is applied independently by rank;
- rapid changes are coalesced before updating the browser.

The proxy publishes a standard display with a stable display ID. Live updates
replace that display, and the final update becomes the notebook's saved
rank-aware output. Plain-text and HTML fallbacks keep saved notebooks readable
when the frontend extension is unavailable.

## Lifecycle and failure behavior

- A process-count change warns the user and clears all in-memory rank state.
- A normal Jupyter restart recreates the complete group at the current size.
- Interrupt is fanned out to all child kernel managers.
- Shutdown attempts to terminate every child even if another child fails.
- A normal exception on one rank does not automatically kill the other ranks.
- A failed user collective may leave framework state unusable; restarting the
  group is the recovery mechanism.
- Multi-rank `input()` is rejected rather than allowing processes to compete
  for one stdin channel.
- Python `breakpoint()` is disabled in distributed mode instead of opening
  competing debugger sessions.

## Distributed framework boundary

PyTorch and JAX are optional. The extension provides convenient environment
variables for both, but the notebook must initialize and manage its own process
groups through the frameworks' standard APIs.

There is deliberately no implicit `torch.distributed.barrier()` after a cell.
Cell completion is coordinated over Jupyter's control path, independently of
the user's process group. Long pauses between cells therefore do not represent
an outstanding collective and do not require extending PyTorch's process-group
timeout.

## Current limitations

- Local, single-machine process launch only.
- JupyterLab 4 is the supported and tested frontend.
- No elastic world-size changes.
- No arbitrary rank-targeted execution mode.
- No stdin routing in distributed mode.
- No interactive debugger integration.
- No bidirectional comm proxying, so `ipywidgets`, `tqdm.notebook`, and similar
  widget protocols are not supported in distributed mode.
- No automatic device assignment, collective initialization, sharding, or
  recovery of failed user process groups.
- Output updates currently publish the accumulated rank snapshot; extremely
  high-volume output may need a more incremental transport.

## Testing strategy

Portable CPU integration tests cover:

- multi-rank launch and environment values;
- persistent state across cells;
- live streams and output-state updates;
- rank-specific results and exceptions;
- interrupt, restart, shutdown, and cleanup;
- debugger capability probes;
- a two-rank Gloo collective when PyTorch is installed.

Release testing should additionally cover JupyterLab browser behavior and
two-GPU NCCL workflows. Demo notebooks should be checked periodically because
TorchTitan and Transformers distributed APIs evolve quickly.

## Roadmap

### Near term

- Add automated JupyterLab browser tests for process selection, live output,
  responsive rank navigation, restart restoration, and error selection.
- Exercise additional kernelspecs and remove Python-specific assumptions from
  generic paths.
- Improve rank startup, liveness, and partial-failure diagnostics.
- Measure high-volume streaming behavior and replace full snapshots with an
  incremental protocol if needed.
- Validate and maintain the TorchTitan and Transformers demos on supported GPU
  configurations.
- Prepare package publishing, compatibility policy, and release automation.

### Later

- Add launcher interfaces for multi-node environments such as Slurm,
  Kubernetes, TorchX, or torch elastic rendezvous.
- Add all-rank and diff output views, identical-output collapsing, and richer
  rank status information.
- Add DTensor/DeviceMesh placement and memory inspection tools.
- Surface collective hangs and NCCL diagnostics without coupling lifecycle
  control to the user process group.
- Consider trusted server-side launch-environment configuration with optional
  rank-aware substitutions.

### Intentionally out of scope

- General task scheduling or load balancing.
- Automatically selecting a parallelism strategy.
- Automatically initializing or repairing user collectives.
- Treating individual ranks as separate notebook sessions.
- Bidirectional widget or debugger multiplexing unless a compelling SPMD use
  case emerges.

## Demos

- [TorchTitan distributed training](demos/01_torchtitan_training.ipynb)
- [Transformers tensor-parallel generation](demos/02_transformers_tp_generation.ipynb)

The demos are integration examples, not dependencies of the core package.
