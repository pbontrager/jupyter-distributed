# Jupyter Distributed — Interactive SPMD Kernel Design & Implementation Handoff

**Status:** implementation design / agent handoff  
**Primary target:** JupyterLab 4.x + Jupyter Server + PyTorch distributed  
**Initial scope:** single-node, multi-GPU, persistent SPMD notebook execution  
**Future scope:** multi-node launchers (torchrun rendezvous, TorchX, Slurm/Kubernetes integrations)

## 1. Product thesis

PyTorch's SPMD distributed programming model makes tensor parallelism (TP), expert parallelism (EP), FSDP, DeviceMesh/DTensor, and combinations of them feel relatively straightforward in normal Python scripts, but the interactive Jupyter workflow is largely lost once the program requires multiple persistent Python processes.

The goal of this project is to restore the normal eager notebook experience for distributed models, including models that **cannot fit on a single GPU**.

A user should be able to:

1. launch JupyterLab normally;
2. install/enable this extension;
3. open a normal Python notebook;
4. choose a process/world size from a notebook toolbar control, defaulting to `1`;
5. when the value is changed to `N > 1`, restart/relaunch the notebook kernel as an N-process SPMD kernel group;
6. continue executing ordinary notebook cells with Shift+Enter;
7. have every code cell execute on every rank with persistent rank-local Python state;
8. see cell output grouped by rank through tabs or a compact aggregated view;
9. stop between cells for minutes or hours, inspect objects, construct inputs interactively, run another few steps, and continue — exactly as with a single-process eager notebook.

The defining UX is not "control N kernels." It is:

> **This notebook has N persistent Python processes.**

The multiplicity should mostly disappear until the user wants rank-specific output or debugging information.

This is especially valuable for TP/EP/FSDP workflows: a distributed/sharded model can remain alive in notebook memory and be inspected or driven interactively even though no individual GPU owns the full model.

---

## 2. Explicit architectural decision

Do **not** build the top-level runtime on ipyparallel.

ipyparallel is useful prior art and may be consulted or reused for narrowly scoped implementation ideas, especially around multi-process rich output aggregation and process lifecycle, but its top-level abstraction is a client/controller/engine system intended for more general parallel execution.

This project instead owns a narrower abstraction:

```text
one notebook
    |
one logical distributed kernel
    |
rank 0 | rank 1 | ... | rank N-1
```

Do not expose a project-owned kernelspec. The user selects an existing
kernelspec, and process count is an independent property applied by the server.
At world size 1 the selected kernel runs directly. At larger world sizes the
server launches persistent copies of that same kernelspec behind one logical
session.

This avoids carrying two overlapping notions of distributed identity (`ipyparallel engine_id` and `torch rank`) and avoids an impedance-matching layer where the notebook is technically attached to a controller kernel while the meaningful Python state lives elsewhere.

---

## 3. Architecture overview

The product should consist of three major pieces packaged together:

```text
JupyterLab frontend extension
        |
        | ordinary notebook UX + distributed metadata/UI
        v
Jupyter Server extension
        |
        | owns one logical DistributedKernelGroup
        v
Distributed kernel launcher / proxy
        |
        | selected kernelspec + torchrun-compatible environment
        v
rank 0 kernel  rank 1 kernel ... rank N-1 kernel
```

### 3.1 JupyterLab frontend extension

Responsibilities:

- Add a toolbar control to notebooks, ideally near the kernel name/status:

  ```text
  Kernel: Python 3    Processes: [ 1 ]
  ```

- Default world size to `1`, preserving normal Jupyter behavior.
- When the user changes `1 -> N` or `N -> M`:
  - warn that in-memory kernel state will be lost;
  - request a group restart/relaunch with the requested world size;
  - preserve the notebook document and cells;
  - do not attempt elastic in-place world-size mutation.
- Treat a distributed kernel group as one logical notebook kernel.
- Render outputs with rank awareness.
- Make interrupt, restart, shutdown, reconnect, and kernel-status actions operate on the logical group.
- Surface per-rank status when useful without cluttering the normal notebook experience.

JupyterLab currently exposes configurable Notebook toolbar extension points and semantic kernel commands, so this should be implemented as an extension rather than a JupyterLab fork.

### 3.2 Jupyter Server extension

A server component is intentional even for the single-node MVP.

It should own lifecycle and routing because it outlives any individual rank process and is a better authority for:

- launching the rank group;
- tracking rank PIDs/processes and kernel connection information;
- mapping one notebook session to one logical distributed kernel group;
- fanning execution requests out to all ranks;
- aggregating rank responses;
- group interrupt/restart/shutdown;
- cleanup after kernel/browser failure;
- maintaining the independent notebook control plane even if NCCL or the user process group is unhealthy.

Do not require users to run a separate controller command such as `ipcluster start`.

The intended installation/launch experience should remain roughly:

```bash
pip install -e .
jupyter lab
```

### 3.3 DistributedKernelGroup

The core backend abstraction should be explicit and implementation-owned.

Illustrative interface:

```python
class DistributedKernelGroup:
    world_size: int

    async def start(self) -> None: ...
    async def execute(self, request) -> GroupExecution: ...
    async def interrupt(self) -> None: ...
    async def restart(self, world_size: int | None = None) -> None: ...
    async def shutdown(self) -> None: ...
    async def status(self) -> GroupStatus: ...
```

Each rank is an ordinary persistent instance of the user's selected Jupyter
kernel, but the notebook-facing abstraction is the group.

### 3.4 Torchrun-style launcher

For `world_size > 1`, launch all rank kernels from the selected kernelspec as
one coherent distributed world rather than asking the user to select a custom
proxy kernelspec.

Each process should receive the standard distributed environment, including at minimum:

```text
RANK
LOCAL_RANK
WORLD_SIZE
LOCAL_WORLD_SIZE
MASTER_ADDR
MASTER_PORT
```

The server-side launcher may use an internal proxy for Jupyter message fanout,
but that proxy is an implementation detail. It must preserve the selected
kernelspec's name and configuration in the notebook UI.

The system must **not** automatically choose a language-specific distributed
runtime or parallelism strategy. It creates a persistent SPMD process group.
User code may initialize PyTorch DeviceMesh, TP, EP, FSDP, Julia distributed
primitives, or any other runtime-specific communication mechanism.

---

## 4. Control plane vs. user distributed plane

This separation is a central design requirement.

### Notebook control plane

Independent communication between the Jupyter Server extension and every rank kernel. It owns:

- execute-cell fanout;
- completion tracking;
- logical busy/idle state;
- interrupt/restart/shutdown;
- rank liveness;
- stdin routing;
- debugger coordination;
- rank output metadata.

### User distributed plane

Owned by the user's PyTorch program and may contain:

- NCCL/Gloo;
- DeviceMesh / DTensor;
- TP;
- EP;
- FSDP/HSDP;
- PP/CP;
- custom process groups and collectives.

**Never implement notebook synchronization by injecting `torch.distributed.barrier()` at the end of cells.**

The control plane must remain useful even if the user's NCCL/process-group state is desynchronized or broken.

---

## 5. Cell execution semantics

### 5.1 SPMD fanout

For every ordinary code-cell execution:

1. receive one logical execute request from JupyterLab;
2. create a logical execution ID;
3. dispatch the same cell source concurrently to every live rank;
4. preserve the rank-local Python namespace from previous cells;
5. collect IOPub/reply messages from every rank and tag them with rank identity;
6. keep the logical notebook kernel `busy` while any rank is busy;
7. consider the logical cell complete only once every rank has completed, errored, or been explicitly interrupted;
8. then transition the logical kernel to `idle` and permit the next cell execution.

This is the notebook-level barrier. It is **not** a user collective.

### 5.2 Execution count

Expose one logical notebook execution count.

Do not allow visible execution numbers to drift by rank. Internally the rank kernels may have their own counters, but the notebook output/model should present a single logical `In [N]` sequence.

### 5.3 Errors

One-rank failure makes the logical cell a failed execution, but the system should still collect useful output/status from all ranks.

Desired UI direction:

```text
Cell failed on 1 / 8 ranks

[Rank 0 ok] [Rank 1 ok] [Rank 2 error] [Rank 3 ok] ...
```

Prefer focusing the first errored rank automatically while preserving tabs for all rank outputs.

Do not automatically kill the whole world on an ordinary Python exception unless continuation is known to be impossible. Users are explicitly using this tool to inspect state after failures.

Some collective failures may leave the user process group unusable; make group restart easy rather than pretending recovery is always safe.

---

## 6. Output model and frontend UX

The frontend needs structured rank identity, not text that has already been flattened together.

For each logical cell, preserve rank-indexed streams/results conceptually like:

```json
{
  "execution_id": "...",
  "rank_outputs": {
    "0": ["...Jupyter messages..."],
    "1": ["...Jupyter messages..."]
  }
}
```

### Minimum UI

Render a compact tab strip for distributed cells:

```text
[ Rank 0 ] [ Rank 1 ] [ Rank 2 ] [ Rank 3 ]
```

Keep tabs while each rank can retain a reasonable minimum width. When the
available output width cannot fit them, replace the tab strip with a rank
dropdown while preserving the selected rank.

The selected tab displays that rank's stdout/stderr/rich outputs/exceptions.

Create the rank-aware output at execution start and update it as rank IOPub
messages arrive. Coalesce rapid updates, preserve terminal carriage-return and
backspace behavior, and apply `display_data`, `update_display_data`, and
`clear_output` semantics independently for every rank. Do not wait for all
ranks to finish before showing prints, logs, or progress.

Useful behavior:

- remember the last selected rank at notebook level;
- indicate ranks with errors in their tabs;
- avoid rendering N copies when outputs are byte-for-byte or semantically identical if an aggregated mode is selected;
- keep rank 0 as the default selected rank initially.

### Future UI modes (not required for MVP)

- `Rank 0 | All | Diff` selector;
- detect identical outputs and show `all ranks identical`;
- tensor-aware summaries/comparisons;
- rank status panel;
- distributed deadlock/desynchronization diagnostics;
- per-rank variable inspector.

### Notebook file compatibility

Do not require a custom nbformat fork.

Prefer standard outputs plus metadata/custom MIME payloads where needed, with a reasonable plain-text/rank-0 fallback when the extension is unavailable. The exact persistence format may be deferred until the live execution path works, but compatibility with normal `.ipynb` tooling is a requirement.

---

## 7. Kernel lifecycle UX

### 7.1 Toolbar selector

Primary control:

```text
Processes: [1 v]
```

Accept any positive integer and reject empty, fractional, zero, or negative
values before requesting a restart.

Persist non-default values as `metadata.jupyter_distributed.world_size` in the
notebook document. Restore that value when the notebook is reopened; ordinary
Jupyter clients ignore the namespaced metadata when the extension is absent.

Do not call it "parallel kernels" in the primary UI. The user should think of process/world size as a property of the notebook kernel.

### 7.2 World-size changes

Changing process count always performs a restart/relaunch.

Example:

```text
Changing process count will restart this notebook's kernel.
All in-memory state will be lost.

Restart with 8 processes
```

No elastic resizing in the initial implementation.

### 7.3 Standard kernel commands

Existing Jupyter actions should map to group behavior:

- **Interrupt Kernel** -> interrupt all ranks;
- **Restart Kernel** -> restart the complete rank group with the same world size;
- **Shutdown Kernel** -> shut down all ranks;
- **Reconnect** -> reconnect to the logical group/control plane;
- **Restart and Run All** -> restart whole group and then SPMD-execute cells normally.

The normal JupyterLab UI should remain functional rather than introducing a separate distributed-only command ecosystem.

---

## 8. Breakpoints and interactive debugging

Regular Python `breakpoint()` cannot be allowed to spawn N independent pdb sessions that compete for stdin.

### 8.1 MVP behavior

Install a distributed-aware breakpoint hook in rank kernels using Python's supported breakpoint hook mechanism (`PYTHONBREAKPOINT` and/or `sys.breakpointhook`).

However, do **not** blindly rewrite every `breakpoint()` into `torch.distributed.breakpoint(rank=0)`: rank-conditional code can make a barrier-based breakpoint deadlock.

Instead, implement breakpoint coordination through the independent notebook control plane.

Conceptual behavior:

```text
rank reaches breakpoint()
        |
        v
rank notifies server control plane
        |
        v
server chooses/validates active debug rank
        |
        +--> selected rank receives interactive debugger stdin
        |
        +--> participating peer ranks are suspended by control-plane coordination
        |
continue
        |
server releases suspended peers
```

Default debug rank can be rank 0 for MVP.

The control-plane design should leave room for selecting the debug rank from the UI later.

### 8.2 `torch.distributed.breakpoint`

Do not prevent users from explicitly calling PyTorch's own distributed breakpoint. It should still work when used correctly.

Document that PyTorch's distributed breakpoint uses distributed synchronization and therefore depends on the user's process group being healthy and on participating ranks entering it consistently.

The extension's normal `breakpoint()` should eventually be more robust because it does not depend on NCCL/user process-group health.

### 8.3 stdin

At most one rank may own interactive stdin at a time.

For normal `input()` outside debugger mode, MVP may initially route stdin to rank 0 only and either:

- make other ranks receive the same response; or
- explicitly mark multi-rank `input()` unsupported until semantics are defined.

Do not let multiple kernels race for the notebook stdin channel.

---

## 9. Timeouts and NCCL behavior for interactive use

Interactive notebooks have fundamentally different timing than batch jobs. A user may leave a distributed model idle between cells for a long period.

### 9.1 Important distinction

Idle time **between** collectives is harmless. A process-group timeout only matters when an operation is outstanding and peers fail to join/complete it.

Therefore, do not disable every NCCL watchdog mechanism merely because the notebook can sit idle.

### 9.2 Process-group timeout

PyTorch's NCCL process-group collective timeout is currently finite by default (official docs currently describe a 10-minute NCCL default).

The extension itself should not silently monkey-patch arbitrary user calls to `dist.init_process_group()` unless there is a clean supported hook.

For MVP, provide one or both of:

1. a small helper API for interactive initialization with a long timeout, e.g.:

   ```python
   from jupyter_distributed import init_process_group
   init_process_group(timeout="24h")
   ```

2. documented first-class guidance that notebook users should initialize process groups with a long timeout, e.g. `timedelta(hours=24)`.

If a reliable supported PyTorch configuration/environment mechanism exists at implementation time to establish a long default without overriding explicit user settings, use it. Do not depend on an undocumented NCCL environment variable as a substitute for the PyTorch process-group `timeout` argument.

### 9.3 NCCL watchdog/diagnostics

Do not broadly disable NCCL monitoring/watchdog behavior. A genuine NCCL/CUDA hang should still be diagnosable.

Consider an optional diagnostics mode using current ProcessGroupNCCL flight-recorder features such as trace buffers and timeout dumps. Keep this out of the critical MVP path unless it is nearly free.

The control plane should be capable of telling the user that the notebook is waiting on a rank even if the user distributed plane is stuck.

---

## 10. Process environment and devices

For a local N-GPU launch, assign one process per GPU by default.

Rank entrypoint should establish the expected local CUDA device early, e.g. using `LOCAL_RANK`, while avoiding unnecessary CUDA initialization before user code where possible.

Expose standard environment values exactly as torchrun users expect so normal distributed libraries work unchanged.

The design must not special-case DDP. Success criteria explicitly include:

- DeviceMesh / DTensor;
- tensor parallelism;
- expert parallelism;
- FSDP2 / composable sharding;
- hybrid combinations;
- ordinary custom SPMD code.

---

## 11. Suggested repository structure

Use a single Python distribution that includes the server extension, Python runtime/launcher, and prebuilt/developable JupyterLab extension.

Suggested layout:

```text
jupyter-distributed/
├── pyproject.toml
├── README.md
├── LICENSE
├── package.json                    # if using standard JupyterLab TS build layout
├── tsconfig.json
├── src/                            # JupyterLab frontend TypeScript
│   ├── index.ts
│   ├── plugin.ts
│   ├── toolbar.ts                  # Processes selector
│   ├── distributedSession.ts       # frontend model/client
│   ├── outputRenderer.tsx          # rank tabs / output presentation
│   ├── commands.ts                 # restart/interrupt integration
│   └── style/
│       └── index.css
├── jupyter_distributed/            # Python package
│   ├── __init__.py
│   ├── _version.py
│   ├── server_extension.py         # Jupyter Server extension registration
│   ├── handlers.py                 # API/websocket endpoints if required
│   ├── kernel_group.py             # DistributedKernelGroup
│   ├── kernel_proxy.py             # logical Jupyter-kernel facade/message routing
│   ├── rank_kernel.py              # per-rank connection/liveness wrapper
│   ├── launcher/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── torchrun.py             # initial single-node launcher
│   ├── rank_entrypoint.py          # starts persistent rank ipykernel
│   ├── protocol.py                 # rank-tagged control/message types
│   ├── breakpoint.py               # control-plane breakpoint hook
│   ├── timeout.py                  # optional interactive PG helper
│   └── diagnostics.py              # optional NCCL/rank diagnostics hooks
├── jupyter-config/
│   ├── jupyter_server_config.d/
│   └── labextensions/              # packaging metadata as appropriate
├── tests/
│   ├── unit/
│   │   ├── test_kernel_group.py
│   │   ├── test_protocol.py
│   │   └── test_breakpoint.py
│   ├── integration/
│   │   ├── test_world_size_2.py
│   │   ├── test_state_persistence.py
│   │   ├── test_group_interrupt.py
│   │   ├── test_rank_exception.py
│   │   └── test_collective.py
│   └── ui/
│       └── ...                     # Playwright/JupyterLab UI tests
└── demos/
    ├── 01_torchtitan_training.ipynb
    └── 02_transformers_tp_generation.ipynb
```

Names are provisional. Optimize for clean ownership boundaries more than exact filenames.

---

## 12. Server/API boundary

Prefer integrating through Jupyter's existing kernel/session semantics where feasible, but do not contort the implementation to make each rank separately visible to the frontend.

A distributed session should have one logical identity plus rank metadata.

If additional extension APIs are needed, keep them narrow. For example:

```text
GET  /jupyter-distributed/session/<kernel-id>
POST /jupyter-distributed/session/<kernel-id>/world-size
GET  /jupyter-distributed/session/<kernel-id>/ranks
```

Potential websocket/control channel messages:

```text
rank_status
rank_output
breakpoint_reached
breakpoint_continue
group_state
```

Use existing Jupyter protocol messages for normal execute/interrupt/etc. whenever practical; custom protocol should carry only genuinely distributed metadata/control that standard Jupyter does not represent.

---

## 13. First implementation milestones

### Milestone 1 — group launch

- Launch a notebook with world size 1 normally.
- Change toolbar selector to 2.
- Restart into two persistent rank kernels.
- Verify each sees correct `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`.
- Restart/shutdown cleans up every child process.

### Milestone 2 — SPMD execution and state persistence

Run:

```python
import os
x = int(os.environ["RANK"])
```

Then in a later cell:

```python
x
```

and show different rank-local values without reinitializing state.

### Milestone 3 — rich output aggregation

Support at least:

- stdout;
- stderr;
- `execute_result`;
- `display_data`;
- Python exceptions;

with rank tabs.

### Milestone 4 — PyTorch collective

From a notebook cell:

```python
import torch
import torch.distributed as dist
from datetime import timedelta

dist.init_process_group("nccl", timeout=timedelta(hours=24))
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

Then in another cell run a collective and prove concurrent fanout does not deadlock.

### Milestone 5 — lifecycle correctness

- one-rank Python exception;
- group interrupt while cells are busy;
- group restart;
- browser reconnect;
- notebook shutdown;
- process cleanup.

### Milestone 6 — breakpoint MVP

Ensure ordinary `breakpoint()` does not create multiple competing pdb sessions. Implement rank-0 debugger routing through the control plane or, if that is too large for the first merge, fail with an explicit distributed-aware message rather than hanging unpredictably.

### Milestone 7 — demos

The demos are part of the definition of done, not optional examples.

---

## 14. Demo 1: TorchTitan components — small interactive training run

File:

```text
demos/01_torchtitan_training.ipynb
```

### Goal

Demonstrate that a real distributed training stack can be assembled and stepped interactively across cells while state persists on every rank.

This is **not** meant to reproduce TorchTitan's full CLI trainer. It should use TorchTitan/PyTorch-native components in a notebook-native way to showcase the extension.

Current TorchTitan supports composable multi-dimensional parallelism including TP, FSDP and EP-oriented components/configuration. Its APIs are under active development, so keep the demo code isolated and pinned/tested in CI.

### Hardware target

Prefer a demo that runs on **2 GPUs** and can scale to 4. It should be small enough for common development GPUs.

### First cell: extra dependencies only

Per product requirement, all demo-only dependencies are installed in the first notebook cell. Do not put demo extras into the extension's base install.

Example shape (exact pins to be validated at implementation time):

```python
%pip install -q "torchtitan @ git+https://github.com/pytorch/torchtitan.git" sentencepiece
```

If TorchTitan's packaging requires a different supported install path at implementation time, use that. Avoid silently replacing the user's core PyTorch build unless strictly required; document any minimum/nightly PyTorch requirement in the notebook markdown adjacent to the install cell.

### Suggested notebook flow

1. **Install dependencies** (first cell).
2. Show rank/world/device identity.
3. Initialize a long-timeout process group for interactive use if TorchTitan component initialization has not already done so.
4. Build a tiny/debug model using TorchTitan model/config components or a deliberately small Llama-style configuration.
5. Construct a DeviceMesh / parallel dimensions.
6. Apply a composable distributed strategy (prefer TP + FSDP if this can remain small and stable; otherwise one representative TorchTitan sharding path is sufficient for MVP).
7. Build optimizer and synthetic or tiny local data without requiring a large dataset download.
8. Run **one training step in one cell**.
9. In following cells inspect:
   - local/sharded parameter representation;
   - rank-local parameter metadata or DTensor placement;
   - loss;
   - gradient norms or selected state.
10. Run a few more steps in a later cell, proving that the distributed model/optimizer remained alive and persistent while the user inspected it.
11. Show rank-tab output with at least one rank-specific value.

### Important demo characteristic

The notebook should deliberately pause between steps and inspect the sharded model. It should not hide the interesting work inside a call that simply launches TorchTitan as an external training script.

---

## 15. Demo 2: Accelerate + Transformers — TP text generation

File:

```text
demos/02_transformers_tp_generation.ipynb
```

### Goal

Demonstrate the primary "model larger than one device, but notebook feels ordinary" inference experience using Hugging Face tooling.

Current Transformers supports native tensor-parallel loading/inference via `tp_plan="auto"` for supported model families when launched in a torchrun-style distributed world. The model API also accepts TP/device-mesh-related configuration. Accelerate should be used for lightweight distributed environment/device/process utilities where useful; Transformers owns the actual TP partition plan.

### Hardware target

Default to **2 GPUs**.

Choose a smallish, ungated model with native Transformers TP support and weights that are reasonable to download for a demo. Candidate families include Qwen2/Qwen2.5, Phi/Phi-3, Gemma, Llama-family ungated derivatives, etc. Select and pin a concrete model only after validating current TP support and generation behavior in CI.

A reasonable target class is roughly 1B–3B parameters so the demo is fast enough to run but still meaningfully illustrates TP. Prefer a model that does not require authentication.

### First cell: extra dependencies only

Example shape (pin tested versions before release):

```python
%pip install -q "transformers>=5" "accelerate>=1" safetensors sentencepiece
```

Do not make these dependencies mandatory for the extension itself.

### Suggested notebook flow

1. **Install dependencies** (first cell).
2. Import Accelerate/Transformers and show process/rank/device identity.
3. Use `accelerate.PartialState` or the current equivalent for convenient process/device metadata if appropriate.
4. Load tokenizer.
5. Load `AutoModelForCausalLM.from_pretrained(..., tp_plan="auto", dtype=...)` inside the already-running distributed notebook world.
6. Inspect the model's TP plan / DeviceMesh / selected local shard information interactively in a separate cell.
7. Tokenize a prompt.
8. Call generation in an SPMD-safe way.
9. Decode/display user-facing generated text only from rank 0 while allowing rank tabs to expose distributed diagnostics/shape info from peers.
10. In another cell inspect one tensor-parallel layer's local parameter shape or DTensor placement.
11. Change the prompt and generate again **without reloading the model**, proving persistent interactive distributed state.

### Why Accelerate is included

The demo should genuinely import/use Accelerate for process/device/environment ergonomics, but do not force Accelerate to own the TP implementation if current Transformers native TP is the cleaner path. The demo is about interoperability with common Hugging Face workflows, not about artificially routing every operation through Accelerate.

---

## 16. Dependency policy

The core package should depend only on what is necessary to provide the Jupyter/PyTorch distributed kernel experience.

Avoid pulling TorchTitan, Transformers, Accelerate, model tokenizers, datasets, etc. into base dependencies.

Demo-specific packages must be installed in the **first code cell of each notebook** using `%pip install ...`.

The demo notebooks should be self-describing and runnable from a fresh environment that already has:

- a compatible JupyterLab/Jupyter Server;
- this extension installed;
- a compatible CUDA PyTorch build;
- enough GPUs for the selected world size.

---

## 17. Testing strategy

### Unit tests

Focus on deterministic logic:

- group-state aggregation;
- rank-output tagging;
- logical busy/idle calculation;
- execution-ID mapping;
- restart world-size handling;
- breakpoint coordination state machine;
- cleanup behavior.

### Integration tests

CPU/Gloo tests where possible:

- world size 2 launch;
- environment correctness;
- state persistence across cells;
- concurrent collective;
- one-rank exception;
- interrupt all;
- restart all;
- shutdown cleanup.

GPU/NCCL tests gated in CI/infrastructure:

- 2-GPU NCCL collective;
- TP smoke test;
- long idle gap followed by successful cell execution;
- optional controlled collective timeout diagnostic test.

### UI tests

Use the JupyterLab-supported browser testing stack (typically Playwright in the Jupyter ecosystem) for:

- process selector appears;
- changing process count prompts/restarts;
- rank tabs render;
- error tab highlighting;
- kernel interrupt/restart commands act on the whole group;
- reconnection retains group metadata.

### Demo smoke tests

At minimum, validate notebook code periodically against pinned known-good versions. Both ecosystems evolve quickly, especially TorchTitan.

---

## 18. MVP non-goals

Do not let these delay the first useful version:

- multi-node execution;
- TorchX launcher;
- Slurm/Kubernetes launcher;
- elastic world-size mutation;
- arbitrary rank-targeted execution as the default mode;
- full distributed variable explorer;
- tensor diff UI;
- custom debugger GUI;
- automatic recovery of a corrupted NCCL process group;
- support for every Jupyter frontend besides JupyterLab;
- replacing nbformat;
- generalized task scheduling/load balancing (ipyparallel territory).

---

## 19. Future extensions

After the single-node UX is solid:

### Multi-node launchers

Add a launcher abstraction behind `DistributedKernelGroup`:

```python
class Launcher(Protocol):
    async def start(world_spec: WorldSpec) -> RankSet: ...
    async def terminate(...) -> None: ...
```

Potential implementations:

- local torchrun;
- torch elastic rendezvous across nodes;
- TorchX;
- Slurm;
- Kubernetes.

The notebook/session/output protocol should not need redesign for multinode.

### Better distributed inspection

- rank selection toolbar;
- all-rank/diff output modes;
- DTensor-aware display;
- parameter placement visualizer;
- DeviceMesh visualization;
- rank-local memory metrics;
- NCCL flight-recorder surfacing;
- "which rank is blocking this cell?" status UI.

### Debugger evolution

- UI-selected debug rank;
- multi-rank breakpoint arrival visualization;
- continue/step controls integrated into Lab;
- safe handling of conditional breakpoints;
- stack inspection for non-active ranks if practical.

---

## 20. Key invariants for implementation agents

Treat these as product requirements, not suggestions:

1. **SPMD first:** ordinary code cells execute on all ranks by default.
2. **Persistent state:** each rank keeps its Python namespace across cells.
3. **One logical kernel:** users should not manage N notebook sessions.
4. **World size is kernel configuration:** changing it restarts the whole world.
5. **Control plane is independent of NCCL/user collectives.**
6. **No implicit `dist.barrier()` after cells.** Cell completion is coordinated externally.
7. **Interrupt/restart/shutdown are group operations.**
8. **Do not impose DDP or any other parallelism strategy.** TP/EP/FSDP/DeviceMesh are first-class intended uses.
9. **Regular `breakpoint()` must not create N competing pdb sessions.**
10. **Long interactive pauses are normal.** Timeout defaults/guidance must account for this without globally disabling genuine hang detection.
11. **Core dependencies stay small.** TorchTitan/Transformers/Accelerate are demo-only dependencies installed in notebook cell 1.
12. **Both demos are part of the deliverable.** They must show persistent interactive distributed state, not shell out to scripts.
13. **Do not fork Jupyter.** Use JupyterLab and Jupyter Server extension points unless a concrete blocker is proven.

---

## 21. Initial acceptance criteria

The first release candidate is successful when all of the following work on one machine:

- `pip install` the project and launch `jupyter lab` normally;
- open a standard Python notebook and see `Processes: 1`;
- change it to `2` and restart into a two-rank world;
- run `RANK/WORLD_SIZE` inspection and see correct per-rank output tabs;
- define rank-local state in one cell and inspect/reuse it in later cells;
- initialize a torch distributed process group and execute a collective across notebook cells;
- build a distributed model and leave it resident while running/inspecting multiple later cells;
- group interrupt, restart, and shutdown work without orphan rank processes;
- a Python error on one rank is surfaced without losing all other rank output;
- ordinary `breakpoint()` does not devolve into multiple stdin-contending pdb instances;
- TorchTitan demo performs a small interactive distributed training run;
- Transformers + Accelerate demo loads a TP-split causal LM, generates text, inspects local TP state, changes prompt, and generates again without model reload.

---

## 22. Current upstream references checked for this design

These are implementation references, not hard API contracts; upstream libraries move quickly.

- JupyterLab common extension points / notebook toolbar: https://jupyterlab.readthedocs.io/en/latest/extension/extension_points.html
- JupyterLab kernel lifecycle APIs: https://jupyterlab.readthedocs.io/en/stable/api/functions/services.KernelAPI.restartKernel.html
- PyTorch distributed/process-group timeout and `torch.distributed.breakpoint`: https://docs.pytorch.org/docs/stable/distributed.html
- PyTorch ProcessGroupNCCL diagnostics / flight recorder: https://docs.pytorch.org/tutorials/prototype/flight_recorder_tutorial.html
- TorchTitan repository and current component/config structure: https://github.com/pytorch/torchtitan
- TorchTitan configuration model: https://github.com/pytorch/torchtitan/blob/main/torchtitan/config/README.md
- Transformers tensor parallel inference: https://huggingface.co/docs/transformers/main/perf_infer_gpu_multi
- Transformers model `tp_plan` / `device_mesh` parameters: https://huggingface.co/docs/transformers/en/main_classes/model
- Transformers + Accelerate integration overview: https://huggingface.co/docs/transformers/accelerate

---

## 23. Implementation philosophy

Bias toward the simplest architecture that preserves the illusion that this is still an ordinary eager notebook.

The project is successful if a researcher can forget that Jupyter historically assumed a single Python process and can interact with a sharded model in the same exploratory rhythm as a single-GPU model:

```python
model = build_model()
model = parallelize(model)
```

inspect it, execute a forward pass, inspect activations, run a step, examine gradients, edit code, run another step — while the distributed world simply remains alive behind the notebook.

That is the product.
