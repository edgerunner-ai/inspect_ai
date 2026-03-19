# vLLM Batch Provider for Inspect AI

## Overview

This fork adds a `vllm_batch` model provider to Inspect AI that uses `vllm run-batch` for offline batch inference. It enables a **two-phase evaluation flow** optimized for GPU utilization:

- **Phase 1 (Generation)**: All tasks run generation using the evaluated model with all GPUs.
- **Phase 2 (Scoring)**: All completed logs are scored using the judge model with all GPUs.

This is significantly faster than the default flow where model and judge share GPUs or run sequentially per-task.

## Repository Layout


| Repository                    | Branch                        | Commit    | Description                                                      |
| ----------------------------- | ----------------------------- | --------- | ---------------------------------------------------------------- |
| `edgerunner-ai/inspect_ai`    | `feature/vllm-batch-provider` | `a00a5f5` | Fork with vllm_batch provider + sequential scoring               |
| `edgerunner-ai/Battle-tester` | `inspect-ai`                  | `7e3c893` | Task files updated for provider-agnostic scorer model resolution |


### Upstream

The inspect_ai fork is based on `UKGovernmentBEIS/inspect_ai`. To sync with upstream:

```bash
cd /mnt/weka/aris/inspect_ai_fork
git remote add upstream git@github.com:UKGovernmentBEIS/inspect_ai.git  # if not already added
git fetch upstream
git rebase upstream/main  # or merge, depending on preference
# Resolve any conflicts in the files listed below, then push
git push origin feature/vllm-batch-provider --force-with-lease
```

## Files Changed in inspect_ai Fork


| File                                            | Change                                                                                                                                                                                          |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/inspect_ai/model/_providers/vllm_batch.py` | **New file.** `VLLMBatchAPI` provider using `vllm run-batch` with shared batching across instances.                                                                                             |
| `src/inspect_ai/model/_providers/providers.py`  | Registered `vllm_batch` provider via `@modelapi(name="vllm_batch")`.                                                                                                                            |
| `src/inspect_ai/_eval/eval.py`                  | Added `sequential_scoring` parameter through eval/eval_async/eval_set_async. Auto-detects `vllm_batch` provider and enables sequential scoring. Sets `parallel = task_definitions` when active. |
| `src/inspect_ai/_eval/run.py`                   | Added `run_sequential_scoring()` function: Phase 1 runs generation without scoring, Phase 2 scores all logs concurrently via `asyncio.gather`.                                                  |
| `src/inspect_ai/_cli/eval.py`                   | Added `--sequential-scoring` CLI flag (optional, auto-enabled for `vllm_batch`).                                                                                                                |
| `src/inspect_ai/log/_log.py`                    | Added `sequential_scoring` field to `EvalConfig`.                                                                                                                                               |


## Files Changed in Battle-tester

All 18 task files under `src/battle_tester/tasks/model_graded_qa/` were updated:

**Before:**

```python
elif scorer_model == "vllm/local" and scorer_model_path:
    model = get_model(model="vllm/" + scorer_model_path, **kwargs)
```

**After:**

```python
elif scorer_model and scorer_model.endswith("/local") and scorer_model_path:
    provider = scorer_model.split("/")[0]
    model = get_model(model=f"{provider}/{scorer_model_path}", **kwargs)
```

This change is backwards-compatible: `vllm/local` still works exactly as before, while also supporting `vllm_batch/local`.

## Environment Setup

### Virtual Environment

```bash
# Location
/mnt/weka/venv/inspect_batch/

# Python
Python 3.13.5 (from /mnt/weka/venv/python-installs/cpython-3.13.5-linux-x86_64-gnu/)
```

### Library Versions


| Package      | Version          | Notes                                  |
| ------------ | ---------------- | -------------------------------------- |
| `torch`      | `2.10.0+cu128`   | CUDA 12.8                              |
| `vllm`       | `0.17.0`         |                                        |
| `flash-attn` | `2.8.3`          | Pre-built wheel for cu128 + torch 2.10 |
| `inspect_ai` | editable install | From this fork                         |


### Flash Attention Wheels

Pre-built wheels from [https://github.com/mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels):

- **CUDA 12.8 + torch 2.10**: `flash_attn-2.8.3+cu128torch2.10-cp313-cp313-linux_x86_64.whl`
- **CUDA 12.9 + torch 2.9**: `flash_attn-2.8.3+cu129torch2.9-cp313-cp313-linux_x86_64.whl`

### Install Commands

```bash
# Create venv
/mnt/weka/venv/python-installs/cpython-3.13.5-linux-x86_64-gnu/bin/python -m venv /mnt/weka/venv/inspect_batch
source /mnt/weka/venv/inspect_batch/bin/activate
pip install uv

# Install torch + vllm
uv pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install vllm==0.17.0

# Install flash-attn (match your CUDA + torch version)
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp313-cp313-linux_x86_64.whl

# Install inspect_ai fork in editable mode
cd /mnt/weka/aris/inspect_ai_fork
uv pip install -e ".[dev]"

# Install battle_tester
cd /mnt/weka/aris/Battle-tester
uv pip install -e .
```

## Usage

### CLI (eval-set)

```bash
inspect eval-set \
  src/battle_tester/tasks/model_graded_qa \
  --model vllm_batch/local \
  -M model_path=/path/to/evaluated/model \
  -M tensor_parallel_size=2 \
  -M data_parallel_size=4 \
  -T scorer_model=vllm_batch/local \
  -T scorer_model_path=/path/to/judge/model \
  -T tensor_parallel_size=4 \
  -T data_parallel_size=2 \
  --display plain \
  --log-dir /path/to/logs
```

Key points:

- `vllm_batch/local` as model name triggers the `vllm_batch` provider.
- `-M key=value` passes arguments to the evaluated model.
- `-T key=value` passes arguments to tasks (and through `**kwargs` to the scorer model).
- `--sequential-scoring` is **not needed** -- it is auto-enabled when `vllm_batch` is detected.
- `tensor_parallel_size` and `data_parallel_size` control vLLM's parallelism. TP x DP should equal your total GPU count.

### How It Works

1. **VLLMBatchAPI** collects all generation requests into a queue, waits for a configurable delay (`batch_send_delay`, default 30s), then writes them to a JSONL file and invokes `vllm run-batch`.
2. **Shared batching**: All `VLLMBatchAPI` instances for the same model share a single request queue (via `_SharedBatchState`), so multiple tasks' requests are batched into one `vllm run-batch` call.
3. **Sequential scoring**: After all generation completes and the vLLM process exits, `run_sequential_scoring()` reads the logs from disk and scores them concurrently, triggering a second `vllm run-batch` for the judge.

### Configurable Parameters


| Parameter              | Default        | Description                                                                                 |
| ---------------------- | -------------- | ------------------------------------------------------------------------------------------- |
| `batch_send_delay`     | `30` (seconds) | Time to wait for more requests before dispatching a batch. Set via `-M batch_send_delay=N`. |
| `tensor_parallel_size` | `1`            | Number of GPUs per model replica (tensor parallelism).                                      |
| `data_parallel_size`   | `1`            | Number of model replicas (data parallelism).                                                |
| `max_model_len`        | auto           | Max sequence length. Reduce to save GPU memory (e.g., `8192` for 80GB H100 with 20B model). |


## Known Behaviors

- **Generation vs scoring request counts may differ** for tasks with custom scorers that bypass the LLM judge (e.g., `mil_deflect_gold_alpha` uses substring matching to skip the judge when no rejection markers are found).
- **FlashInfer cleanup warnings** (`ImportError: sys.meta_path is None`) during vLLM process shutdown are harmless.
- **CUDA event destruction warnings** under memory pressure are non-fatal but indicate throughput may be degraded; reduce `max_model_len` or increase TP.

