# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

vLLM-Omni is an efficient omni-modality model inference and serving framework that extends vLLM to support:
- Non-autoregressive architectures (especially Diffusion Transformers/DiT)
- Multi-modal inputs/outputs (text, image, video, audio)
- Disaggregated multi-stage pipeline execution

The codebase is built on vLLM's efficient autoregressive (AR) support and adds native diffusion model support with extensive acceleration techniques.

## Development Setup

### Environment Setup

Use `uv` as the environment manager (recommended):

```bash
# Create virtual environment with Python 3.12 (recommended)
uv venv --python 3.12 --seed
source .venv/bin/activate

# Install in development mode with dev dependencies
uv pip install -e ".[dev]"

# Install ffmpeg for multimedia tests
apt-get install -y ffmpeg  # or brew install ffmpeg on macOS

# Install pre-commit hooks
uv pip install pre-commit
pre-commit install
```

### Platform-Specific Installation

The build system (`setup.py`) automatically detects your hardware platform and installs appropriate dependencies from `requirements/`:
- `cuda.txt` - NVIDIA GPUs (default)
- `rocm.txt` - AMD GPUs
- `npu.txt` - Ascend NPUs
- `xpu.txt` - Intel GPUs
- `cpu.txt` - CPU-only

Override detection with: `export VLLM_OMNI_TARGET_DEVICE=cuda|rocm|npu|xpu|cpu`

## Common Development Commands

### Testing

```bash
cd tests

# L1 tests (CPU, run on every PR)
pytest -s -v -m "core_model and cpu"

# L2 tests (GPU, run on every PR)
pytest -s -v -m "core_model and not cpu" --run-level=core_model

# L3/L4 tests (advanced models, run on merge/nightly)
pytest -s -v -m "advanced_model" --run-level=advanced_model

# Run specific test file
pytest -s -v tests/test_xxx.py --run-level=core_model

# Run tests for specific platform/hardware
pytest -s -v -m "core_model and distributed_cuda and L4" --run-level=core_model
```

Test markers (defined in `pyproject.toml`):
- **CI/CD levels**: `core_model` (L1/L2), `advanced_model` (L3/L4)
- **Platforms**: `cpu`, `gpu`, `cuda`, `rocm`, `npu`
- **Hardware**: `H100`, `L4`, `MI325`, `A2`, `A3`
- **Features**: `diffusion`, `omni`, `cache`, `parallel`

### Linting and Formatting

```bash
# Run pre-commit hooks manually
pre-commit run                    # staged files only
pre-commit run --all-files        # all files

# Bypass pre-commit (use sparingly)
git commit --no-verify

# Skip specific hook
SKIP=ruff-check git commit
```

Pre-commit runs:
- `ruff-check` and `ruff-format` (Python linting/formatting)
- `typos` (spell checking)
- `actionlint` (GitHub Actions linting)
- `signoff-commit` (automatic DCO sign-off)

### Documentation

```bash
# Install docs dependencies
uv pip install -e ".[docs]"

# Serve documentation locally with live reload
mkdocs serve                              # ~10 min (includes API reference)
API_AUTONAV_EXCLUDE=vllm_omni mkdocs serve  # ~15 sec (excludes API reference)

# Build documentation
mkdocs build
```

Documentation uses MkDocs with Material theme and lives in `docs/`.

### Running Models

```bash
# Offline inference (Python API)
python examples/offline_inference/qwen3_omni/run_single_prompt.sh

# Online serving (API server)
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091

# With custom stage configs
vllm serve <model> --omni --stage-configs-path /path/to/custom_config.yaml

# Common CLI options
vllm serve <model> --omni \
  --port 8091 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code
```

## Architecture

### Multi-Stage Pipeline System

vLLM-Omni uses a **stage-based architecture** where models are decomposed into multiple stages, each running in separate processes with dedicated GPU resources:

**Key components:**
- **OmniRouter**: Routes omni-modality requests to appropriate stages
- **EntryPoints** (`vllm_omni/entrypoints/`):
  - `Omni` / `AsyncOmni`: Python API for offline/online inference
  - `omni_stage.py`: OmniStage abstraction for AR/DiT stages
  - `openai/`: OpenAI-compatible API server
- **AR Module**: Autoregressive stages with efficient KV cache (inherits from vLLM)
- **Diffusion Module** (`vllm_omni/diffusion/`): Native DiT implementation with acceleration
- **OmniConnector**: Inter-stage communication (shared memory, Mooncake, etc.)

### Stage Configurations

Stage configs (YAML files in `vllm_omni/model_executor/stage_configs/`) define:
- Stage partitioning and class implementations
- Disaggregation topology (which GPU, process isolation)
- Engine arguments (memory, parallelism, scheduling)
- Input/output dependencies between stages
- Default sampling parameters

Example models:
- **Qwen2.5-Omni / Qwen3-Omni**: AR + DiT (thinker → talker → code2wav stages)
- **BAGEL**: AR-main with DiT for visual generation
- **GLM-Image**: DiT-main with AR text encoder
- **Qwen3-TTS**: Text-to-speech pipeline

Critical stage config fields:
- `stage_id`: Unique identifier for dependency resolution
- `runtime.devices`: GPU assignment (e.g., "0", "1", "0,1")
- `runtime.process`: Run in separate process (usually `true`)
- `engine_args.model_arch`: Registered model class in `model_executor/models/registry.py`
- `engine_args.worker_type`: `ar` or `generation` worker
- `engine_input_source`: List of upstream stage IDs

### Directory Structure

```
vllm_omni/
├── core/sched/           # Schedulers (OmniARScheduler, OmniGenerationScheduler)
├── diffusion/            # Diffusion engine, cache, attention, quantization
├── entrypoints/          # API servers, Omni/AsyncOmni, CLI
├── engine/               # Engine implementations
├── model_executor/       # Model implementations, stage configs, workers
│   ├── models/           # Model implementations (registered in registry.py)
│   └── stage_configs/    # Default YAML configs for supported models
├── distributed/          # Distributed inference (TP, PP, DP, EP)
├── platforms/            # Platform-specific code (CUDA, ROCm, NPU, XPU)
├── worker/               # GPU workers (ARWorker, GenerationWorker)
└── config/               # Configuration classes
```

### Model Implementation

When adding new models:
- Diffusion models: See `docs/contributing/model/adding_diffusion_model.md`
- Omni models: See `docs/contributing/model/adding_omni_model.md`
- TTS models: See `docs/contributing/model/adding_tts_model.md`

All models must be registered in `vllm_omni/model_executor/models/registry.py`.

## Pull Request Guidelines

### PR Title Prefixes

Use one of the following prefixes:
- `[Bugfix]` - Bug fixes
- `[CI/Build]` - Build or CI improvements
- `[Doc]` - Documentation changes
- `[Model]` - New model or model improvements (include model name)
- `[Frontend]` - API server, OmniLLM class changes
- `[Kernel]` - CUDA kernel or compute kernel changes
- `[Core]` - Core logic (OmniProcessor, schedulers, etc.)
- `[Hardware][Vendor]` - Hardware-specific (e.g., `[Hardware][Ascend]`)
- `[Misc]` - Other changes (use sparingly)

### Commit Sign-off (DCO)

All commits must include a `Signed-off-by:` line:

```bash
git commit -s -m "Your commit message"
```

Pre-commit hooks automatically add sign-off if missing.

### Before Submitting

1. Run L1/L2 tests locally and include results in PR description
2. Ensure all pre-commit hooks pass
3. Add tests for new functionality
4. Update `docs/` if user-facing behavior changes
5. For large changes (>500 LOC), create an RFC issue first

## Key Concepts

### Classifier-Free Guidance (CFG) Flow

For diffusion models, CFG is handled via "companion requests":
1. `prompt_expand_func`: Expands prompt into positive + negative (cfg_text) pairs
2. AR stage processes both in parallel, outputs KV caches
3. OmniConnector transfers both KV caches to DiT stage
4. `cfg_kv_collect_func`: Collects cfg_text KV caches for CFG guidance

### Disaggregated Inference

Stages can run in separate processes on different GPUs:
- E (Encoding) / P (Processing) / D (Decoding) / G (Generation) disaggregation
- Configured via `runtime.process`, `runtime.devices`, `runtime.max_batch_size`
- Communication via OmniConnector (shared memory, Mooncake, YuanRong)

### Parallelism

- **Tensor Parallel (TP)**: Split model across GPUs
- **Pipeline Parallel (PP)**: Split stages across GPUs
- **CFG Parallel (CP)**: Parallel CFG computation
- **Sequence Parallel (SP/USP)**: Sequence-level parallelism
- **Data Parallel**: Batch-level parallelism

## Additional Resources

- [Architecture Overview](docs/design/architecture_overview.md)
- [Contributing Guide](docs/contributing/README.md)
- [Test Guide](docs/contributing/ci/test_guide.md)
- [Examples](examples/)
- [User Forum](https://discuss.vllm.ai)
- [Developer Slack](https://slack.vllm.ai) - #sig-omni channel
