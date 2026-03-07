# 📚 Comprehensive Learning Plan for vLLM-Omni

## Overview

This tutorial provides a complete learning path for understanding the vLLM-Omni repository, from foundational concepts to advanced implementation details. Follow this guide step-by-step to gain comprehensive knowledge of the codebase.

---

## Phase 1: Foundation & Concepts (1-2 days)

### Step 1: Understand the "Why" and High-Level Architecture
**Goal**: Grasp the motivation and big picture before diving into code.

1. **Read these documents in order**:
   - `README.md` - Project overview and goals
   - `docs/design/architecture_overview.md` - Core architectural concepts
   - `docs/design/index.md` - Design philosophy

2. **Key concepts to internalize**:
   - Why vLLM-Omni exists (extending vLLM for omni-modality)
   - Multi-stage pipeline architecture
   - Disaggregated inference model
   - AR (Autoregressive) vs DiT (Diffusion Transformer) stages

3. **Draw a mental model**:
   - OmniRouter → EntryPoints → OmniStage → Engine → Worker → Model
   - Understand data flow: Request → Stage 0 → OmniConnector → Stage 1 → Output

---

## Phase 2: Hands-On Experience (1-2 days)

### Step 2: Set Up Development Environment
**Goal**: Get the system running on your machine.

```bash
# Follow this exact sequence
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

### Step 3: Run Simple Examples
**Goal**: See the system in action before understanding internals.

**Start with the simplest example**:
```bash
# 1. Qwen2.5-Omni single prompt (if you have GPU access)
cd examples/offline_inference/qwen2_5_omni
python end2end.py
```

**Observe**:
- How long it takes to load
- Console output showing stage initialization
- Output format (text + audio if applicable)

**If no GPU, read these files instead**:
- `examples/offline_inference/qwen2_5_omni/end2end.py`
- `examples/offline_inference/qwen3_omni/end2end.py`

### Step 4: Understand Stage Configurations
**Goal**: Learn how pipelines are configured.

1. **Read a complete stage config**:
   - Open: `vllm_omni/model_executor/stage_configs/qwen2_5_omni.yaml`

2. **Identify for each stage**:
   - `stage_id` - Sequential identifier
   - `runtime.devices` - Which GPU
   - `engine_args.model_stage` - Stage name (thinker/talker/code2wav)
   - `engine_args.worker_type` - AR vs generation
   - `engine_input_source` - Which stage feeds this one
   - `default_sampling_params` - Generation settings

3. **Read the documentation**:
   - `docs/configuration/stage_configs.md` - Complete reference

4. **Compare different models**:
   - `qwen2_5_omni.yaml` - 3-stage AR+AR+Generation pipeline
   - `qwen3_tts.yaml` - 2-stage TTS pipeline
   - `glm_image.yaml` - DiT-based image generation

---

## Phase 3: Code Flow Understanding (2-3 days)

### Step 5: Trace a Request End-to-End
**Goal**: Follow a single request through the entire system.

**Start point**: `vllm_omni/entrypoints/cli/main.py`

```
User runs: vllm serve Qwen/Qwen2.5-Omni-7B --omni
    ↓
1. cli/main.py:main() → routes to OmniServeCommand
2. cli/serve.py:OmniServeCommand.run() → starts API server
3. openai/api_server.py:app → receives HTTP request
4. openai/serving_chat.py:create_chat_completion() → processes request
5. entrypoints/async_omni.py:AsyncOmni.generate() → orchestrates stages
6. entrypoints/omni_stage.py:OmniStage → manages stage execution
7. core/sched/omni_ar_scheduler.py → schedules batches
8. worker/gpu_model_runner.py → executes model
9. distributed/omni_connectors/ → transfers to next stage
10. Back through stages until final output
```

**Files to read in this order**:
1. `vllm_omni/entrypoints/cli/serve.py` (entry point)
2. `vllm_omni/entrypoints/async_omni.py` (orchestrator - focus on `generate()` method)
3. `vllm_omni/entrypoints/omni_stage.py` (stage manager - focus on `run_omni_stage()`)
4. `vllm_omni/core/sched/omni_ar_scheduler.py` (scheduling logic)
5. `vllm_omni/worker/gpu_model_runner.py` (model execution)

**Exercise**: Create a flowchart of the request lifecycle.

### Step 6: Understand Data Structures
**Goal**: Know how data is represented at each stage.

**Read these files**:
1. `vllm_omni/request.py` - `OmniRequest` class
   - How requests carry embeddings
   - How additional_information works for inter-stage data

2. `vllm_omni/inputs/data.py` - Input types
   - `OmniTextPrompt`, `OmniTokensPrompt`, `OmniEmbedsPrompt`
   - `OmniSamplingParams`

3. `vllm_omni/outputs.py` - `OmniRequestOutput`
   - How outputs are structured
   - Difference between intermediate and final outputs

4. `vllm_omni/config/model.py` - `OmniModelConfig`
   - Stage-specific configuration
   - How `engine_output_type` determines data flow

**Exercise**: Write pseudocode for how data transforms between stages.

### Step 7: Deep Dive into Stage Communication
**Goal**: Understand how stages talk to each other.

**Read**:
1. `vllm_omni/distributed/omni_connectors/` directory
   - `base.py` - Abstract connector interface
   - `shared_memory_connector.py` - Fastest for single-machine
   - `zmq_connector.py` - Network-based communication

2. `vllm_omni/engine/output_processor.py`
   - How outputs are serialized
   - `PromptEmbedsPayload` structure

3. `vllm_omni/engine/input_processor.py`
   - How next stage receives data
   - Embedding reconstruction

**Key concept**: Inter-stage data flow uses serialized embeddings/features, not just text.

---

## Phase 4: Model Implementation (2-3 days)

### Step 8: Analyze a Complete Model Implementation
**Goal**: Understand how models are structured in vLLM-Omni.

**Choose Qwen2.5-Omni as your reference model**:

1. **Model architecture**:
   - `vllm_omni/model_executor/models/qwen2_5_omni/modeling_qwen2_5_omni.py`
   - Study the `Qwen2_5OmniForConditionalGeneration` class
   - Understand how different stages (thinker/talker/code2wav) are implemented

2. **Stage input processors**:
   - `vllm_omni/model_executor/stage_input_processors/qwen2_5_omni.py`
   - Focus on `thinker2talker()` - how stage 0 output becomes stage 1 input

3. **Model registry**:
   - `vllm_omni/model_executor/models/registry.py`
   - How models are registered and discovered

**Compare with another model** (choose based on interest):
- **Diffusion**: `vllm_omni/diffusion/models/` (e.g., `hunyuan_image3/`)
- **TTS**: `vllm_omni/model_executor/models/qwen3_tts/`
- **Audio**: `vllm_omni/model_executor/models/mimo_audio/`

### Step 9: Understand Schedulers
**Goal**: Know how request batching and scheduling works.

**Read**:
1. `vllm_omni/core/sched/omni_ar_scheduler.py`
   - `OmniARScheduler` - For autoregressive stages
   - Key methods: `schedule()`, `_allocate_and_set_running()`

2. `vllm_omni/core/sched/omni_generation_scheduler.py`
   - `OmniGenerationScheduler` - For non-AR generation
   - Understand the difference from AR scheduling

**Key questions to answer**:
- How does the scheduler decide what to batch together?
- How is KV cache managed?
- What's the difference between AR and generation scheduling?

### Step 10: Workers and Execution
**Goal**: Understand how models actually run on GPUs.

**Read**:
1. `vllm_omni/worker/base.py`
   - `OmniGPUWorkerBase` - Base class
   - GPU memory tracking
   - Process lifecycle

2. `vllm_omni/worker/gpu_model_runner.py`
   - `OmniGPUModelRunner` - Actual model execution
   - Forward pass implementation
   - Metadata handling

3. `vllm_omni/worker/gpu_ar_worker.py` and `gpu_generation_worker.py`
   - Specialized workers for different stage types

---

## Phase 5: Advanced Topics (3-5 days)

### Step 11: Diffusion Models
**Goal**: Understand non-autoregressive architectures.

**Read in order**:
1. `docs/design/module/dit_module.md` - Diffusion Transformer concepts
2. `vllm_omni/diffusion/diffusion_engine.py` - Engine implementation
3. `vllm_omni/diffusion/data.py` - Diffusion-specific data structures
4. `docs/user_guide/diffusion/` - User-facing documentation

**Example to run**:
```bash
cd examples/offline_inference/text_to_image
python end2end.py
```

### Step 12: Parallelism and Distribution
**Goal**: Learn how vLLM-Omni scales across GPUs.

**Read**:
1. `docs/design/feature/tensor_parallel.md` - Tensor parallelism
2. `docs/design/feature/sequence_parallel.md` - Sequence parallelism
3. `docs/design/feature/cfg_parallel.md` - CFG parallelism
4. `vllm_omni/distributed/` - Distribution implementations

**Advanced concepts**:
- VAE parallelism: `docs/design/feature/vae_parallel.md`
- HSDP: `docs/design/feature/hsdp.md`
- Disaggregated inference: `docs/design/feature/disaggregated_inference.md`

### Step 13: Optimization and Acceleration
**Goal**: Understand performance optimizations.

**Read**:
1. `docs/design/feature/cache_dit.md` - DiT caching
2. `docs/design/feature/teacache.md` - TeaCache optimization
3. `docs/design/feature/async_chunk_design.md` - Async chunking for TTS
4. `vllm_omni/diffusion/cache/` - Cache implementations

**Profiling**:
- `docs/contributing/profiling.md` - How to profile performance
- `vllm_omni/benchmarks/` - Benchmark examples

### Step 14: API Server and OpenAI Compatibility
**Goal**: Understand the serving layer.

**Read**:
1. `vllm_omni/entrypoints/openai/api_server.py` - FastAPI server setup
2. `vllm_omni/entrypoints/openai/serving_chat.py` - Chat completions
3. `vllm_omni/entrypoints/openai/serving_video.py` - Video generation
4. `vllm_omni/entrypoints/openai/serving_speech.py` - Speech/audio
5. `vllm_omni/entrypoints/openai/protocol.py` - Request/response schemas

**Test with curl**:
```bash
# Start server
vllm serve <model> --omni --port 8091

# Send request (see examples/online_serving/*/run_curl_*.sh)
```

---

## Phase 6: Contributing & Testing (2-3 days)

### Step 15: Understand the Test Framework
**Goal**: Learn how to write and run tests.

**Read**:
1. `docs/contributing/ci/test_guide.md` - Complete test guide
2. `docs/contributing/ci/tests_markers.md` - Test markers
3. `docs/contributing/ci/CI_5levels.md` - CI/CD levels

**Run tests**:
```bash
cd tests

# L1 (CPU)
pytest -s -v -m "core_model and cpu"

# L2 (GPU) - if you have GPU
pytest -s -v -m "core_model and not cpu" --run-level=core_model

# Specific test
pytest -s -v tests/e2e/test_qwen2_5_omni.py -k "test_name"
```

**Explore test structure**:
- `tests/e2e/` - End-to-end tests
- `tests/unit/` - Unit tests
- `tests/perf/` - Performance tests
- `tests/conftest.py` - Pytest configuration

### Step 16: Add a New Model (Practical Exercise)
**Goal**: Apply your knowledge by adding a model.

**Follow these guides**:
1. `docs/contributing/model/README.md` - Overview
2. Choose based on model type:
   - `docs/contributing/model/adding_omni_model.md` - Omni models
   - `docs/contributing/model/adding_diffusion_model.md` - Diffusion models
   - `docs/contributing/model/adding_tts_model.md` - TTS models

**Steps**:
1. Create model class in `vllm_omni/model_executor/models/<your_model>/`
2. Register in `model_executor/models/registry.py`
3. Create stage config YAML
4. Write input processor if multi-stage
5. Add tests in `tests/e2e/`
6. Update documentation

### Step 17: Review Best Practices
**Goal**: Understand code quality standards.

**Read**:
1. `docs/contributing/README.md` - Contributing guidelines
2. `.pre-commit-config.yaml` - Linting rules
3. `pyproject.toml` - Ruff and mypy configuration
4. `docs/contributing/DOCS_GUIDE.md` - Documentation standards

**Key practices**:
- Always run `pre-commit run --all-files` before committing
- Add type annotations (mypy enforced)
- Follow test marker conventions
- DCO sign-off on commits (`git commit -s`)

---

## Phase 7: Deep Expertise (Ongoing)

### Step 18: Platform-Specific Code
**Goal**: Understand multi-platform support.

**Explore**:
- `vllm_omni/platforms/cuda/` - NVIDIA GPU
- `vllm_omni/platforms/rocm/` - AMD GPU
- `vllm_omni/platforms/npu/` - Ascend NPU
- `vllm_omni/platforms/xpu/` - Intel GPU

**Each platform has**:
- Custom stage configs
- Platform-specific kernels
- Memory management adaptations

### Step 19: Advanced Features
**Goal**: Master complex features.

**Topics**:
1. **CFG Companion Flow**:
   - Read: `docs/design/architecture_overview.md` (CFG section)
   - Study `vllm_omni/entrypoints/cfg_companion_tracker.py`
   - Understand `prompt_expand_func` and `cfg_kv_collect_func`

2. **Ray-Based Execution**:
   - Read: `docs/design/feature/ray_based_execution.md`
   - Distributed execution across machines

3. **Custom Pipelines**:
   - Read: `docs/features/custom_pipeline.md`
   - `examples/offline_inference/custom_pipeline/`

4. **ComfyUI Integration**:
   - Read: `docs/features/comfyui.md`
   - `apps/ComfyUI-vLLM-Omni/`

### Step 20: Research Papers and Design Docs
**Goal**: Understand theoretical foundations.

**Read**:
1. vLLM-Omni paper: [arXiv:2602.02204](https://arxiv.org/abs/2602.02204)
2. All files in `docs/design/feature/`
3. All files in `docs/design/module/`

---

## Learning Path Recommendations

### **For Different Goals:**

**If you want to USE vLLM-Omni**:
- Focus on: Phase 1-2, Step 3-4, Step 14
- Skip: Phase 4-5 (implementation details)
- Time: 2-3 days

**If you want to ADD MODELS**:
- Follow: Phase 1-4, Step 16
- Focus heavily on: Step 8, Step 13
- Time: 1-2 weeks

**If you want to CONTRIBUTE TO CORE**:
- Complete all phases
- Extra focus on: Phase 5, Step 11-13
- Time: 2-3 weeks

**If you want to OPTIMIZE PERFORMANCE**:
- Focus on: Step 13, Step 15, Phase 7
- Study: `vllm_omni/diffusion/cache/`, `vllm_omni/diffusion/attention/`
- Time: 1-2 weeks

---

## Quick Reference Cheat Sheet

```python
# Key Files for Quick Reference
ARCHITECTURE:     docs/design/architecture_overview.md
STAGE CONFIGS:    docs/configuration/stage_configs.md
REQUEST FLOW:     vllm_omni/entrypoints/async_omni.py
STAGE EXECUTION:  vllm_omni/entrypoints/omni_stage.py
SCHEDULERS:       vllm_omni/core/sched/
MODELS:           vllm_omni/model_executor/models/
WORKERS:          vllm_omni/worker/
API SERVER:       vllm_omni/entrypoints/openai/

# Common Commands
SERVE:            vllm serve <model> --omni
TEST L2:          pytest -s -v -m "core_model and not cpu" --run-level=core_model
LINT:             pre-commit run --all-files
DOCS:             mkdocs serve
```

---

## Recommended Learning Order Summary

```
Week 1: Foundation
├─ Day 1-2: Phase 1 (Concepts) + Phase 2 (Setup & Examples)
└─ Day 3-5: Phase 3 (Code Flow)

Week 2: Implementation
├─ Day 1-3: Phase 4 (Model Implementation)
└─ Day 4-5: Phase 5 (Advanced Topics - selective)

Week 3: Mastery
├─ Day 1-2: Phase 6 (Testing & Contributing)
├─ Day 3-4: Phase 7 (Platform-specific, as needed)
└─ Day 5: Build something (add model, optimize, etc.)
```

---

## Key Architectural Insights

### Multi-Stage Pipeline System

vLLM-Omni's core innovation is decomposing complex omni-modal models into stages:

**Example: Qwen2.5-Omni (3 stages)**
```
Stage 0 (Thinker):  Text/Image/Video → Understanding → Latent embeddings
       ↓ (OmniConnector - Shared Memory/ZMQ)
Stage 1 (Talker):   Latent → Speech reasoning → Audio tokens
       ↓ (OmniConnector)
Stage 2 (Code2Wav): Audio tokens → Waveform generation → Audio output
```

**Key benefits**:
- **Resource isolation**: Each stage on different GPU with custom memory allocation
- **Pipeline parallelism**: Stages process different requests concurrently
- **Flexibility**: Mix AR and DiT stages, different scheduling strategies
- **Scalability**: Disaggregate across machines with OmniConnector

### Request Lifecycle Detail

```python
# 1. User Request
{
  "model": "Qwen/Qwen2.5-Omni-7B",
  "messages": [{"role": "user", "content": "Hello"}],
  "sampling_params_list": [params_stage0, params_stage1, params_stage2]
}

# 2. AsyncOmni orchestrates
async_omni = AsyncOmni(model=model_name, stage_configs=stage_yaml)
output = await async_omni.generate(request)

# 3. Stage 0 processes
omni_stage_0.put_task(request)
→ Scheduler batches and schedules
→ Worker executes model forward pass
→ Outputs text + latent embeddings

# 4. OmniConnector transfers
shared_memory.send(embeddings, from_stage=0, to_stage=1)

# 5. Stage 1 receives and processes
stage_input_processor.thinker2talker(stage0_output)
→ Converts embeddings to stage 1 input format
→ Scheduler batches
→ Worker generates audio tokens

# 6. Repeat for stage 2, then return final output
```

### Critical Configuration Patterns

**Stage Config YAML Structure**:
```yaml
stage_args:
  - stage_id: 0
    runtime:
      process: true          # Separate process
      devices: "0"           # GPU 0
      max_batch_size: 1
    engine_args:
      model_stage: thinker   # Stage identifier
      worker_type: ar        # Autoregressive worker
      gpu_memory_utilization: 0.8
      engine_output_type: latent  # Output embeddings
    engine_input_source: []  # No upstream stage
    final_output: true       # Contributes to final response
```

---

## Common Pitfalls and Solutions

### Pitfall 1: GPU Memory Errors
**Symptom**: OOM (Out of Memory) during model loading or inference

**Solutions**:
- Adjust `gpu_memory_utilization` in stage config (try 0.7, 0.6)
- Use different GPUs for different stages (`runtime.devices`)
- Enable CPU offloading for diffusion models
- Check per-stage memory with `vllm_omni/worker/gpu_memory_utils.py`

### Pitfall 2: Stage Communication Failures
**Symptom**: Timeout or hanging between stages

**Solutions**:
- Check OmniConnector configuration in stage config
- Verify `engine_input_source` references correct upstream stage IDs
- Use `runtime.defaults.window_size: -1` for simpler debugging
- Check logs for handshake failures between stages

### Pitfall 3: Model Registry Not Finding Model
**Symptom**: `Model architecture not found` error

**Solutions**:
- Ensure model class is decorated with `@VLLM_OMNI_MODEL_REGISTRY.register_model()`
- Check `model_arch` in stage config matches registered name
- Import model module in `vllm_omni/model_executor/models/__init__.py`
- Verify model path in registry with `print(VLLM_OMNI_MODEL_REGISTRY._models)`

### Pitfall 4: Input Processor Not Applied
**Symptom**: Stage receives wrong input format

**Solutions**:
- Specify `custom_process_input_func` in stage config
- Path format: `vllm_omni.model_executor.stage_input_processors.<model>.<func>`
- Ensure function signature matches: `func(stage_output) -> new_input`
- Test processor independently before full pipeline

---

## Development Workflow Best Practices

### 1. Iterative Testing Strategy
```bash
# Start small
pytest -s -v tests/unit/test_specific_component.py

# Then integration
pytest -s -v tests/e2e/test_model_name.py -k "test_single_stage"

# Finally full pipeline
pytest -s -v tests/e2e/test_model_name.py --run-level=core_model
```

### 2. Debugging Multi-Stage Pipelines
```python
# Add logging in stage config
import logging
logging.basicConfig(level=logging.DEBUG)

# Check stage outputs
omni_stage.get_result()  # Inspect intermediate outputs

# Monitor GPU memory per stage
watch -n 1 nvidia-smi
```

### 3. Performance Profiling
```bash
# Profile specific stage
VLLM_OMNI_PROFILE=1 python your_script.py

# Generate flame graph
python -m vllm_omni.benchmarks.profile_model --model <model>

# Check stage timing
# Look for stage_latency metrics in logs
```

---

## Additional Resources

### Essential Documentation
- [Architecture Overview](docs/design/architecture_overview.md)
- [Stage Configs Reference](docs/configuration/stage_configs.md)
- [Contributing Guide](docs/contributing/README.md)
- [Test Guide](docs/contributing/ci/test_guide.md)

### Example Code
- [Examples Directory](examples/) - Offline and online serving examples
- [Benchmark Scripts](benchmarks/) - Performance testing
- [Test Suite](tests/) - Unit and integration tests

### Community
- [User Forum](https://discuss.vllm.ai)
- [Developer Slack](https://slack.vllm.ai) - #sig-omni channel
- [GitHub Issues](https://github.com/vllm-project/vllm-omni/issues)
- [Weekly Developer Meetings](https://tinyurl.com/vllm-omni-meeting) - Tuesdays 19:30 PDT

### Papers and Research
- [vLLM-Omni Paper](https://arxiv.org/abs/2602.02204)
- [vLLM Original Paper](https://arxiv.org/abs/2309.06180)
- Model papers: Qwen-Omni, BAGEL, GLM-Image, etc.

---

This tutorial takes you from complete beginner to confident contributor. Take your time, run examples, and don't hesitate to read code multiple times—this is a complex but well-architected system. Good luck! 🚀
