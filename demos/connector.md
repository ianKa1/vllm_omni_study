# vLLM-Omni Stages & Connectors — A Complete Tutorial

---

## 1. The Problem Being Solved

Modern AI applications mix multiple model types. Generating an image from text might require:
1. A large language model (AR) to encode the prompt into rich embeddings
2. A diffusion transformer (DiT) to denoise and render pixels

These two networks have completely different execution patterns — AR generates tokens one-by-one; DiT runs a fixed number of denoising steps in parallel. Putting them in the same process and the same engine is awkward, limits GPU utilization, and makes it impossible to assign them to different hardware.

vLLM-Omni solves this with a **stage-based pipeline**: each model component is a stage, isolated in its own process, communicating through structured channels.

---

## 2. What Is a Stage?

A **stage** is a self-contained inference unit with:
- Its own OS process (spawned by `multiprocessing`)
- Its own GPU assignment
- Its own engine (`OmniLLM` for AR, `OmniDiffusion` for DiT)
- A pair of ZMQ queues for receiving tasks and returning results

**Every stage runs the same core loop** (`_stage_worker` in `omni_stage.py:694`):

```
while True:
    task = in_queue.get(timeout)     # wait for work
    result = engine.generate(task)   # run model
    out_queue.put(result)            # send result back
```

The **orchestrator** (`Omni` in `omni.py`) sits above all stages. It receives user requests, submits them to stage 0, then routes each stage's output to the next stage, and finally collects the final output and returns it to the caller.

```
User code
   │  omni_lm.generate(inputs, sampling_params_list)
   ▼
Orchestrator (main process)
   ├── ZMQ → Stage 0 process (GPU 0)
   │              │ result
   ├── ZMQ → Stage 1 process (GPU 1)
   │              │ result
   └── ZMQ → Stage 2 process (GPU 1)
                  │ final result
   ◄──────────────┘
```

---

## 3. Stage Config Deep Dive

Every stage is described by a YAML block. Here is the full anatomy, using the Qwen3-Omni thinker stage as a reference:

```yaml
- stage_id: 0                          # unique integer ID, used for routing
  stage_type: llm                      # "llm" (AR) or "diffusion" (DiT)

  # --- process/hardware config ---
  runtime:
    devices: "0"                       # which GPU(s) this stage owns
    max_batch_size: 64                 # max requests batched together
    process: true                      # run in a separate OS process

  # --- engine config (passed to OmniLLM or OmniDiffusion) ---
  engine_args:
    model_stage: thinker               # which sub-model to load from the HF checkpoint
    model_arch: Qwen3OmniMoeForConditionalGeneration  # registered class in registry.py
    worker_type: ar                    # "ar" = autoregressive, "generation" = diffusion
    scheduler_cls: ...OmniARScheduler  # which scheduler to use
    gpu_memory_utilization: 0.9        # fraction of GPU VRAM to reserve
    enforce_eager: false               # disable CUDA graph capture (slower but safer)
    engine_output_type: latent         # what this stage emits: text/latent/token_ids/audio/image
    tensor_parallel_size: 1            # TP degree
    distributed_executor_backend: "mp" # "mp" (multiprocess) or "ray"
    quantization: "fp8"                # optional: fp8/awq/gptq/bitsandbytes

  # --- pipeline wiring ---
  engine_input_source: [0]             # list of stage_ids that feed into this stage
  custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker
                                       # transforms upstream output into this stage's input format

  # --- output config ---
  final_output: true                   # is this a terminal stage for some output modality?
  final_output_type: text              # "text" / "audio" / "image"
  is_comprehension: true               # true = multimodal understanding; false = generation only

  # --- defaults (overridden by caller's SamplingParams) ---
  default_sampling_params:
    temperature: 0.9
    max_tokens: 2048
```

Key field interactions:
- `engine_input_source` defines **edges** in the DAG — stage 1 will only run after all its sources (e.g., `[0]`) produce output
- `custom_process_input_func` is the **adapter function** that reshapes stage N's output tensor/tokens into stage N+1's expected input format
- `engine_output_type` tells the orchestrator what format the output is in so it knows how to route it
- `final_output: true` marks this stage as a terminal node — the orchestrator collects its output and returns it to the user

---

## 4. The Two Stage Types

### 4a. `stage_type: llm` — Autoregressive Stages

Uses `OmniLLM` (wrapping vLLM's `LLMEngine`). Runs the standard AR generation loop: processes a prompt token-by-token, applying KV cache reuse, batching, and continuous batching.

Worker class is determined by `worker_type`:
- `worker_type: ar` → `GPUARWorker` — the standard vLLM GPU worker
- `worker_type: generation` → `GPUGenerationWorker` — for non-standard AR that produces non-text (e.g., codec tokens for audio)

### 4b. `stage_type: diffusion` — DiT Stages

Uses `OmniDiffusion`. Runs N denoising steps (configured by `num_inference_steps`), not token-by-token. Consumes the AR stage's output (KV cache, hidden states, or token IDs) as conditioning. Outputs images, video frames, or audio waveforms.

The diffusion stage is also where **Classifier-Free Guidance (CFG)** is handled — it receives two sets of KV caches (positive and negative prompt) via `cfg_kv_collect_func` and applies CFG during denoising.

---

## 5. Three Real Pipeline Examples

### Example A: GLM-Image (AR → DiT, 2 stages)

```
Text prompt
    │
    ▼
Stage 0: AR (GlmImageForConditionalGeneration)       GPU 0
    engine_output_type: token_ids
    Produces: 1281 "prior tokens" (visual vocabulary)
    │
    │  custom_process_input_func: ar2diffusion()
    ▼
Stage 1: DiT (GlmImagePipeline)                      GPU 1
    engine_output_type: image
    Runs: 50 denoising steps conditioned on prior tokens
    Produces: 1024×1024 RGB image
```

The AR here is not generating text — it's generating *visual prior tokens* from a visual vocabulary of size 16512. The DiT uses those tokens as its conditioning signal.

### Example B: BAGEL (AR + CFG → DiT, 2 stages)

```
Text prompt  →  prompt_expand_func (expand_cfg_prompts)
    │
    ├── positive prompt ──┐
    └── negative prompt ──┤  (both submitted to Stage 0 in parallel)
                          ▼
Stage 0: AR (BagelForConditionalGeneration)          GPU 0
    omni_kv_config.need_send_cache: true
    Produces: KV caches for positive AND negative prompts
    │
    │  cfg_kv_collect_func (collect_cfg_kv_caches)
    │  — pairs up the two KV caches
    ▼
Stage 1: DiT (BAGEL diffusion)                       GPU 0
    omni_kv_config.need_recv_cache: true
    Runs CFG: output = pos_kv + guidance_scale * (pos_kv - neg_kv)
    Produces: image
```

Notice both stages share GPU 0 here — `gpu_memory_utilization: 0.35 + 0.55 = 0.9`.

### Example C: Qwen3-Omni (AR → AR → DiT, 3 stages)

```
Audio/video/text input
    │
    ▼
Stage 0: Thinker (Qwen3OmniMoe AR)                   GPU 0
    engine_output_type: latent   ← hidden states, NOT text tokens
    final_output_type: text      ← ALSO emits decoded text to user
    │
    │  thinker2talker()          ← extract last hidden state + text token IDs
    ▼
Stage 1: Talker (Qwen3OmniMoe AR, talker sub-model)  GPU 1
    worker_type: ar
    engine_output_type: latent   ← RVQ codec token sequences (8 codebooks)
    detokenize: false
    stop_token_ids: [2150]       ← CODEC_EOS
    │
    │  talker2code2wav()         ← reshape codec tokens into frame batches
    ▼
Stage 2: Code2Wav (Qwen3OmniMoe generation)          GPU 1
    worker_type: generation
    engine_output_type: audio    ← 24kHz waveform
    final_output_type: audio     ← returned to user
```

Stage 0 is special: it has **two** output paths simultaneously:
- Text tokens → decoded and returned to user immediately (`final_output: true, final_output_type: text`)
- Hidden states → forwarded to stage 1 for audio synthesis

This is why in the demo code both text and audio can be requested from the same generate call.

---

## 6. Stage Lifecycle in Detail

When you call `Omni(model=...)`, here is what happens step by step:

**Step 1 — Load stage configs**

`Omni.__init__` reads the YAML (from model's HF repo, or `stage_configs_path`), resolves all stage definitions.

**Step 2 — Create OmniStage objects**

One `OmniStage` object per stage. These are just config holders at this point — no processes yet.

**Step 3 — Attach ZMQ queues**

For each stage, two ZMQ sockets are created: `in_q` (PULL) and `out_q` (PUSH). The orchestrator holds the other end.

**Step 4 — Init connectors**

`initialize_orchestrator_connectors()` reads edge definitions from the YAML, creates `OmniConnectorBase` instances for each directed edge (e.g., `(0, 1)`, `(1, 2)`). If no connector is explicitly configured, `SharedMemoryConnector` is auto-created for every `engine_input_source` edge.

**Step 5 — Spawn stage processes**

`stage.init_stage_worker()` calls `mp.Process(target=_stage_worker, ...)`. The worker process:
- Sets `CUDA_VISIBLE_DEVICES` to `runtime.devices`
- Acquires a sequential init lock (so two stages don't race to initialize on the same GPU)
- Constructs `OmniLLM` or `OmniDiffusion`
- Builds its own connector instances (receiver side)
- Sends `{"type": "stage_ready"}` back to orchestrator via `out_q`

**Step 6 — Orchestrator waits for all `stage_ready` signals**

Only then does `Omni.__init__` return control to user code.

---

## 7. The Request Flow During `generate()`

```
omni_lm.generate([input], sampling_params_list)
```

1. **Prompt expansion** (if `prompt_expand_func` configured): expand one input into positive+negative pair for CFG
2. **Submit to Stage 0**: orchestrator serializes input + sampling_params[0], puts it on Stage 0's `in_q`
3. **Stage 0 generates**: runs AR loop, produces output (`text`, `latent`, or `token_ids`)
4. **Orchestrator receives Stage 0 result**: calls `custom_process_input_func` to transform it for Stage 1
5. **Transfer via connector** (`try_send_via_connector`):
   - Serializes payload → writes to `/dev/shm/<req_id>` (SharedMemoryConnector)
   - Sends lightweight notification to Stage 1's `in_q`: `{"from_connector": True, "connector_metadata": {"shm": {...}}}`
6. **Stage 1 receives notification**: calls `try_recv_via_connector` → reads from `/dev/shm/<req_id>`
7. Repeat for Stage 2
8. **Final stage** (`final_output: true`) sends result back through `out_q` to orchestrator
9. Orchestrator yields `OmniStageOutput` to caller

---

## 8. The Connector in Depth

### Why not just use the ZMQ queue for everything?

The ZMQ queue is lightweight — it's designed for small control messages. Passing a full KV cache (hundreds of MB) or a batch of hidden states through a queue would be:
- **Slow**: serialized, copied through the kernel, deserialized
- **Memory-wasteful**: double-copies the data

The connector solves this by using **shared memory** (or RDMA for cross-node) for the bulk payload, while only the handle (a small dict with a shm name and size) travels through the ZMQ queue.

### The two-phase handshake

```
Stage N (sender)                       Stage N+1 (receiver)
     │                                        │
     │  1. connector.put(payload)             │
     │     → serialize payload to /dev/shm    │
     │     → returns metadata={shm:{name,size}}
     │                                        │
     │  2. out_q.put({                        │
     │       "from_connector": True,          │
     │       "connector_metadata": metadata   │
     │     })                                 │
     │                 ─────────────────────► │
     │                                        │  3. in_q.get() → sees from_connector=True
     │                                        │     connector.get(metadata=shm_handle)
     │                                        │     → reads from /dev/shm, deserializes
     │                                        │     → gets payload back
     │                                        │
     │                                        │  4. engine.generate(payload)
```

Phase 1+2 happen in the orchestrator (main process).
Phase 3+4 happen inside the stage worker process.

### Connector backends

| Backend | Transport | When to use |
|---|---|---|
| `SharedMemoryConnector` | POSIX `/dev/shm` | Default. Same node, any GPU. |
| `MooncakeStoreConnector` | TCP/RDMA via Mooncake | Cross-node clusters. |
| `MooncakeTransferEngineConnector` | RDMA with role-based init | High-throughput cross-node. |
| `YuanrongConnector` | Alibaba internal RDMA | Internal infra. |

### Configuring a custom connector

In the stage YAML, declare connectors in the top-level `runtime.connectors` block, then reference by name in each stage's `input_connectors` / `output_connectors`:

```yaml
runtime:
  connectors:
    my_shm:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 65536

    my_mooncake:
      name: MooncakeStoreConnector
      extra:
        host: "127.0.0.1"
        metadata_server: "http://10.0.0.1:8080/metadata"
        master: "10.0.0.1:50051"

stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: my_mooncake   # reference by name

  - stage_id: 1
    input_connectors:
      from_stage_0: my_mooncake  # must match
    output_connectors:
      to_stage_2: my_shm

  - stage_id: 2
    input_connectors:
      from_stage_1: my_shm
```

If you omit `input_connectors`/`output_connectors` entirely (as in the simple `qwen3_omni_moe.yaml`), `SharedMemoryConnector` is **auto-configured** for every `engine_input_source` edge — no explicit declaration needed for same-node setups.

---

## 9. Summary

```
Stage Config (YAML)
    └── stage_id, stage_type, runtime.devices, engine_args.*
              │
              ▼
    OmniStage (Python object)
    ├── stage_type = "llm"       → OmniLLM  (wraps vLLM LLMEngine)
    └── stage_type = "diffusion" → OmniDiffusion (DiT denoising loop)
              │
              │  runs in its own OS process
              │  owns its own GPU via CUDA_VISIBLE_DEVICES
              │
    _stage_worker(in_q, out_q)
              │
              │  receives tasks via ZMQ in_q
              │  returns results via ZMQ out_q
              │
    OmniConnector (per directed edge)
    ├── put(payload) → write to /dev/shm  }  bulk data
    └── get(handle)  → read from /dev/shm }  via shared memory

    Orchestrator (Omni, main process)
    ├── routes requests stage-by-stage following engine_input_source DAG
    ├── calls custom_process_input_func between stages
    ├── calls try_send_via_connector to transfer inter-stage data
    └── yields final outputs to user code
```
