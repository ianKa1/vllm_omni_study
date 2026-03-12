# BAGEL End-to-End Run Analysis

## What Happened

### Process Topology

3 separate OS processes (not threads):

- **Main process** (pid 12839) — orchestrator, runs `Omni`, dispatches requests, collects outputs
- **Stage-0 process** — AR thinker, spawned via `multiprocessing.spawn`
  - Sub-process **EngineCore_DP0** — vLLM v1 engine core
  - Sub-process **Worker** — GPU worker doing actual forward passes
- **Stage-1 process** — DiT diffusion worker

They communicate via **ZMQ queues** (not shared memory for control signals).
Log confirms: `ZMQ transport detected; disabling SHM IPC`

---

## Connector Used

**`SharedMemoryConnector`** — `/dev/shm`

Two KV cache blobs transferred from Stage-0 → Stage-1:

| Payload                        | Size          |
|-------------------------------|---------------|
| Positive prompt KV cache       | **748,264 bytes** (~730 KB) |
| CFG negative companion KV cache | **117,490 bytes** (~115 KB) |

Mechanics: Stage-0 serializes KV tensors → writes to `/dev/shm/<uuid>` → sends a lightweight **metadata handle** (filename + size) over ZMQ → Stage-1 reads the handle, mmaps the shm file, deserializes tensors.

---

## CFG Flow (Classifier-Free Guidance)

The orchestrator **doubled** the request behind the scenes:

1. Original prompt: `"a cup of coffee on a wooden table, cinematic lighting"`
2. CFG companion: empty negative prompt (tagged `cfg_text`)

Stage-0 ran **both through the AR model** (batched as 2 requests, `max_batch_size: 2` in yaml). Both got KV caches transferred separately. Stage-1 received both, then used the CFG formula during denoising:

```
output = uncond + scale * (cond - uncond)
```

Log evidence:
```
kv_transfer_manager.py] KV transfer OK: ...710b68a9..., 748264 bytes       ← positive
kv_transfer_manager.py] KV transfer OK: ...710b68a9...__cfg_text, 117490 bytes  ← negative
pipeline_bagel.py] CFG enabled with multi-KV: using injected cfg_text KV Cache
```

---

## GPU Memory

| Stage | Model Weights | Notes |
|---|---|---|
| Stage-0 (AR, fp8) | **8.78 GiB** | KV cache pool: 2.72 GiB (50,896 tokens capacity) |
| Stage-1 (DiT, bf16) | **26.47 GiB** | 27.08 GiB process-scoped total |

Total GPU usage: ~**35+ GiB** on a single L40S (46 GiB). Both stages share the same GPU (`cuda:0`).

The **2.72 GiB KV cache headroom** is the *capacity* of Stage-0's paged KV cache pool — pre-allocated GPU memory vLLM uses as a ring buffer for active requests during AR decoding. It is sized for worst-case concurrency (50,896 tokens). Only the leftover VRAM after model weights becomes KV cache.

---

## Timeline

| Time | Event |
|---|---|
| 22:11:41 | Orchestrator starts, creates SharedMemoryConnector |
| 22:11:49 | Stage-0 and Stage-1 processes spawn (in parallel) |
| 22:12:10 | Stage-0 starts loading weights |
| 22:12:00 | Stage-1 starts loading weights (parallel with Stage-0) |
| 22:15:22 | Stage-0 weights done (190s) |
| 22:15:24 | Stage-1 weights done (202s) |
| 22:15:28 | Both stages report ready; request dispatched |
| 22:15:28 | Stage-0 runs AR forward pass, transfers KV caches |
| 22:15:28 | Stage-1 receives KV, starts 30-step denoising |
| 22:16:01 | Denoising done (33s) → image saved |

Stage-0 and Stage-1 load weights **in parallel**, cutting init time nearly in half.

---

## Inter-Stage Communication

### Control Channel: ZMQ (TCP loopback)

ZMQ sockets carry lightweight control messages — serialized Python objects:

- **Orchestrator → Stage-0**: `OmniRequest` object (prompt token IDs, sampling params, modality flags)
- **Stage-0 → Stage-1**: KV cache **metadata handle** — a small dict `{"shm": {"name": "...", "size": 748264}}`
- **Stage-1 → Orchestrator**: result metadata (image dimensions, timing)
- **Shutdown signals**

### Data Channel: `/dev/shm` (shared memory, zero-copy)

Large payloads bypass ZMQ entirely. Stage-0 writes KV tensors to `/dev/shm`, sends the filename over ZMQ, Stage-1 mmaps the same file.

```
Orchestrator ──ZMQ──► Stage-0 ──ZMQ handle──► Stage-1
                          │                      │
                          └──/dev/shm tensors────┘
                               (748 KB + 115 KB)
```

---

## Where KV Cache Lives at Each Stage

### Stage-0: GPU → CPU → `/dev/shm`

The KV cache lives in vLLM's **paged KV cache pool on GPU VRAM** (the 2.72 GiB pool), organized as fixed-size blocks. At the end of AR generation, `_extract_kv_cache()` gathers only the blocks used by this request, slices to `seq_len`, then copies GPU → CPU:

```python
# kv_transfer_manager.py
key_cache[layer_idx] = flat_k.detach().cpu().contiguous()   # GPU → CPU copy
value_cache[layer_idx] = flat_v.detach().cpu().contiguous()
```

The 28 key + 28 value CPU tensors are then serialized via **msgpack** into raw bytes and written to `/dev/shm`.

### Stage-1: `/dev/shm` → CPU → `req.past_key_values`

Stage-1 reads the bytes, deserializes back into CPU `torch.Tensor` objects:

```python
# serialization.py
arr = torch.frombuffer(buffer, dtype=torch.uint8)
return arr.view(torch_dtype).reshape(shape)   # stays on CPU after deserialization
```

Then attaches to the request:

```python
# kv_transfer_manager.py
req.past_key_values = SimpleNamespace(key_cache=[...], value_cache=[...])
req.sampling_params.past_key_values = kv_obj   # for BagelPipeline compatibility
```

The DiT cross-attention uses `past_key_values` during each of the 30 denoising steps, moving to GPU on-the-fly as needed.

### Full Data Path

```
Stage-0 GPU VRAM (paged pool, 2.72 GiB capacity)
    │ gather blocks + slice to seq_len
    │ .detach().cpu().contiguous()
    ▼
CPU RAM (28 key + 28 value tensors, ~730 KB)
    │ msgpack serialize → bytes
    ▼
/dev/shm (raw bytes, fcntl-locked)
    │ ZMQ handle (filename + size)
    ▼
Stage-1 reads /dev/shm → torch.frombuffer → CPU tensors
    │ attach to req.past_key_values
    ▼
DiT cross-attention (GPU, on each denoising step)
```

> **Note**: The deserialized tensors land on CPU first, not directly on GPU. There is a `TODO: Enable zero-copy support` comment in `serialization.py` — meaning the current implementation has an extra CPU roundtrip that could be eliminated with RDMA or CUDA IPC.

---

## Available Connectors

Four connectors are registered in `OmniConnectorFactory`:

| Name | Transport | Use Case |
|---|---|---|
| `SharedMemoryConnector` | `/dev/shm` + fcntl lock | Default; single-node, same machine |
| `MooncakeStoreConnector` | Mooncake distributed KV store | Multi-node; RDMA or TCP |
| `MooncakeTransferEngineConnector` | Mooncake transfer engine | Multi-node; high-throughput RDMA |
| `YuanrongConnector` | YuanRong transport | Multi-node; alternative RDMA |

---

## Other Connectors

### Why SharedMemoryConnector Was Auto-Selected

**It has nothing to do with GPU count.** The selection logic is purely YAML-driven.

Your `bagel_l40s.yaml` has no `input_connectors` or `output_connectors` section → the code sees an edge (`0 → 1`) with no explicit connector → falls back to `SharedMemoryConnector`:

```python
# initialization.py
if edge_key not in connectors:
    logger.info(f"Auto-configuring SharedMemoryConnector for edge {edge_key}")
    connectors[edge_key] = ConnectorSpec(name="SharedMemoryConnector", ...)
```

### How to Enable a Different Connector

Two ways in the YAML:

**Option 1: Inline per-stage**

```yaml
stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1:
        name: MooncakeStoreConnector
        extra:
          endpoint: "http://mooncake-host:8080"

  - stage_id: 1
    input_connectors:
      from_stage_0:
        name: MooncakeStoreConnector
        extra:
          endpoint: "http://mooncake-host:8080"
```

**Option 2: Global reference (cleaner, avoids repetition)**

```yaml
runtime:
  connectors:
    my_mooncake:
      name: MooncakeStoreConnector
      extra:
        endpoint: "http://mooncake-host:8080"

stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: my_mooncake   # reference by name

  - stage_id: 1
    input_connectors:
      from_stage_0: my_mooncake
```

Both sides of an edge **must declare the same connector name** or a `ValueError` is raised at startup.

### When to Use Each Connector

| Connector | When |
|---|---|
| `SharedMemoryConnector` | Stages on the **same machine** (default, optimal for single-node) |
| `MooncakeStoreConnector` | Stages on **different machines**, TCP-based, simpler setup |
| `MooncakeTransferEngineConnector` | Different machines, **RDMA**, high-throughput production |
| `YuanrongConnector` | ByteDance internal RDMA transport |

On a single L40S node, `SharedMemoryConnector` is the **optimal** choice — the others would add network overhead for no benefit. RDMA connectors are only meaningful when stages are disaggregated across separate physical hosts.

---

## How Mooncake Communicates Between Two Machines

### MooncakeStoreConnector — It DOES use the network (TCP or RDMA)

This one uses a **central Mooncake server** as intermediary. Both machines connect to it over TCP (or RDMA):

```
Machine A (Stage-0)     Mooncake Server     Machine B (Stage-1)
       │                      │                     │
       │── TCP: put(key) ────►│ holds bytes in RAM   │
       │                      │                     │
       │── ZMQ: send key ────────────────────────► │
       │                      │                     │
       │                      │◄── TCP: get(key) ───│
       │                      │─── bytes ──────────►│
```

The network here is **regular TCP** (or optionally RDMA through the Mooncake server). It's the simplest multi-node option.

---

### MooncakeTransferEngineConnector — RDMA IS a network, just a different one

The "no network" description was misleading — RDMA **is** a network protocol, but it bypasses the OS kernel and CPU. It uses a dedicated **InfiniBand** or **RoCE (RDMA over Converged Ethernet)** NIC that machines in a datacenter cluster share.

Here's what actually happens:

```
Machine A (Stage-0)                           Machine B (Stage-1)
     │                                              │
     │  1. Stage-1 pre-allocates CPU buffer         │
     │     registers it with RDMA NIC ◄─────────── │
     │                                              │
     │  2. ZMQ handshake (TCP):                     │
     │     A sends {hostname, RDMA port,  ─────────►│
     │              dst_addr, length}               │
     │                                              │
     │  3. RDMA NIC on Machine A reads              │
     │     KV bytes from CPU memory                 │
     │     writes DIRECTLY into Machine B's ───────►│ bytes appear in
     │     pre-registered buffer                    │ Stage-1's RAM
     │     (no CPU on either side involved)         │
     │                                              │
     │  4. ZMQ: TRANS_DONE signal ────────────────►│
```

Key insight: **RDMA is still a physical network** (cables between machines), but:
- No OS kernel involvement — NIC handles everything in hardware
- No CPU copy — the NIC DMA-writes directly into the destination process's memory
- No TCP/IP stack overhead — uses its own protocol (IB or RoCE)
- Latency ~1–5 µs vs ~100+ µs for TCP

The ZMQ handshake (step 2) still uses regular TCP — just to exchange the memory address. The actual bulk data moves via RDMA.

### Summary

| | MooncakeStore | MooncakeTransferEngine |
|---|---|---|
| Uses network? | Yes — TCP to central server | Yes — RDMA NIC between machines |
| Central server | Yes | No (peer-to-peer) |
| CPU on data path | Yes (serialize + send) | No (NIC handles it) |
| Special hardware | No | InfiniBand or RoCE NIC required |
| Latency | ~ms | ~µs |

---

## Transfer Performance vs Inference Time

From the actual BAGEL run, all timings are in the log:

| Step | Time |
|---|---|
| Stage-0 AR forward pass (13 tokens, fp8) | **< 1 second** (same timestamp 22:15:28) |
| KV transfer via SharedMemoryConnector (~845 KB) | **< 1 second** (same timestamp 22:15:28) |
| Stage-1 DiT denoising (30 steps, 640px) | **33 seconds** |

The transfer is so fast it doesn't even show a separate timestamp — it completed within the same second as the AR forward pass.

### Why Transfer is Negligible Here

The KV cache was only **845 KB total** (13-token prompt × 28 layers × K+V):

```
SharedMemoryConnector write ≈ RAM bandwidth (~50 GB/s)
845 KB ÷ 50 GB/s ≈ 0.017ms
+ msgpack serialization   ≈ 1–5ms
Total                     ≈ 5ms
```

vs DiT denoising: **33,000ms** → transfer is ~**0.015%** of total time.

### When Transfer Starts to Matter

| Scenario | KV Cache Size | Transfer Time (SHM) | Proportion of 33s |
|---|---|---|---|
| Short prompt (13 tokens) | ~845 KB | ~5ms | 0.015% |
| Long prompt (1000 tokens) | ~65 MB | ~100ms | 0.3% |
| Image input (4900 tokens) | ~320 MB | ~500ms | 1.5% |
| Multi-node TCP | ~65 MB | ~500ms+ | ~1.5% |
| Multi-node RDMA | ~65 MB | ~5ms | ~0.015% |

Transfer only becomes meaningful at **multi-node scale with large image inputs**. That's exactly the scenario RDMA connectors are designed for — when a vision encoder produces thousands of tokens that need to cross machine boundaries.

### Bottom Line

In the BAGEL pipeline, **DiT denoising is the overwhelming bottleneck** and will remain so regardless of connector choice. The connector matters more for latency-sensitive pipelines with very long context (image/video understanding) disaggregated across nodes.
