# PrisKV Connector for vLLM-Omni — Implementation Plan

## 1. Background

### What is PrisKV?

PrisKV is a distributed key-value store developed by ByteDance's PrisDB team, first surfaced publicly
through the AIBrix project (`vllm-project/aibrix`). It is one of several L2 cache backends in
AIBrix's KVCache Offloading Framework alongside InfiniStore, HPKV, RocksDB, and EIC.

Key properties (from source analysis of `aibrix_kvcache/l2/connectors/priskv/priskv.py`):

| Property | Detail |
|---|---|
| Protocol | Redis-compatible wire protocol (default port 6379) |
| Transport | TCP (default) + RDMA (GPUDirect RDMA via SGL) |
| Authentication | Password-based |
| Batch ops | `mput` / `mget` (feature-gated via `USE_MPUT_MGET`) |
| Memory model | Registered memory regions (MRs) for zero-copy RDMA |
| Key encoding | Hex-encoded bytes + configurable suffix |
| Status model | `Status` return objects wrapping results or error codes |
| Client | `PriskvClient` (open-source preparation stage; install via `aibrix_kvcache`) |

### Why add PrisKV to vLLM-Omni?

The current connector options in vLLM-Omni are:

| Connector | Limit |
|---|---|
| `SharedMemoryConnector` | Same-node only (POSIX `/dev/shm`) |
| `MooncakeStoreConnector` | Requires Mooncake infra deployment |
| `MooncakeTransferEngineConnector` | Requires RDMA + Mooncake |
| `YuanrongConnector` | Alibaba-internal only |

PrisKV fills the gap for teams that:
- Run multi-node vLLM-Omni deployments (e.g., stage 0 on node A, stage 1 on node B)
- Already operate a PrisDB/PrisKV cluster
- Want RDMA-accelerated inter-stage transfer without the Mooncake operational overhead

For large inter-stage payloads — hidden states for a 30B model can be 200–800 MB per request —
RDMA transfer via PrisKV would deliver significantly lower latency than TCP-serialized alternatives.

---

## 2. Architecture Overview

### How vLLM-Omni connectors work (recap)

```
Orchestrator (main process)
    │
    │  try_send_via_connector(connector, stage_id, next_stage_id, req_id, payload)
    │
    │  1. connector.put(from_stage, to_stage, req_id, payload_dict)
    │     → returns (success, size, metadata)
    │
    │  2. ZMQ queue.put({"from_connector": True, "connector_metadata": metadata})
    │
Stage N+1 worker process
    │  task = in_q.get()
    │  try_recv_via_connector(task, connectors, stage_id)
    │
    │  3. connector.get(from_stage, to_stage, req_id, metadata)
    │     → returns (payload_dict, size)
```

The connector's only job: **store and retrieve arbitrary serialized Python objects, keyed by
`(from_stage, to_stage, request_id)`**. The metadata dict returned by `put()` is sent via the ZMQ
queue (lightweight) and passed back to `get()` to locate the data.

### Mapping PrisKV AIBrix concepts → vLLM-Omni

| AIBrix PrisKV role | vLLM-Omni equivalent |
|---|---|
| L2 cache backend for KV attention tensors | Inter-stage payload store (hidden states, token IDs, etc.) |
| `PriskvClient.put(key, mr)` | Store serialized `payload_dict` bytes under a unique key |
| `PriskvClient.get(key, mr)` | Retrieve bytes and deserialize back to `payload_dict` |
| SGL / memory regions for RDMA | CPU byte buffers (Phase 1), then GPU tensors via RDMA (Phase 2) |
| `mput` / `mget` | Batch transfer for high-throughput multi-request scenarios |
| Key suffix scheme | `{req_id}@{from_stage}_{to_stage}` (matches `OmniConnectorBase._make_key`) |

---

## 3. File Structure

```
vllm_omni/distributed/omni_connectors/
├── connectors/
│   ├── base.py                      # OmniConnectorBase (existing)
│   ├── shm_connector.py             # (existing)
│   ├── mooncake_store_connector.py  # (existing)
│   ├── priskv_connector.py          # ← NEW: PrisKV connector
│   └── __init__.py                  # add PrisKVConnector export
├── factory.py                       # ← MODIFY: register PrisKVConnector
└── utils/
    └── config.py                    # no changes needed
```

---

## 4. Implementation Plan

### Phase 1 — TCP path (basic, no RDMA)

Goal: functional PrisKV connector using TCP transport and `OmniSerializer` for payload encoding.
This mirrors `SharedMemoryConnector` but stores bytes in a remote PrisKV server instead of
`/dev/shm`.

#### 4.1 `PrisKVConnector` class skeleton

```python
# vllm_omni/distributed/omni_connectors/connectors/priskv_connector.py

from dataclasses import dataclass
from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


@dataclass
class PrisKVConfig:
    remote_addr: str = "127.0.0.1"
    remote_port: int = 6379
    password: str = ""
    use_mput_mget: bool = False
    connect_timeout_s: float = 5.0
    key_suffix: str = "vllm_omni"


class PrisKVConnector(OmniConnectorBase):
    """
    OmniConnector backed by PrisKV (ByteDance PrisDB).

    Supports same-node and cross-node inter-stage transfer.
    Phase 1: TCP path with OmniSerializer serialization.
    Phase 2 (future): RDMA zero-copy via registered memory regions.
    """

    supports_raw_data: bool = False  # Phase 1; True in Phase 2 with RDMA

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = PrisKVConfig(
            remote_addr=config.get("remote_addr", "127.0.0.1"),
            remote_port=int(config.get("remote_port", 6379)),
            password=config.get("password", ""),
            use_mput_mget=bool(config.get("use_mput_mget", False)),
            connect_timeout_s=float(config.get("connect_timeout_s", 5.0)),
            key_suffix=config.get("key_suffix", "vllm_omni"),
        )
        self._client = None   # PriskvClient, initialized in _ensure_connected()
        self._metrics = {"puts": 0, "gets": 0, "bytes_transferred": 0, "errors": 0}

    # ------------------------------------------------------------------ #
    # OmniConnectorBase interface                                         #
    # ------------------------------------------------------------------ #

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            self._ensure_connected()
            key = self._make_key(put_key, from_stage, to_stage)
            payload: bytes = self.serialize_obj(data)
            size = len(payload)
            self._client.put(key, payload)             # TCP put
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            metadata = {"priskv_key": key, "size": size}
            return True, size, metadata
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("PrisKVConnector put failed for req %s: %s", put_key, e)
            return False, 0, None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict | None = None,
    ) -> tuple[Any, int] | None:
        try:
            self._ensure_connected()
            # Prefer key from metadata (sent via ZMQ); fall back to recomputing it
            if metadata and "priskv_key" in metadata:
                key = metadata["priskv_key"]
                size = int(metadata.get("size", 0))
            else:
                key = self._make_key(get_key, from_stage, to_stage)
                size = 0

            payload: bytes = self._client.get(key)
            if payload is None:
                logger.error("PrisKVConnector: key %s not found", key)
                return None
            if size == 0:
                size = len(payload)
            obj = self.deserialize_obj(payload)
            self._client.delete(key)   # consume-once semantics (mirrors SHM)
            self._metrics["gets"] += 1
            self._metrics["bytes_transferred"] += size
            return obj, size
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("PrisKVConnector get failed for req %s: %s", get_key, e)
            return None

    def cleanup(self, request_id: str) -> None:
        # Best-effort deletion if receiver never consumed
        if self._client:
            try:
                self._client.delete(request_id)
            except Exception:
                pass

    def health(self) -> dict[str, Any]:
        connected = self._client is not None
        return {"status": "healthy" if connected else "disconnected", **self._metrics}

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_connected(self) -> None:
        if self._client is not None:
            return
        try:
            from priskv import PriskvClient  # installed via aibrix_kvcache package
        except ImportError as e:
            raise ImportError(
                "PrisKV client not installed. "
                "Install via: pip install aibrix_kvcache[priskv]"
            ) from e
        self._client = PriskvClient(
            host=self._cfg.remote_addr,
            port=self._cfg.remote_port,
            password=self._cfg.password,
        )
        self._client.open()
        logger.info(
            "PrisKVConnector connected to %s:%d",
            self._cfg.remote_addr,
            self._cfg.remote_port,
        )
```

#### 4.2 Register in factory

```python
# vllm_omni/distributed/omni_connectors/factory.py  (add)

def _create_priskv_connector(config: dict[str, Any]) -> OmniConnectorBase:
    from .connectors.priskv_connector import PrisKVConnector
    return PrisKVConnector(config)

OmniConnectorFactory.register_connector("PrisKVConnector", _create_priskv_connector)
```

#### 4.3 YAML usage

```yaml
# Single-node (same server, different GPUs)
runtime:
  connectors:
    priskv_local:
      name: PrisKVConnector
      extra:
        remote_addr: "127.0.0.1"
        remote_port: 6379

# Cross-node (stage 0 on node-0, stage 1 on node-1)
runtime:
  connectors:
    priskv_remote:
      name: PrisKVConnector
      extra:
        remote_addr: "10.0.1.50"   # PrisKV server address
        remote_port: 6379
        password: "secret"

stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: priskv_remote

  - stage_id: 1
    engine_input_source: [0]
    input_connectors:
      from_stage_0: priskv_remote
```

---

### Phase 2 — RDMA path (zero-copy for large tensors)

Goal: use PrisKV's RDMA + SGL (Scatter-Gather List) capability to transfer large tensors
(hidden states, KV caches) without CPU serialization overhead. Relevant for payloads > ~50 MB.

#### How AIBrix PrisKV RDMA works

From the source analysis:
- `register_slabs(tensors)`: registers torch tensors as RDMA Memory Regions (MRs) with the PrisKV server; fills `_register_cache` mapping `tensor.data_ptr()` → registration descriptor
- `_get(key, mr)`: issues an RDMA `READ` into the pre-registered memory region (zero-copy)
- `_put(key, mr)`: issues an RDMA `WRITE` from the pre-registered memory region (zero-copy)
- SGL = Scatter-Gather List, a descriptor pointing to registered buffer address + length

#### Design for Phase 2

```
Stage N output (torch.Tensor hidden states)
    │
    │  1. Identify tensors in payload dict
    │  2. For each tensor:
    │     a. Pin memory (tensor.pin_memory()) if not already pinned
    │     b. PriskvClient.register_slabs([tensor])  → MR descriptor
    │     c. PriskvClient._put(key, sgl)  → RDMA WRITE (zero-copy)
    │  3. Metadata: {"priskv_key": key, "size": ..., "rdma": True, "shape": ..., "dtype": ...}
    │     → sent via ZMQ queue (tiny)
    │
Stage N+1 worker
    │  1. Receives metadata from ZMQ queue
    │  2. Allocates output tensor (same shape/dtype)
    │  3. PriskvClient.register_slabs([output_tensor])
    │  4. PriskvClient._get(key, sgl)  → RDMA READ into output_tensor (zero-copy)
    │  5. Reconstruct payload dict with output_tensor
```

New connector flag for Phase 2:
```python
supports_raw_data: bool = True   # can handle raw torch.Tensor natively
```

New method additions:
```python
def _put_rdma(self, key: str, tensor: torch.Tensor) -> dict:
    """Register tensor MR and issue RDMA WRITE."""
    ...

def _get_rdma(self, key: str, metadata: dict) -> torch.Tensor:
    """Allocate output tensor, register MR, issue RDMA READ."""
    ...
```

Selection logic in `put()`:
```python
# Threshold-based: use RDMA for tensors above RDMA_THRESHOLD_BYTES
if self._rdma_available and tensor_bytes > self._rdma_threshold:
    return self._put_rdma(key, tensor)
else:
    return self._put_tcp(key, payload_bytes)
```

#### RDMA prerequisites

- `IPC_LOCK` capability in container (`securityContext.capabilities.add: [IPC_LOCK]`)
- RDMA-capable NIC (Mellanox/NVIDIA ConnectX) on both nodes
- PrisKV server configured for RDMA (same as AIBrix InfiniStore setup)
- Environment variable: `VLLM_OMNI_PRISKV_RDMA_DEVICE=mlx5_0:1` (NIC:GID index)

---

### Phase 3 — Batch / async path (throughput optimization)

Goal: use `mput` / `mget` for high-throughput scenarios where multiple requests are in-flight
simultaneously between stages.

Relevant when:
- `max_batch_size > 1` in stage config (e.g., `max_batch_size: 64`)
- Multiple requests complete Stage 0 at nearly the same time

Implementation: batch `put()` calls collected within a time window, then flushed via `mput()`.
Mirror the `get_batches()` partitioning logic from AIBrix's `PrisKVConnector`.

```python
def _flush_batch(self, batch: list[tuple[str, bytes]]) -> None:
    """Send a batch of (key, payload) pairs via mput."""
    keys = [k for k, _ in batch]
    payloads = [v for _, v in batch]
    results = self._client.mput(keys, payloads)
    # check per-operation Status results
```

Enable via config:
```yaml
extra:
  use_mput_mget: true
  batch_window_ms: 10
```

---

## 5. Key Design Decisions

### 5.1 Consume-once semantics

The SHM connector deletes the shared memory segment after `get()`. The PrisKV connector should
do the same: call `client.delete(key)` after a successful `get()`. This prevents key accumulation
in the PrisKV server and matches the one-shot transfer pattern of all other connectors.

### 5.2 Key namespace

Use the base class `_make_key()` format: `{req_id}@{from_stage}_{to_stage}`. Add the `key_suffix`
from config as a namespace prefix to isolate vLLM-Omni keys from other users of the same PrisKV
server: `vllm_omni:{req_id}@{from_stage}_{to_stage}`.

### 5.3 Serialization in Phase 1

Reuse `OmniConnectorBase.serialize_obj()` / `deserialize_obj()` which call `OmniSerializer`
(msgpack + custom hooks for `torch.Tensor`, `np.ndarray`, `PIL.Image`, `RequestOutput`).
This means Phase 1 has the same serialization overhead as `SharedMemoryConnector` — the only
difference is the transport (network vs. `/dev/shm`).

### 5.4 Lazy connection

`PriskvClient` is initialized lazily on first `put()` or `get()`, not in `__init__`. This matches
the pattern of other connectors and avoids connection failures during the orchestrator setup phase
before stage workers are ready.

### 5.5 Metadata format

```python
# put() returns:
metadata = {
    "priskv_key": "vllm_omni:abc123@0_1",   # compact string, safe to send via ZMQ
    "size": 41943040,                          # bytes, for metrics
    # Phase 2 additions:
    # "rdma": True,
    # "shape": [1, 4096, 8192],
    # "dtype": "bfloat16",
}
```

The metadata travels via ZMQ queue (very small), while the actual payload stays in PrisKV.

---

## 6. Testing Plan

| Test | Level | Description |
|---|---|---|
| `test_priskv_connector_tcp.py` | Unit (CPU) | Mock `PriskvClient`; verify put/get/cleanup lifecycle |
| `test_priskv_serialization.py` | Unit (CPU) | Verify round-trip for all payload types (tensor, dict, RequestOutput) |
| `test_priskv_connector_integration.py` | L2 (GPU, requires PrisKV server) | Real PrisKV server; single-node put/get under load |
| `test_priskv_cross_node.py` | L3 (multi-node) | Two-node Qwen3-Omni pipeline; measure TTFT vs SharedMemoryConnector |
| `test_priskv_rdma.py` | L3 (RDMA NIC required) | Phase 2 RDMA path; verify zero-copy tensor transfer |

Test markers to add in `pyproject.toml`:
```toml
"priskv" = "Tests requiring a live PrisKV server"
"rdma" = "Tests requiring RDMA-capable hardware"
```

---

## 7. Implementation Sequence

```
Phase 1 (TCP, ~2 weeks)
  ├── priskv_connector.py skeleton + put/get/cleanup/health/close
  ├── factory.py registration
  ├── Unit tests with mocked PriskvClient
  ├── Integration test with local PrisKV server (Docker image)
  └── YAML config docs + example stage config

Phase 2 (RDMA, ~3 weeks)
  ├── _put_rdma / _get_rdma methods
  ├── register_slabs lifecycle (init, re-register on reconnect)
  ├── Threshold selection logic (TCP vs RDMA)
  ├── RDMA hardware tests (L3)
  └── Benchmark: latency vs SharedMemoryConnector vs MooncakeTransferEngineConnector

Phase 3 (Batch, ~1 week)
  ├── Batch collection window + mput/mget
  ├── Per-operation Status error handling
  └── Throughput benchmark at batch_size=8,16,32
```

---

## 8. Dependencies

```toml
# requirements/cuda.txt (add, optional)
aibrix_kvcache[priskv]  # provides PriskvClient; install only if PrisKV backend selected
```

Import is guarded behind a lazy `_ensure_connected()` so the connector only requires
`aibrix_kvcache` at runtime when actually configured. All other deployments are unaffected.

---

## 9. References

- [AIBrix repo](https://github.com/vllm-project/aibrix) — `python/aibrix_kvcache/aibrix_kvcache/l2/connectors/priskv/priskv.py`
- [AIBrix KVCache Offloading Framework docs](https://aibrix.readthedocs.io/latest/designs/aibrix-kvcache-offloading-framework.html)
- [AIBrix v0.4.0 release](https://aibrix.github.io/posts/2025-08-04-v0.4.0-release/) — KVCache V1 Connector + PrisDB integration
- [SGLang AIBrix storage integration](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/aibrix_kvcache) — reference for `KVCacheManager` API usage
- vLLM-Omni `OmniConnectorBase` — `vllm_omni/distributed/omni_connectors/connectors/base.py`
- vLLM-Omni `OmniConnectorFactory` — `vllm_omni/distributed/omni_connectors/factory.py`
- vLLM-Omni `OmniSerializer` — `vllm_omni/distributed/omni_connectors/utils/serialization.py`
