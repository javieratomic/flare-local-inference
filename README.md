<p align="center">
  <img src="docs/images/flare-logo.png" alt="Flare — ASCII wordmark above a solar flare" width="912">
</p>

## About this fork

This is a community fork of
[NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference), not an
official NVIDIA release. It targets three DGX Sparks cabled in a triangle with direct 200G
RoCE links, one rank per host. The default branch (`spark-tp3`) is current upstream `main`
plus the changes below. Use it at your own risk; for anything else, refer to upstream.

### What this fork adds over upstream

| Change | Upstream `main` | This fork | Upstream status |
|---|---|---|---|
| DeepSeek V4.1 Flash on three hosts (TP3) | TP4 only; host rendezvous admits 2 or 4 ranks | TP3, one rank per host: query heads padded 64 -> 72 with zero weights, vocabulary 129280 -> 3 x 43136, rendezvous for 2 to 16 ranks | Proposed in [#21](https://github.com/NVIDIA/flare-local-inference/issues/21) ([`tp3-deepseek41`](https://github.com/javieratomic/flare-local-inference/tree/tp3-deepseek41)) |
| Per-rank weight estimate | Divides replicated weights by the rank count (TP4 plans 71.4 GiB, measures 74.2 GiB) | Counts embeddings, vision encoder and other replicated weights on every rank | Part of #21 |
| Multi-host RoCE proxy | Packages exported on current `main` cannot serve on several hosts ("proxy ABI 10, this runtime provides 9", [#7](https://github.com/NVIDIA/flare-local-inference/issues/7)) | Helios proxy aligned with the locked FlashInfer (ABI 10) | Fork-only; tracked by #7 |
| Prefix cache on disk across hosts | Refused: "the host prefix store requires every rank in one process" | Each host keeps its own rank's records on its NVMe; rank zero indexes and evicts, every rank runs each job; records survive restarts | Proposed in [#23](https://github.com/NVIDIA/flare-local-inference/issues/23) ([`prefix-cache-multihost-upstream`](https://github.com/javieratomic/flare-local-inference/tree/prefix-cache-multihost-upstream)) |
| `flare orbit run --prefix-cache-*` | Accepted but not forwarded to the ranks, so silently ignored | Forwarded to every rank | Part of #23 |

### Measured on 3x GB10 (DeepSeek V4.1 Flash, TP3, CSF, DSpark)

- Decode: 39.8 tok/s with one stream, 84.4 tok/s aggregate with four.
- Each rank holds about 93.6 GiB of weights and keeps about 7.7 GiB for KV.
- With `--prefix-cache-memory 0 --prefix-cache-disk DIR --prefix-cache-disk-limit 200G`, the
  "from disk" column restarts all three ranks first, so the prefix comes only from each
  host's NVMe:

| Prompt tokens | Cold prefill | From disk after restart |
|---|---|---|
| 65k | 21.5 s | 2.0 s |
| 262k | 87.1 s | 2.4 s |
| 1.04M | 581.9 s | 4.1 s |

- Answers to 16-key recall prompts were 16/16 and byte-identical across cold, cached and
  disk runs.
- The 262k and 1M answers match a vLLM TP3 deployment of the same checkpoint on the same
  hosts.
- A 1M-token prompt takes about 1.1 GB of disk per host.

### Using it

To prepare the model, follow the "DeepSeek V4.1 Flash" section of
[docs/model-preparation.md](docs/model-preparation.md) with `--ranks 3`. To serve it, follow
[docs/multi-host.md](docs/multi-host.md), adding the `--prefix-cache-*` options above to
`flare orbit run` to keep the prefix cache on disk. On DGX OS, preparation needs
`python3.12-dev` and `liburing-dev` (b12x.loader builds a native extension) and the system
CPython 3.12; standalone Python builds without `os.memfd_create` fail on the peer ranks.
[docs/prefix-cache-offload.md](docs/prefix-cache-offload.md#214-ranks-on-separate-hosts)
describes the multi-host prefix store.

---

> **Coding agents:** skip this README and read [AGENTS.md](AGENTS.md) directly.

Flare is a local inference server for NVIDIA SM120/SM121 GPUs, built for
long conversations, coding agents, and repeated tool use. Its goal is to make
the memory, bandwidth, and compute in workstations and compact systems such
as DGX Spark go further, while keeping generation responsive.

This is a research prototype, and is made available via open-source to enable
technical collaboration and collect early feedback.

The design rests on a few principles:

- **Embrace the standard PyTorch ecosystem at build time; serve from Rust.**
  Ignition uses PyTorch, Transformers, and FlashInfer/b12x to load models,
  tune kernels, compile forwards, and capture CUDA graphs into a runtime package.
  Helios, the Rust runtime, serves that package without Python. Compilation and
  graph capture finish before serving.

- **Build on the model ecosystem.** Reuse canonical model implementations
  wherever their dataflow fits, and specialize the parts that need different
  layouts or execution. Model development keeps the tools and libraries that
  make the Python ecosystem productive.

- **Make memory and state explicit.** Plan buffers and execution capacities
  ahead of time. Treat conversation state—including recurrent state alongside
  KV caches—as a whole, so reuse, branching, and resuming preserve it correctly.

- **Keep the runtime small and shared.** Model recipes own model computation;
  the runtime owns scheduling, cache storage, and request handling. A focused
  hardware and kernel stack keeps those boundaries manageable.

[vLLM](https://docs.vllm.ai/en/latest/) covers a much broader range of models,
hardware, and deployment patterns. Flare trades that breadth for a focused
serving contract around selected models and local workloads. Flare is not, and
will never, be a replacement for production inference englines like vLLM.

Think of Flare as a specialized option when its hardware and model coverage
fit your machine.

Start with [model preparation and serving](docs/model-preparation.md).
See the [CLI guide](docs/cli-design.md) for configuration and the
[core design](docs/core-design.md) for the architectural reasoning.
