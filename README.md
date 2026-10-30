<p align="center">
  <img src="docs/images/flare-wordmark.png" alt="Flare — ASCII wordmark above a solar flare" width="912">
</p>

> **Coding agents:** skip this README and read [AGENTS.md](AGENTS.md) directly.

## About this branch

This `integration` branch is upstream `main` plus the changes below, proposed
upstream in four pull requests and one issue, none merged yet. They speed up
multi-host model preparation, and were measured with DeepSeek V4.1 Flash on DGX
Spark (GB10) hosts. Measurements are from TP3 exports on three hosts, one run
per configuration. Each row is measured against the build it was developed on,
not always upstream `main`; the linked item gives the exact baseline.

| Change | Upstream | Phase | Before | After |
|---|---|---|---|---|
| `ignition join --local-checkpoint [require\|prefer]`: peers read their own copy of the checkpoint | [#126](https://github.com/NVIDIA/spark-local-inference/issues/126) (issue) | Load weights on the peers | 252 s | 44.7–50.1 s (per peer) |
| `--compile-headroom SIZE` / `[preparation] compile_headroom`: host memory left free for compilers (default stays 8 GiB) | [#122](https://github.com/NVIDIA/spark-local-inference/pull/122) | Compile model forwards | 905.7 s | 507.7 s (6 GiB) |
| Link received payload objects into the package cache instead of copying | [#125](https://github.com/NVIDIA/spark-local-inference/pull/125) | Receive rank export payloads | 457.8 s | 280.6 s |
| Hash Engram planes on parallel readers | [#157](https://github.com/NVIDIA/spark-local-inference/pull/157) | Assemble package payloads | 262.0 s | 169.1 s |
| Hash package payload sources, and resolve shared payload cache objects, on parallel workers | [#124](https://github.com/NVIDIA/spark-local-inference/pull/124) | Verify and publish shared data | 863.7 s | 426.9 s |
| The same, with the shared data already cached; Before is #124 without its cached-object verification | [#124](https://github.com/NVIDIA/spark-local-inference/pull/124) | Verify and publish shared data (shared data already cached) | 238.3 s | 149.6 s |

The packages exported with #122, #124, #125 and #126 served normally with
`flare run` (262k-token retrieval 16/16).

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
