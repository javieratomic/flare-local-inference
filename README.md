<p align="center">
  <img src="docs/images/flare-logo.png" alt="Flare — ASCII wordmark above a solar flare" width="912">
</p>

> **About this fork.** This is a community fork of
> [NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference). It is not
> an official NVIDIA release. It adds tensor-parallel preparation of DeepSeek V4.1 Flash on
> **three** DGX Sparks (TP3, one rank per host), for clusters of three Sparks cabled in a
> triangle. Upstream supports TP4 only. This branch (`spark-tp3`) is current upstream `main`
> plus the five TP3 commits of
> [`tp3-deepseek41`](https://github.com/javieratomic/flare-local-inference/tree/tp3-deepseek41),
> which is proposed upstream and may never be accepted there. It also carries one fork-only
> commit that aligns the Helios RoCE proxy with the locked FlashInfer (proxy ABI 10). Without
> it, no multi-host package exported on current upstream `main` can serve (upstream issue #7).
> The five TP3 commits add:
> - padded tensor-parallel partitions and loaders: query heads 64 -> 72 (zero weights), and
>   vocabulary 129280 -> 3 x 43136;
> - host rendezvous for 2 to 16 ranks;
> - corrected per-rank weight estimates for replicated weights.
>
> To prepare the model, follow the "DeepSeek V4.1 Flash" section of
> [docs/model-preparation.md](docs/model-preparation.md) with `--ranks 3`. To serve it, follow
> [docs/multi-host.md](docs/multi-host.md). On DGX OS, preparation needs `python3.12-dev` and
> `liburing-dev` (b12x.loader builds a native extension) and the system CPython 3.12;
> standalone Python builds without `os.memfd_create` fail on the peer ranks.
>
> **Measured on 3x GB10.** Results with CSF and DSpark, using RoCE over direct 200G links:
> - decode: 39.8 tok/s with one stream; 84.4 tok/s aggregate with four;
> - cold prefill: 262k in 83 s, 523k in 195 s;
> - 16/16 recall at 262k (output identical to vLLM TP3) and at 1.04M tokens (575 s).
>
> Each rank holds about 93.6 GiB of weights and keeps about 7.7 GiB for KV.
> Use at your own risk; for anything else, refer to upstream.

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
