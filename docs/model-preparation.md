# Model preparation and serving

## Runtime packages

Run commands from the repository root. Ignition prepares a source checkpoint
into a **runtime package**: a metadata-only primary `.safetensors` file and
immutable sibling payload blobs named by their SHA-256 hashes. The primary's
`flare.metadata` contains the signed Protobuf envelope; `flare.layout` maps its
logical resources to the blobs. `flare.payload` is empty. Weights, embedding
tables, graphs, kernels, and frontend assets all reside in payload blobs.
Adjacent weight resources owned by one rank share shards up to the file-size
limit. Graphs, kernels, and supporting resources use separate shards so changing
them does not rewrite unchanged weight shards. Rank ownership and logical file
boundaries remain separate for selective distribution. Export, conversion, and
signing use the same grouping rule.

The published payload-file limit is 64 GB (64,000,000,000 bytes), including the
header. The reader also accepts existing payloads up to 64 GiB. `--shard-size SIZE`
overrides `[publishing] shard_size` with a smaller positive limit. Resources
remain contiguous; export rejects a resource that exceeds the limit. Existing
monolithic and numbered-shard packages remain loadable.

Exports hardlink unchanged blobs from `paths.package_cache`. `ignition export
--no-weights` requires exact cached stored-data payloads and creates only missing
execution-resource blobs and metadata. Preparation and weight reads still occur.
Recipes can declare `export_repository`; without `--output`, exports use a unique
snapshot directory under that repository's local cache namespace. See
[package snapshots](cli-design.md#package-snapshots) for conversion and upload.

Tensor-parallel packages contain one resource bundle per rank.
Final packages use a format-4 signed envelope and encrypted cubins and graph
records. Ignition creates a per-user development signing key when `[publishing]`
is absent. `ignition run` passes the exact export key and publisher to Flare for
unattended approval; direct `flare run` needs the corresponding publisher profile
and first-use acceptance. See [signing and development](cli-design.md#signing-and-local-development).
Unsigned rank staging is private to preparation and cannot load in Helios.

Flare loads the package files without Python. Helios is a Rust library;
`flare_heliopause` is a CPython extension imported by Ignition.

Every recipe defaults to FP8 E4M3 KV storage, including learned draft-model
caches. Qwen Flash retains BF16 selector keys; GLM and DeepSeek retain their
packed latent formats, including scale metadata and any BF16 rotary fields.
Recurrent state, compressor accumulators, and hidden activations retain their
model-specific precision. Cache writers and attention readers use matching
scales, and memory planning includes FP8 pages and partial-page snapshots.
Export a package to use these cache layouts; existing packages retain the
layouts captured when they were prepared.

## Environment setup

Set up the preparation environment and build the native CLI:

```sh
uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare/Cargo.toml
```

Build dependencies include a C/C++ compiler, CMake, Ninja, pkg-config, Protobuf's
`protoc`, OpenSSL headers, Python headers, and liburing headers. On Ubuntu:

```sh
sudo apt-get install build-essential cmake ninja-build pkg-config protobuf-compiler libssl-dev python3-dev liburing-dev
```

The JPEG decoder builds its bundled libjpeg-turbo source. Preparation requires
the CUDA toolkit, including `nvcc` and CUPTI; configure `[paths] cuda` when it
is not supplied by the Python environment.

The preparation dependency pins a Git revision of `local-inference-lab/flashinfer`
in `pyproject.toml`; `uv.lock` records the same commit. To change it, update the
revision, run `uv lock --upgrade-package flashinfer-python`, then sync the environment.
For local FlashInfer development, install the sibling
checkout explicitly with `uv pip install --python .venv/bin/python --no-deps -e ../flashinfer`;
`uv sync --locked --extra prepare --extra test` restores the Git dependency.
Inspect that checkout and its local changes before modifying it. After changing
Rust code in `heliopause/`, or Helios code used by the extension, rebuild the
installed extension explicitly:

```sh
uv pip install --python .venv/bin/python --no-deps ./heliopause
```

## GPU selection

Inspect GPU occupancy with `nvidia-smi` before preparation or loading. Select
available GPUs explicitly on shared hosts. `--gpus 2` means one rank on GPU 2;
`--gpus 0,1` means two ranks on GPUs 0 and 1. With `CUDA_VISIBLE_DEVICES` set,
indices refer to the exposed device list. Each tensor-parallel size requires a
separately prepared package; a TP2 package cannot be loaded as TP1. Qwen TP1 and TP2 and
GLM TP4 have been exercised end to end; Qwen TP4 still requires model validation.

## Preparation resources

`--compile-workers N` sets a positive CPU compiler process budget per host for
b12x kernel compilation and complete-forward compilation. Available RAM and
device memory can reduce it; local preparation ranks share the budget. Set
`[preparation] compile_workers = N` for a persistent default; the CLI flag
overrides that setting. Omitting both retains automatic sizing, including
shared-memory limits on DGX Spark.

`--tuning-gpus LIST` adds local GPU replicas for benchmarking b12x candidates.
The list begins with `--gpus` and continues in complete TP groups: for example,
`--gpus 0 --tuning-gpus 0-3 --compile-workers 24` prepares a TP1 package with
four tuning GPUs. Corresponding GPUs in each replica must match in model and
architecture and have enough free memory for the model and preparation storage.
Helpers exit and release their resources after tuning; serving TP size and capture
devices stay fixed. Serving ranks then compile and capture their forwards.
Both GPU lists use `CUDA_VISIBLE_DEVICES` ordinals. Kernel compilation runs in
CPU workers, independently of the number of tuning GPUs.

Multi-host `--ranks` preparation retains its existing sharding and transport;
`--tuning-gpus` is local-only and cannot be combined with it. An explicit
`--compile-workers` budget applies separately on each host.

## Exporting runtime packages

Choose an unused output filename; export appends `.safetensors` if needed and
refuses to overwrite one. Do not
delete an existing package merely to reuse its name. `MODEL` can also be a
supported repository ID or a local source-checkpoint directory. Ignition
downloads missing source weights, loads and tunes kernels, compiles forwards,
warms and captures execution variants, writes the package, and exits.
Compilation is enabled by default; `--no-compile` is for development.

## Nemotron 3.5 Bolt 74B-A9B 24GB

`nemotron-bolt-24gb` prepares
`nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` on one SM120/SM121 GPU.
The recipe includes the C-RADIO encoder for still images. Video is not prepared. LiLiCorr uses the matching
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` checkpoint and defaults to
seven draft tokens with adaptive verification. Both checkpoints use the standard
Hugging Face hub cache (`~/.cache/huggingface/hub`), or `[paths].hub_cache` when configured.

```sh
.venv/bin/python -m ignition export nemotron-bolt-24gb --gpus 0 \
  --max-concurrency 8 --prefill-chunk-size 4096 --mtp-draft-tokens 7 \
  --speculative-token-capacity 64 \
  --output /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/nemotron-bolt-images-lilicorr-c8.safetensors \
  --gpus 0 --bind 127.0.0.1:8001
```

Use `--no-mtp` during preparation to omit the draft model. Mamba-2 prefill
uses b12x's packed selective scan; decode and speculative verification use
FlashInfer state updates. Continuation rows retain BF16 recurrent state,
convolution history, partial FP8 KV pages, and LiLiCorr context and rollback
records. Prefix reuse and speculative acceptance preserve that complete state
through the shared Helios cache contract. Image features enter causal prefill
and may cross chunk boundaries.

Shared experts overlap routing and routed experts for graph capacities through
64 rows. Larger capacities execute those branches serially. LiLiCorr's separately
scaled NVFP4 gate and up projections also overlap. Prepare with
`--no-stream-overlap` to place those projections and expert branches on the main stream.
Routing excludes padded target rows using the packed batch's valid-token count.
The 64-row speculative capacity covers eight requests with seven draft tokens each.
For concurrency 32 with seven draft tokens per request, use
`--max-concurrency 32 --speculative-token-capacity 256`.

## Qwen3.8 27B

`qwen3.8-27b` prepares `nvidia/Qwen3.8-27B-NVFP4` for text generation on one GPU.
It uses the canonical Qwen3.5 dense decoder with tensor-scaled FP8 attention
projections and NVFP4 MLP and vocabulary weights. FP8 KV pages and Gated DeltaNet
continuations reside in device memory; the recipe has no host-mapped tables,
vision graphs, or learned draft model. Prepare on the same GPU architecture
as the serving device:

```sh
.venv/bin/ignition export qwen3.8-27b --gpus 0 --max-concurrency 4 \
  --output /data/flare-runtime/qwen38-27b-tp1-c4.safetensors --verbose
```

## Qwen3.8 Flash

Export Qwen with up to five MTP draft tokens per round:

```sh
.venv/bin/ignition export qwen3.8-flash --gpus 2 \
  --mtp-draft-tokens 5 --max-concurrency 4 \
  --output /data/flare-runtime/qwen38-tp1-mtp5.safetensors --verbose

.venv/bin/ignition export qwen3.8-flash --gpus 0,1 \
  --mtp-draft-tokens 5 --max-concurrency 4 \
  --output /data/flare-runtime/qwen38-tp2-mtp5.safetensors --verbose
```

Qwen defaults to five MTP draft tokens, with explicit flags taking precedence.
Changing this default does not change an already exported package.

Vision recipes accept `--image-concurrency N`, independently of
`--max-concurrency`. It defaults to `min(4, max_concurrency)` and can also be set
as `image_concurrency` in `[defaults]` or a model's configuration section.
Explicit flags override configuration. Text-only recipes ignore the configured
image default and reject an explicit image-concurrency flag.
Helios reserves a feature slot when a request needs vision computation. Requests
whose image spans are already covered by a reused prefix checkpoint need no
feature slot. Requests needing a slot queue when those slots are occupied;
eligible requests continue to admit. A reserved slot remains owned until
request retirement, including completion of outstanding GPU work after cancellation.
The limit does not change images per request or the prepared patch capacities.
Changing it requires exporting the package again.
Packages with vision require the prepared feature-slot capacity and binding.
Images replay one at a time into bounded windows in the request's feature buffer.
Chunked prefill consumes each window before the buffer is reused. Combined image
tokens may exceed the buffer's capacity; each image must fit one window and a
captured patch capacity. HTTP body and total context limits also apply.
PNG, JPEG, and static WebP data URLs are accepted.

For Qwen, each feature slot holds 16,384 BF16 vectors of width 2,560: 80 MiB per
rank. Four image slots therefore reserve 320 MiB even with 32 active request slots.

N-gram tables use indirect delivery exclusively: every graph consumes compact
GPU row buffers filled by Helios. Qwen PLE tables are stored once in the package.
Helios reads the 26.8 GiB table from disk by default on every GPU;
`flare run --embedding-table-storage ram` keeps it in resident host memory.
The graph publishes GPU-produced lookup IDs before layer
0 and waits for row delivery at the layer-1 consumer. Disk and RAM delivery
use the same prepared model graphs; the flag selects table storage only.

## DeepSeek V4.1 Flash

The `deepseek-v4.1-flash` alias selects `deepseek-ai/DeepSeek-V4.1-Flash-1017`
and supports TP4, or TP3 with one rank per host, for text generation, reasoning,
DSML tool calls, and images
when the checkpoint contains its vision encoder.
Exports use the native DeepSeek V4.1 prompt encoder and include only the tokenizer
asset. Vision exports bind the atomic `<｜deepseek_image｜>` token as ID 129264,
which must agree with checkpoint `image_token_id`. Packages without explicit
prompt-encoder metadata continue to use their tokenizer and Jinja template.
Sampling defaults are temperature 1.0 and top-p 1.0, following the
[model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference).
Add `--expert-scale-compression csf` to losslessly compress backbone and DSpark
MXFP4 expert scales during loading. Compressed scales stay resident, with shared
expansion buffers reused by serial expert layers. Planning conservatively estimates
4.7% less resident weight storage; loading measures the actual allocation. Omitting
the flag preserves native scales.
DSpark is its learned draft method, defaulting to five draft tokens per round
with separate proposal and verification graphs, using native conditional
confidence and measured execution cost for adaptive verification. Image preprocessing
preserves the checkpoint's row-major patch layout and 1024-token image limit.
Its 188.8 GiB of Engram tables are stored once in the package, partitioned by
hash head across TP ranks. Heads are assigned round-robin, so each TP4 rank
owns two bigram, two trigram, and two four-gram heads, with their projection
input columns. Table rows and projection scales retain checkpoint precision.
Each rank reads its table rows, computes a partial
projection, and reduces the result before residual mixing. Multi-host package
distribution transfers only the table partitions owned by the receiving rank.
Disk storage reads rows on demand through bounded direct-I/O staging;
`flare run --embedding-table-storage ram` loads the assigned partitions into
ordinary resident memory. Both modes copy compact rows into each rank's prepared
buffers and signal captured consumer events. Preparation appends the immutable
checkpoint slices without materializing the tables. Packages containing shared
table planes retain their prepared replicated computation and shared row delivery.
For disk storage, `--embedding-table-resident-scales` retains the original scale
bytes in RAM and removes scale reads on row-cache misses. DeepSeek V4.1 Flash
TP4 uses about 1.4 GiB per rank for these scales, in addition to any row-cache
budget. The loader reserves this memory before sizing KV storage.
Select four available GPUs explicitly:

```sh
.venv/bin/python -m ignition export deepseek-v4.1-flash --gpus 0,4,6,8 \
  --max-concurrency 4 --mtp-draft-tokens 5 --prefill-chunk-size 4096 \
  --expert-scale-compression csf \
  --output /data/flare-runtime/deepseek-v41-flash-1017-csf-tp4-c4-dspark5.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/deepseek-v41-flash-1017-csf-tp4-c4-dspark5.safetensors \
  --gpus 0,4,6,8 --host 127.0.0.1 --port 8001
```

For four DGX Sparks with one GPU per host, replace `--gpus 0,4,6,8` during
export with `--gpus 0 --ranks 4 --rendezvous 10.200.0.12:29630` and start an
`ignition join` process on each peer. Rank zero retains the checkpoint; peers
request their weight ranges during preparation. From a paired controller, use
`flare orbit run maxwell:/data/flare-runtime/deepseek-v41-flash-1017-csf-tp4-c4-dspark5.safetensors --cluster sparks --gpus 0`
with a four-node `sparks` cluster. The prepared model uses RoCEnante collectives
across hosts. See [multi-host preparation and serving](multi-host.md) for
preparation authentication, orbit setup, and rank placement.

Three DGX Sparks prepare TP3 with `--ranks 3` and two `ignition join` peers.
Ranks own whole eight-head output groups, so attention pads 64 query heads to
72 and its eight output groups to nine; the padded group has zero query and
output weights and adds nothing to the attention output. The vocabulary head
pads 129280 rows to 3 x 43136, and gathered logits drop the padded columns.
Expert widths (2304 / 3 = 768) and Engram heads (24 / 3) divide exactly. Each
rank holds about 93.7 GiB of weights with `--expert-scale-compression csf` and
DSpark, about 98.1 GiB without compression, leaving roughly 8 GiB of KV pages
on a Spark with 115 GiB available and a 4 GiB reserve. Single-host TP3 is not
supported: b12x PCIe vocabulary gathers and DMA reductions do not partition
across three ranks.

`--image-concurrency` defaults to the smaller of four and `--max-concurrency`.
For example, add `--max-concurrency 64 --image-concurrency 4` during export to
prepare 64 text request slots while retaining four independent image feature slots.
The native server can lower text concurrency at load time. Engram histories
exclude image positions, so n-grams cannot cross an image boundary.

DeepSeek V4.1's logical cache pages cover 128 tokens across compressed latent and
index regions with different byte sizes. Continuation rows retain sliding
windows, pending compression state, partial pages, DSpark context windows, and the
last hidden state. Both use runtime-sized virtual ranges; exported packages
contain their layouts and capacity bounds without cache payloads.
Engram indexing runs in the graph over committed history and candidate tokens.
The three preceding tokens live in continuation state, including reused prefixes.
DSpark proposals, target verification, and accepted-state commit use the shared
native speculative contract; Helios contains no DSpark model implementation.

## DeepSeek V4 Flash

DeepSeek V4 Flash uses `local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF`, including its
vision encoder and checkpoint-integrated DSpark model. The `deepseek-v4-flash`
alias prepares five draft tokens by default, with adaptive verification. It
accepts images within its image-token budget and has no offloaded embedding tables.
Exports use the native DeepSeek V4 prompt encoder, carry only the tokenizer asset,
and bind the atomic `<｜deepseek_image｜>` placeholder as token 129264.
Both native DeepSeek encoders default to thinking enabled and high effort, accept
`low`, `high`, `xhigh`, and `max`, and reject numeric effort, `drop_thinking`,
assistant continuation, and `add_generation_prompt=false`. System-prefix caching
derives reusable token prefixes from the official encoder's rendered conversations.
Previously exported Jinja packages retain their
packaged rendering behavior and require both tokenizer and template assets.
Use two or four available GPUs for preparation and serving:

```sh
.venv/bin/python -m ignition export deepseek-v4-flash --gpus 3,4,5,6 \
  --max-concurrency 4 --mtp-draft-tokens 5 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/deepseek-v4-flash-vision-tp4-c4-dspark5.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/deepseek-v4-flash-vision-tp4-c4-dspark5.safetensors \
  --gpus 3,4,5,6 --host 127.0.0.1 --port 8001
```

V4 uses 256-token logical pages containing 64 C4 latent records with an FP8
retrieval index, or two C128 latent records with dense history attention. Each latent page
stores FP8/BF16 payloads followed by its own scale footer. Continuation rows
preserve the complete 128-token compression interval, C4 overlap, sliding
windows, partial pages, and DSpark context. Only accepted tokens update the
continuation state. Bidirectional image spans stay within one prefill chunk;
the scheduler shortens a preceding text chunk when necessary. Image preprocessing,
feature delivery, text execution, and output parsing all run natively in Helios.
Packages with 128-token V4 pages are incompatible and must be re-exported.

## MiMo V2.6 Pro MOPD

MiMo V2.6 Pro MOPD uses `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` with checkpoint
MXFP4 experts, block-FP8 dense projections, and BF16 draft and vision weights.
The `mimo-v2.6-pro-mopd` alias prepares text, images,
and seven DFlash draft tokens with adaptive verification. Audio is not prepared.
Select eight available GPUs:

```sh
.venv/bin/ignition export mimo-v2.6-pro-mopd --gpus 0,1,2,3,4,5,6,7 \
  --max-concurrency 4 --mtp-draft-tokens 7 --prefill-chunk-size 4096 \
  --output /data/flare-runtime/mimo-v26-pro-mopd-tp8-c4-dflash7.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/mimo-v26-pro-mopd-tp8-c4-dflash7.safetensors \
  --gpus 0,1,2,3,4,5,6,7 --host 127.0.0.1 --port 8001
```

Global attention uses paged KV storage; continuation rows retain sliding-window
KV, partial pages, and accepted DFlash context. DFlash consumes selected target
layer features and predicts an eight-position block containing the pending
token and seven proposals. Helios uses the shared speculative execution contract;
draft modeling and state updates are captured by Ignition. The vision encoder
publishes compact image features for causal text prefill. Loading with
`--mtp-draft-tokens 0` selects ordinary execution without draft-state updates.

## Step-5 Preview

Step-5 Preview uses `stepfun-ai/Step-5-Preview-NVFP4`. The
`step5-preview-nvfp4` alias prepares text, images,
and three causal MTP draft tokens with adaptive verification. Checkpoint NVFP4
routed experts use W4A4. Routed experts declared BF16 in checkpoint metadata
are converted to NVFP4 during loading and use b12x W4A16 with BF16 activations;
the converted weights are retained in captured graphs and exported packages.
Shared, dense, and MTP MLP gate/up weights are packed and converted to MXFP8
during loading. Target-model projections select BF16 or quantized activations
through autotuning. MTP projections retain BF16 activations to bound rounding
error through the predictor feedback chain. Down and attention projections remain
BF16, with FP32 residual accumulation. Paged and sliding-window
K/V, including MTP windows, use FP8 E4M3 with unit descales;
sliding attention consumes live candidates in BF16 before committing them to FP8 history;
Q/K normalization and rotary computation retain their BF16 rounding before the
FP8 store. With CSF scales and shared scratch, the concurrency-16/MTP3 graph
catalog, including vision, fits on four RTX PRO 6000 Blackwell GPUs:

```sh
.venv/bin/ignition export step5-preview-nvfp4 --gpus 0,1,2,3 \
  --max-concurrency 16 --mtp-draft-tokens 3 --no-mtp-adaptive-verification \
  --speculative-token-capacity 16 --prefill-chunk-size 4096 --expert-scale-compression \
  --output /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/step5-tp4-mtp3-c16-fp8kv.safetensors \
  --gpus 0,1,2,3 --host 127.0.0.1 --port 8001
```

Add `--expert-scale-compression csf` to compress all Step-5 routed-expert block
scales losslessly during loading. Both checkpoint W4A4 experts and converted
W4A16 experts use CSF, retaining their activation modes. Serialized layers share
one pair of expansion buffers, whose contents are zero-initialized on export.
One transient loading pool reuses conversion scratch across layers and retires
after weight loading, retaining live weights for graph export. The separate
graph-capture pool remains alive. Vision, prefill, decode, and draft activations
reuse typed graph buffers. MTP feedback, sampling, metadata, and resume scratch
reuse a separate buffer pool across graph variants, with disjoint buffers within
each graph. Sequential predictors share their norm-concatenation output. Kernel
workspaces use the shared operator arena, with separate slices for overlapping
expert branches. Compiler-created intermediates use one shared CUDA graph pool.
The memory plan estimates 6% lower resident weight storage; actual memory depends
on encoded exceptions and is reported after loading. Larger concurrency requires
separate preparation and memory validation.

Sliding attention keeps a 512-token continuation window. Sparse attention uses
128-token logical pages containing FP8 keys and values and an FP8 retrieval
index for eight-token regions. Continuation rows also preserve partial index
regions and accepted MTP state. Accepted target tokens, hidden states, and embeddings
remain in continuation storage until the following round consumes them in three
packed predictor passes. Candidate KV never enters committed predictor windows.
Region scoring, selection, and attention use
b12x kernels. Attention output gating is fused into the direct output or split
merge store, preserving BF16 rounding of attention and sigmoid before their
product. SwiGLU clamps the gate after SiLU. Helios manages these resources
through the shared cache and speculative execution contracts. The perception encoder
prepares 504-pixel local crops and a 728-pixel overview. Serving defaults follow
the model card: temperature 1, top-p 0.95, no top-k filter, and neutral penalties.

One-shot TP reductions with up to 36 rows fuse the residual update and
zero-centered RMSNorm into the collective, preserving FP32 residual storage and
BF16 normalized outputs. Attention and dense MLP projections enter the fused
collective directly as BF16; routed MLP updates enter as FP32. Attention and dense
MLP reductions round to BF16 before residual addition; routed MLP reductions retain
FP32. MLP reductions consume the following layer's input-normalization weight.
The final target and predictor MLPs
retain their separate final residual update. DMA prefill and hierarchical
collectives use separate reduction and normalization kernels.

## Kimi K3

Kimi K3 uses the local `/data/models/Kimi-K3-MXFP4-CSF` checkpoint. The
`kimi-k3-mxfp4-csf` alias prepares text generation, reasoning, XTML tool calls,
and images on sixteen GPUs. It preserves MXFP4 experts
with compressed scales, BF16 projections, and FP32 KDA recurrent state.
Its 93-layer decoder combines KDA, unrotated latent attention, attention
residuals, and latent SiTU experts. Red Hat DSpark uses
`RedHatAI/Kimi-K3-speculator.dspark` revision
`38a88101e0d46bb22134b9da340f381b954d40d4`, downloaded to the Hugging Face hub
cache (or `[paths].hub_cache`), and defaults to seven draft tokens per round.
A target checkpoint may instead carry that draft in its `dspark/` directory;
`kimi-k3-dspark` selects this layout at `/data/kimi-k3-dspark`.

For Kimi K3, `--online-mxfp8` converts KDA input, forget-gate, and output
projections; MLA latent/gate, query, and output projections; shared
expert gate/up and down projections; and both tensor-parallel layouts of the
MoE latent down projection and its latent up projection. Activations remain
BF16. MLA's batched key/value projections, routers, the first dense MLP,
embeddings, vocabulary head, and vision encoder retain checkpoint
precision. DSpark converts its context projection, attention outputs, and MLPs.
Shared and routed experts keep disjoint
kernel scratch when their execution overlaps.

```sh
.venv/bin/python -m ignition export /data/models/Kimi-K3-MXFP4-CSF --gpus 0-15 \
  --max-concurrency 4 --image-concurrency 4 --mtp-draft-tokens 7 \
  --prefill-chunk-size 4096 --compile-workers 16 --online-mxfp8 \
  --output /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-images-redhat-dspark-7-mxfp8.safetensors --verbose

./flare/target/release/flare run \
  /data/flare-runtime/kimi-k3-mxfp4-csf-tp16-c4-images-redhat-dspark-7-mxfp8.safetensors \
  --gpus 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 --host 127.0.0.1 --port 8001
```

The draft consumes AttnRes mixtures entering target layers 24, 48, 72, 88,
and 92, before input normalization. Its five causal Qwen3 layers use
2,048-token BF16 key/value windows. An anchor followed by mask tokens produces
seven proposals; each preceding proposal conditions the Markov bias.
Ignition uses greedy proposals and adaptive target verification widths,
without the draft's confidence head. The
[Red Hat model card](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark)
reports eight-token probabilistic proposals with block rejection sampling,
which differs from this execution configuration. Only accepted target features
update draft windows; KDA verification commits the accepted recurrent prefix.
Loading with `--mtp-draft-tokens 0` selects ordinary execution; exporting with
`--no-mtp` excludes draft weights and state.

Target FP8 MLA latent records occupy 128-token pages.
With context parallelism across sixteen ranks, concurrency four, and a
4096-token prefill chunk, preparation backs ten logical KV pages and nine
continuation rows: one null row, four live rows, and four snapshot destinations.
Retaining CUDA graph handles for export does not expand this working set.
Native package loading sizes serving caches from available device memory;
in-process diagnostic serving limits admission to physically backed storage.
Continuation rows preserve KDA
convolution and recurrent state, DSpark key/value windows, partial latent pages, and the normalized
last hidden state. The MoonViT encoder publishes hidden-column-sharded features
for causal text prefill, with storage for 65,536 patches per active feature
window. Conversations can consume multiple windows through chunked prefill.
XTML encoding distinguishes protocol tokens from literal marker text.
Serving defaults follow the model card's agentic sampling recommendation:
temperature 1, top-p 1, no top-k filter, and neutral penalties. Reasoning is
always enabled; requests to disable thinking are ignored without an error.
Effort defaults to `max`; requests can select `low`, `high`, or `max` and
override sampling defaults. Preserve assistant reasoning in conversation history.

## Storage and graph capacities

Every model export prints a VRAM inventory for each rank after final publication,
including exports without `--verbose`. Heliopause measures device usage and package
allocator mappings; Ignition attributes retained ranges to checkpoint tensors,
scratch pools, graph temporaries, I/O and persistent state, communication buffers,
kernel globals, and allocator slack. Owner fields and defining source locations
identify the largest consumers in each category. Aliased ranges count once;
shared scratch displays representative tensor owners.

Preparation-only storage is separate from fixed package storage. Cache figures
count physically backed pages, not their virtual serving capacity. Mapped host
storage and imported peer mappings are listed outside local VRAM totals. PyTorch
allocated and reserved bytes overlap the attributed storage. Cubin data sizes are
not resident code sizes: graph executable overhead, kernel code, other driver
allocations, and other processes remain in the unattributed device-usage remainder.
The returned `RuntimePackage.memory_inventory` contains every attributed source
group; console output shows the largest five per category. Source locations name
owning classes or closures, not allocation stack traces.

Ignition commands accept `--config PATH`; the default configuration is
`~/.config/flare/config.toml`. Storage locations are configured with
`[paths] hub_cache`, `kernel_cache`, and `package_cache`; `[memory] reserve`
leaves GPU memory available for other processes during preparation and serving.
Ignition additionally budgets 1 GiB for warm-up and capture. Preparation KV backing
covers a prefill chunk plus boundary pages per captured request row, unless an
explicit preparation page count is supplied programmatically. Continuation
backing covers the null row, live request and verification rows, and snapshot
destinations used during preparation. Planned prefix-cache slots remain virtual;
they do not require physical backing during export. The memory report separates
the declared continuation extent from `preparation_state_rows` and
`preparation_state_bytes` where those extents differ. Both caches retain their
full virtual extents in the package. Helios
sizes its caches after uploading the selected graphs, keeping a separate 64 MiB
driver margin.
At package load time, `flare run --kv-cache-limit context` bounds the shared
KV pool by one prepared context; `concurrency` uses one context per selected
active request, and the default `memory` fills the available budget. Runtime
`--memory-limit` accepts an occupied size (`30G`), free reserve (`-30G`), or
fraction of total device memory (`0.8`). It replaces the exported serving
reserve and includes model resources as well as caches. `ignition run` forwards
both flags without selecting a different prepared package. See
[runtime memory limits](cli-design.md#runtime-memory-limits) for rounding,
minimum capacities, and integrated-memory accounting. Preparation retains its
own existing memory policy.
Ignition captures power-of-two
batch capacities through the concurrency limit
and width one at every bucket. Wider speculative and verification shapes require
`batch * width <= speculative_token_capacity`; the preparation flag
`--speculative-token-capacity` defaults to the larger of padded concurrency and
maximum verification width. Separate proposal catalogs cover every bucket and
load-time draft depth independently of that limit. Flare can reduce
concurrency and draft tokens at load time; prefill capacity stays prepared.
With `--mixed-prefill`, Qwen also captures mixed prompt/decode graphs: every
prompt bucket at width one and one balanced prompt bucket at each wider
verification depth, for each batch capacity of at least two. The ordinary and
draft-state-maintaining catalogs remain distinct. Mixed capture is disabled by
default and independent of the MTP flags.
See [mixed graph coverage](scheduling-runtime-design.md).

## Serving runtime packages

Load an exported `.safetensors` package directly with the native CLI:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp2-mtp5.safetensors \
  --gpus 0,1 --host 127.0.0.1 --port 8001
curl --fail http://127.0.0.1:8001/health
```

Select a subset of a package's captured execution capacities:

```sh
./flare/target/release/flare run /data/flare-runtime/qwen38-tp1-c4-mtp5.safetensors \
  --gpus 2 --max-concurrency 2 --mtp-draft-tokens 3 \
  --no-mtp-adaptive-verification --host 127.0.0.1 --port 8001
```

`--mtp-draft-tokens N` limits each round to N drafts plus its pending input.
`--mtp-draft-tokens 0` selects ordinary prefill, decode, and prefix-cache graphs
without draft execution or draft-state updates. Every speculative package also
contains these ordinary graphs. Adaptive width-one rounds retain draft state so
later rounds can resume speculation. Ignition's `--no-mtp` excludes the draft model
during preparation. `--mtp-adaptive-verification` selects among the loaded
widths; `--no-mtp-adaptive-verification` selects the fixed width where captured,
with width-one execution available at larger batch capacities. Omitted flags
use the package's serving defaults. Limits cannot exceed captured capacity;
missing required batch/width combinations fail before GPU initialization.
Concurrency three requires the batch-four bucket. Only selected graphs and
required kernel functions are loaded. Weights, MTP weights, and transport buffers
retain their prepared layouts. Continuation rows and KV pages occupy separate
fixed virtual ranges with declared maximum extents. Helios backs those ranges
according to selected concurrency, verification width, and available GPU memory;
reducing execution state leaves more physical memory for KV pages.

The selected GPU count must match the package, and each GPU must match its
rank's architecture and memory requirements. Flare serves all ranks in one
process with one execution thread per rank. Its dashboard runs on an
interactive terminal. Use `--api-key-file PATH` to supply a serving API key;
exported packages do not contain the key. Stop Flare with Ctrl-C or SIGTERM
so native replay drains before CUDA resources are released.

## DGX Spark

DGX Spark's GB10 is SM121 and shares physical DRAM with the CPU. Prepare an
SM121 package on Spark; an SM120 package does not satisfy the architecture
check. GPU weights, KV, registered host buffers, resident embedding tables,
compiler workers, and the OS compete for that memory. Set `[memory] reserve`
for host activity as well as driver headroom; `reserve = "4G"` leaves four GiB
outside the model's cache budget. RAM embedding tables load before cache
sizing. Disk embedding tables use bounded staging and can retain their scale
planes in RAM with `--embedding-table-resident-scales`. The loader accounts for
resident scales and the optional row cache before sizing KV, for both PLE and
Engram packages.

Spark's NVML interface does not report VRAM capacity. Ignition identifies an
integrated device through CUDA and uses Linux `MemAvailable`, capped by CUDA's
device total, for planning. Qwen preparation and Helios use the same available
RAM budget when allocating weights and caches; swap is not GPU capacity.
CUDA free-memory queries exclude reclaimable page cache on GB10. Compiler worker
budgets account for both CPU and GPU allocations in shared RAM. Helios sizes
its caches independently after loading.
VMM allocations and registered host mappings remain distinct CUDA
memory types despite sharing DRAM; do not dereference a device allocation on
the CPU or replace system-scope synchronization with volatile accesses. See
[NVIDIA's Spark CUDA guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).

A model too large for one Spark runs tensor-parallel across two or four Sparks
(three for DeepSeek V4.1 Flash),
one rank per host, reducing through b12x RoCEnante over their ConnectX-7 ports:
`ignition export --ranks N --rendezvous ADDR:PORT` on rank zero with
`ignition join` on each peer writes one package on rank zero, and
`flare orbit run NODE:PACKAGE --cluster NAME` serves it from any machine,
starting every rank through its host's `flare agent`. See
[serving across DGX Sparks](multi-host.md) for agent installation, pairing,
and failures.

## Cached and development serving

For preparation followed immediately by serving, use the cached workflow:

```sh
.venv/bin/ignition run qwen3.8-flash --gpus 0,1 --port 8001 --verbose
```

`ignition run` caches packages under `paths.package_cache` (default
`~/.cache/flare/packages`) and replaces its process with Flare via `execv`.
It finds `flare` on `PATH` first, then `flare/target/release/flare`; check for
stale installed executables when testing a local build. Changes to preparation
inputs select a different cache entry. To test a serving-only Rust change,
rebuild Flare and load an existing compatible package directly. Re-export when
changing prepared kernels, capacities, or the package format.

Use `.venv/bin/ignition export --help` and `.venv/bin/ignition run --help` for
preparation flags. See [the runtime package design](runtime-package-design.md)
for resource ownership and [the CLI design](cli-design.md) for configuration.

For model development, `ignition serve` hands prepared graphs directly to Helios
without exporting a package. It requires Heliopause's `qualification` Cargo
feature. Select idle devices explicitly; see [the development launch example](cli-design.md#development-without-package-export)
for the extension build and serving commands.
