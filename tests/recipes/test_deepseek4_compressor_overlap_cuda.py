# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Overlapped compression preserves cache bytes and input lifetimes during native replay."""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.compressed_mla import PreparedCompressedCacheWriter
from ignition.components.compressor_overlap import ConcurrentGatedCompression
from ignition.components.fp8_indexer import _write_keys
from ignition.components.gated_compression import PreparedGatedCompressor
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.components.paged_batch import Metadata
from ignition.components.rotary import PreparedRotaryAngles, _rotate
from ignition.preparation.compile_pool import CompilePool
from ignition.preparation.graphs import PreparedGraphBundle
from ignition.preparation.scratch import GraphBufferPool, ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek4.config import Compressor
from ignition.recipes.deepseek4.pool import ContinuationPool
from .test_deepseek4_pool import model

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an assigned CUDA device")


class CompressionForward:
    def __init__(self, source, seed, overlap, main, index, branch, arena, angles):
        self.source, self.seed, self.overlap, self.main, self.arena = source, seed, overlap, main, arena
        self.index = index
        self.branch = branch
        self.angles = angles

    def __call__(self):
        self.source.copy_(self.seed)
        self.angles()
        self.overlap.start(self.source)
        self.main(self.source)
        self.index.start(self.main.output)
        self.branch(self.main.output)
        self.index.finish(self.main.output, self.branch.output)
        self.overlap.finish(self.source, [self.main.output, self.index.outputs[-1], self.branch.output])
        self.source.fill_(float("nan"))
        self.main.output.fill_(float("nan"))
        self.arena.tensor.fill_(255)
        self.angles.values.fill_(float("nan"))


@pytest.mark.parametrize("requests,inactive", [(1, False), (2, False), (2, True)])
@torch.no_grad()
def test_compression_overlap_retains_inputs_until_native_join(requests, inactive, tmp_path):
    device = torch.device("cuda", 0)
    torch.manual_seed(4621 + requests)
    tokens = requests * 4
    config = model()
    config.text["rms_norm_eps"] = 1e-6
    pool = ContinuationPool.allocate(config, requests + 2, 4, device)
    buffers, arena = GraphBufferPool(device), ScratchArena(device)
    cursor = buffers.cursor()
    metadata = Metadata.allocate(tokens, requests, 2, device)
    metadata.starts.copy_(torch.tensor([127, 255][:requests], device=device))
    metadata.state_rows.copy_(torch.arange(1, requests + 1, device=device))
    metadata.requests.copy_(torch.arange(requests, device=device).repeat_interleave(4))
    metadata.lengths.fill_(4)
    metadata.query_start_loc.copy_(torch.arange(requests + 1, device=device) * 4)
    metadata.positions.copy_((metadata.starts[:, None] + torch.arange(4, device=device)).flatten())
    if inactive:
        metadata.state_rows[-1] = 0
        metadata.positions[-4:] = -1
    emitted = (metadata.positions >= 0) & ((metadata.positions + 1) % 4 == 0)
    physical = metadata.requests + 1
    slots = torch.where(emitted, physical * 64 + (metadata.positions % 256) // 4, -1)
    positions = ((metadata.positions + 1) // 4 * 4 - 4).clamp_min(0)
    frequencies = torch.linspace(.01, 1, 32, device=device)
    angles = PreparedRotaryAngles(positions, frequencies, buffers=cursor)
    source = torch.zeros((tokens, 64), dtype=torch.bfloat16, device=device)
    seed = torch.randn_like(source)
    projected = torch.zeros((tokens, 2560), device=device)
    projection_weight = torch.randn((2560, 64), device=device, dtype=torch.bfloat16) * .1
    head_weights = torch.zeros((tokens, 64), dtype=torch.bfloat16, device=device)
    head_matrix = torch.randn((64, 64), dtype=torch.bfloat16, device=device) * .1
    main_output = torch.empty((tokens, 128), device=device, dtype=torch.bfloat16)
    index_output, branch_output = torch.empty_like(main_output), torch.empty_like(main_output)
    main_matrix = torch.randn((128, 64), device=device, dtype=torch.bfloat16) * .1
    branch_matrix = torch.randn((128, 128), device=device, dtype=torch.bfloat16) * .1
    from b12x.gemm.blockscaled import pack_weight
    index_matrix = pack_weight((torch.randn((128, 128), device=device) * 16).to(torch.float8_e4m3fn),
        torch.full((128, 4), 117, device=device, dtype=torch.uint8), recipe="mxfp8")
    with RecordedPreparationSession(device=device, checkpoint="compression-overlap",
            workload_schema="compression-overlap-v1", numerical_policy="bf16-fp32") as session:
        session.register_scratch(arena)
        projection = UnquantizedProjection(source, projection_weight, projected, session=session, name="projection")
        main = UnquantizedProjection(source, main_matrix, main_output, session=session, name="main")
        index_projection = QuantizedProjection(main_output, index_matrix, index_output,
            session=session, name="index", workspace=arena)
        branch = UnquantizedProjection(main_output, branch_matrix, branch_output,
            session=session, name="branch")
        index_chain = ConcurrentProjectionChain((index_projection,), parallel_plans=(branch.plan,),
            session=session, workspace=arena)
        head = UnquantizedProjection(source, head_matrix, head_weights, session=session, name="head")
        compressors, offset = [], 0
        for index, dim in enumerate((512, 128)):
            spec = Compressor(2, 4, dim, index=bool(index))
            width = spec.state_shape[1]
            weights = {"ape": torch.randn((4, width), device=device),
                       "norm": torch.randn(dim, device=device)}
            compressors.append(PreparedGatedCompressor(spec, weights, source, metadata, pool, slots,
                session=session, name=f"compress{dim}", deferred=True, buffers=cursor,
                emitted=emitted, emitted_slots=slots,
                projected=(projected[:, offset:offset + width],
                           projected[:, offset + width:offset + 2 * width])))
            offset += 2 * width
            compressors[-1].state_values.normal_()
            compressors[-1].state_scores.normal_()
        writer = PreparedCompressedCacheWriter(compressors[0].output, pool.pages["layer2.latent_page"],
            slots, page_size=64, kind="indexed", cache_format="deepseek_v4", session=session, name="writer")
        indexer = SimpleNamespace(cache=pool.pages["layer2.index_page"])
        overlap = ConcurrentGatedCompression(projection, compressors, writer, indexer=indexer,
            index_weight=head, metadata=metadata, positions=positions, frequencies=frequencies,
            angles=angles.values,
            parallel_plans=(main.plan, branch.plan), session=session, workspace=arena)
        session.tune()

        def sequential():
            source.copy_(seed)
            projection(source)
            main(source)
            index_projection(main_output)
            branch(main_output)
            for index, compressor in enumerate(compressors):
                values = compressor(source)
                torch.library.wrap_triton(_rotate)[(tokens, 1)](
                    values, positions, frequencies, values.stride(0), 0, values.shape[1] - 64, 1, None, False)
                if index == 0:
                    writer(values)
                else:
                    torch.library.wrap_triton(_write_keys)[(tokens,)](
                        values, indexer.cache.view(torch.float8_e4m3fn), indexer.cache.view(torch.float32),
                        slots, indexer.cache.stride(0), num_warps=4)
            head(source)
            source.fill_(float("nan"))
            main_output.fill_(float("nan"))

        forward = CompressionForward(source, seed, overlap, main, index_chain, branch, arena, angles)
        path = tmp_path / "compression.bin"
        with CompilePool(device) as compiler:
            compiler.submit(forward, path).result()
        with path.open("rb") as file:
            compiled = torch.compiler.load_compiled_function(file)
        compiled(forward)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled(forward)
        session.freeze()
        bundle = PreparedGraphBundle(device=0, graphs={"compression": graph},
            buffers={"source": source, "seed": seed, "output": main_output},
            resources=(overlap, projection, compressors, writer, head, main, index_projection,
                       index_chain, branch, pool, buffers, arena, session, compiled))
        worker = bundle.start_worker()
        try:
            observed = [projected, head_weights, index_output, branch_output,
                        *(value for c in compressors for value in (c.pooled, c.output)), *overlap.caches]
            for gain in (1., -.5, 2.):
                seed.mul_(gain)
                for cache in overlap.caches:
                    cache.zero_()
                sequential()
                expected = [value.clone() for value in observed]
                for value in observed:
                    value.zero_()
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                ticket = worker.submit(0, repetitions=3)
                completion = worker.poll(timeout_ms=120000)
                assert completion is not None and completion.ticket == ticket and completion.error is None
                after = torch.cuda.memory_stats()
                for key in ("allocation.all.allocated", "allocation.all.freed",
                            "segment.all.allocated", "segment.all.freed"):
                    assert before[key] == after[key]
                for actual, wanted in zip(observed, expected, strict=True):
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                assert source.isnan().all()
                assert main_output.isnan().all()
                assert angles.values.isnan().all()
        finally:
            worker.close()
            graph.reset()
