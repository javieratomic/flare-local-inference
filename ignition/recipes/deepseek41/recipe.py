"""DeepSeek V4.1 Flash checkpoint and tensor-parallel memory geometry."""

from types import MappingProxyType

from ignition.planning import HEADROOM_BYTES
from ignition.recipes.declaration import Checkpoint, RankCosts, RankUsage, Recipe, SamplingDefaults, Speculation


def prepare(snapshot, plan, *, compile_forwards, kernel_cache, progress, retain_graphs=False,
            package_storage=None, ranks=None):
    from ignition.preparation.rank_group import prepare_group
    from .bootstrap import PreparationConfig, prepare_rank

    speculation = plan.speculation
    drafting = {} if speculation is None else {
        "speculative_widths": speculation.widths, "speculative_token_capacity": plan.speculative_token_capacity,
        "speculative_fixed_width": speculation.fixed_width, "packed_verification": speculation.packed_verification}
    config = PreparationConfig(prompt_logprobs=plan.prompt_logprobs,
        compile_workers=plan.compile_workers, tuning_replica=plan.tuning_replica,
        image_concurrency=plan.image_concurrency,
        devices=tuple(range(plan.world_size)) if ranks is None else (0,) * plan.world_size,
        hosts=None if ranks is None else ranks.hosts, roce_ring=plan.roce_ring, compile_decoder=compile_forwards,
        retain_graphs=retain_graphs, max_active=plan.concurrency, context=plan.context,
        prefill_chunk_size=plan.prefill_chunk, minimum_pages=plan.minimum_pages,
        snapshot_budget_bytes=plan.snapshots * plan.costs.state_row_bytes,
        headroom_bytes=HEADROOM_BYTES + plan.reserve_bytes, kernel_cache=kernel_cache,
        dspark=speculation is not None, split_speculation=speculation is not None and speculation.draft_tokens > 0,
        stream_overlap=plan.stream_overlap, expert_scale_compression=plan.expert_scale_compression,
        mla_compute_mode=plan.mla_compute_mode, **drafting)
    return prepare_group(snapshot, config, prepare_rank, progress=progress, package_storage=package_storage,
                         ranks=ranks, checkpoint=plan.checkpoint)


def usage(prepared):
    return tuple(RankUsage(rank["pages"], rank["free_device_bytes"]) for rank in prepared.memory_report["ranks"])


RECIPE = Recipe(
    name="deepseek41", served_model_name="DeepSeek-V4.1-Flash", prepare=prepare, usage=usage,
    mla_compute_modes=("auto", "bf16"),
    # Recommended sampling: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017#minimal-inference.
    sampling_defaults=SamplingDefaults(temperature=1., top_p=1., top_k=-1),
    checkpoints=(Checkpoint(alias="deepseek-v4.1-flash", repository="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        revision="1bae0e1088993b9b82de08f59de5be5a34e42170", architecture="DeepseekV41ForCausalLM",
        # Measured resident bytes with DSpark and its confidence head: loading each component of
        # TP2 and TP4 ranks on GB10 gives replicated R = 2 * T4 - T2 and total 4 * T4 - 3 * R.
        # CSF replicates its fixed scale-expansion buffers, so its replicated share bounds both.
        stored_bytes=510_286_023_000, device_bytes=310_274_189_312, draft_bytes=7_932_874_632,
        scale_compressed_device_bytes={"csf": 295_683_306_496}, replicated_device_bytes=2_984_102_912),),
    context=1048576, page_tokens=128, concurrency=4, max_concurrency=64, prefill_chunk=4096,
    speculation=Speculation(draft_tokens=5), max_draft_tokens=7, max_images=4, image_bytes=32 << 20,
    # Single-row decode runs the shared expert on an auxiliary stream.
    stream_overlap=True,
    # Every rank keeps complete continuation rows and cache pages; tensor parallelism splits
    # only weights and per-head activations. Three ranks run one rank per host and pad
    # attention to 72 heads and the vocabulary to 129408 rows.
    ranks=MappingProxyType({world: RankCosts(3_043_840, 113_920, 8 << 30) for world in (3, 4)}),
    plain_ranks=MappingProxyType({world: RankCosts(2_841_088, 113_920, 8 << 30) for world in (3, 4)}),
)
