/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cost-bounded acquisition of request-local speculative yield observations.

use super::{policy::{Candidate, bits}, utility::LaneValue, *};

impl RequestScheduler {
    pub(super) fn optimistic_tokens(&self, request: &LiveRequest, width: usize) -> f64 {
        f64::from(width != 0) + self.speculative_curve(request, true)[..width.saturating_sub(1)].iter().sum::<f64>()
    }

    pub(super) fn acceptance_probe(&self, ordinary: Candidate,
        values: &[LaneValue; REQUEST_LANES], age_price: f64) -> Option<Candidate> {
        let mut candidates = Vec::new();
        self.packed_round_candidates(&mut candidates, values, age_price, true);
        self.mixed_candidates(&mut candidates, values, age_price, true);
        let count = self.prune_padding(&mut candidates);
        let observed_width = |candidate: Candidate, slot: usize, optimistic: bool| {
            if candidate.mask & (1 << slot) == 0 || candidate.prefill_slot == Some(slot)
                || !matches!(candidate.kind, GraphKind::Speculative | GraphKind::Mixed) {
                1
            } else {
                if let Some(lengths) = candidate.host_lengths {
                    usize::from(lengths[slot])
                } else if candidate.packed_token_capacity != 0 {
                    self.estimated_round_lengths(candidate.mask, candidate.batch, candidate.draft_tokens,
                        candidate.capacity, candidate.packed_token_capacity, optimistic)[slot]
                } else {
                    self.speculative_tokens(self.slots[slot].request.as_ref().unwrap(), candidate.capacity)
                }
            }
        };
        candidates[..count].iter().copied().filter(|candidate| {
            candidate.rate.duration_ns != 0 && candidate.rate.compare(ordinary.rate).is_gt()
                && bits(candidate.mask).any(|slot| {
                    let request = self.slots[slot].request.as_ref().unwrap();
                    let width = observed_width(*candidate, slot, true);
                    width > observed_width(ordinary, slot, false)
                        && request.max_new_tokens - request.generated_tokens > width as u32
                        && self.limits.max_sequence_tokens - request.scheduled_consumed as usize > width
                })
        }).max_by(|a, b| {
            let gain = |candidate: &Candidate| (candidate.rate.score - ordinary.rate.score) / candidate.rate.duration_ns as f64;
            let information = |candidate: &Candidate| bits(candidate.mask).flat_map(|slot| {
                (observed_width(ordinary, slot, false) - 1..observed_width(*candidate, slot, true) - 1)
                    .map(move |depth| self.speculative_information(self.slots[slot].request.as_ref().unwrap(), depth))
            }).fold(f64::INFINITY, f64::min);
            gain(a).total_cmp(&gain(b))
                .then_with(|| information(b).total_cmp(&information(a)))
                .then(b.rate.duration_ns.cmp(&a.rate.duration_ns))
                .then(b.batch.cmp(&a.batch)).then(b.capacity.cmp(&a.capacity)).then(b.mask.cmp(&a.mask))
        })
    }
}
