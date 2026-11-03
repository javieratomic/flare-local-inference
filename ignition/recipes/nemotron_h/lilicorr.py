# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate-lattice correlations and predecessor-conditioned draft logits."""

import torch
import torch.nn.functional as F

from ignition.components.embedding import PreparedEmbedding
from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.mlp import PreparedUngatedMLP


def _projection(source, weights, output, *, session, name, workspace):
    weight, bias = weights
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name, bias=bias)
    return QuantizedProjection(source, weight, output, session=session, name=name,
        workspace=workspace, activation_mode="auto", bias=bias)


class _LatticeLayer:
    def __init__(self, weights, source, *, batch, heads, eps, session, name, workspace, buffers):
        self.weights, self.batch, self.heads, self.eps = weights, batch, heads, eps
        rows, hidden = source.shape
        self.normalized = buffers.zeros(source.shape, torch.bfloat16)
        self.projected = buffers.zeros((rows, 3 * hidden), torch.bfloat16)
        self.attended = buffers.zeros(source.shape, torch.bfloat16)
        self.qkv = _projection(self.normalized, weights["qkv"], self.projected,
            session=session, name=name + ".qkv", workspace=workspace)
        self.output = _projection(self.attended, weights["output"], buffers.zeros(source.shape, torch.bfloat16),
            session=session, name=name + ".output", workspace=workspace)
        up, down = weights["mlp"]
        width = up[0].out_features
        self.mlp = PreparedUngatedMLP(self.normalized, up[0], down[0],
            buffers.zeros((rows, width), torch.bfloat16), buffers.zeros((rows, width), torch.bfloat16),
            buffers.zeros(source.shape, torch.bfloat16), activation="silu", up_bias=up[1], down_bias=down[1],
            session=session, name=name + ".mlp", workspace=workspace)

    def __call__(self, source, bias):
        b, h, hidden = self.batch, self.heads, source.shape[-1]
        self.normalized.copy_(F.rms_norm(source, (hidden,), self.weights["attn_norm"], self.eps))
        qkv = self.qkv(self.normalized).view(b, -1, 3, h, hidden // h)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        self.attended.copy_(F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape_as(source))
        residual = source + self.output(self.attended)
        self.normalized.copy_(F.rms_norm(residual, (hidden,), self.weights["mlp_norm"], self.eps))
        return residual + self.mlp(self.normalized)


class PreparedLattice:
    """Compute learned first-slot and adjacent-slot scores from raw head log probabilities."""

    def __init__(self, weights, config, token_embeddings, log_probabilities, pass_hidden, anchor_hidden, *,
                 session, name, workspace, buffers, layer_buffers):
        c = config.text["dflash_config"]
        self.weights, self.eps = weights, config.text["rms_norm_eps"]
        self.vector_eps, self.logit_scale = c["lilicorr_vector_eps"], c["lilicorr_logit_scale"]
        b, slots, k = log_probabilities.shape
        hidden = c["lilicorr_hidden_size"]
        if not 1 <= slots < config.block_size or k != c["lilicorr_candidate_topk"]:
            raise ValueError("LiLiCorr lattice dimensions exceed the trained block or candidate count")
        self.batch, self.slots, self.top_k, self.hidden = b, slots, k, hidden
        rows = b * slots * k
        def matrix(n, width):
            return buffers.zeros((n, width), torch.bfloat16)
        self.states, self.features, self.anchor = matrix(rows, hidden), matrix(rows, 5), matrix(b, hidden)
        self.token = _projection(token_embeddings.view(rows, -1), weights["token_proj"], matrix(rows, hidden),
            session=session, name=name + ".token", workspace=workspace)
        self.passed = _projection(pass_hidden.reshape(b * slots, -1), weights["pass_hidden_proj"], matrix(b * slots, hidden),
            session=session, name=name + ".pass_hidden", workspace=workspace)
        self.context = _projection(anchor_hidden, weights["context_proj"], matrix(b, hidden),
            session=session, name=name + ".context", workspace=workspace)
        up, down = weights["feature_mlp"]
        self.feature_mlp = PreparedUngatedMLP(self.features, up[0], down[0], matrix(rows, hidden), matrix(rows, hidden),
            matrix(rows, hidden), activation="silu", up_bias=up[1], down_bias=down[1],
            session=session, name=name + ".features", workspace=workspace)
        self.layers = tuple(_LatticeLayer(w, self.states, batch=b, heads=c["lilicorr_num_heads"], eps=self.eps,
            session=session, name=f"{name}.layer{i}", workspace=workspace, buffers=layer_buffers.cursor())
            for i, w in enumerate(weights["layers"]))
        factors, bias = weights["factor"]
        self.cross_input = matrix(rows, hidden)
        self.factor_self = _projection(self.states, (factors[0], bias), matrix(rows, hidden),
            session=session, name=name + ".factor_self", workspace=workspace)
        self.factor_anchor = _projection(self.anchor, (factors[1], None), matrix(b, hidden),
            session=session, name=name + ".factor_anchor", workspace=workspace)
        self.factor_cross = _projection(self.cross_input, (factors[2], None), matrix(rows, hidden),
            session=session, name=name + ".factor_cross", workspace=workspace)
        self.factor_hidden = matrix(rows, hidden)
        self.edges = _projection(self.factor_hidden, weights["edges"], matrix(rows, 2 * c["lilicorr_factor_dim"]),
            session=session, name=name + ".edges", workspace=workspace)
        self.anchor_out = _projection(self.anchor, weights["anchor_out_head"], matrix(b, c["lilicorr_factor_dim"]),
            session=session, name=name + ".anchor_out", workspace=workspace)
        rank = torch.arange(k, dtype=torch.float32, device=self.states.device)
        self.rank_fraction, self.is_top1 = rank / (k - 1), (rank == 0).float()
        slot_ids = torch.arange(slots, device=self.states.device).repeat_interleave(k)
        relative = slot_ids[:, None] - slot_ids[None, :]
        bias = weights["relative_slot_bias"][:, relative + config.block_size - 1]
        self.bias = (bias + (relative == 0).to(bias.dtype)[None] * weights["same_slot_bias"][:, None, None])[None].contiguous()

    def __call__(self, token_embeddings, log_probabilities, pass_hidden, anchor_hidden, anchor_valid):
        b, s, k, h = self.batch, self.slots, self.top_k, self.hidden
        logp = log_probabilities.float()
        features = torch.stack((logp, logp.exp(), logp - logp.amax(-1, keepdim=True),
            self.rank_fraction.expand_as(logp), self.is_top1.expand_as(logp)), dim=-1).bfloat16()
        weight, bias = self.weights["feature_norm"]
        self.features.copy_(F.layer_norm(features, (5,), weight, bias, 1e-5).view(-1, 5))
        token = self.token(token_embeddings.view(b * s * k, -1)).view(b, s, k, h)
        passed = self.passed(pass_hidden.reshape(b * s, -1)).view(b, s, 1, h)
        states = token + passed
        states = states + self.feature_mlp(self.features).view(b, s, k, h)
        states = states + self.weights["slot_embedding"][:, 0, :s]
        states = states + self.weights["rank_embedding"][:, 0]
        self.states.copy_(states.flatten(0, 2))
        for layer in self.layers:
            self.states.copy_(layer(self.states, self.bias))
        self.states.copy_(F.rms_norm(self.states, (h,), self.weights["output_norm"], self.eps))
        anchor = self.context(anchor_hidden) * anchor_valid[:, None].to(anchor_hidden.dtype)
        self.anchor.copy_(F.rms_norm(anchor, (h,), self.weights["anchor_norm"], self.eps))
        x = self.states.view(b, s, k, h)
        self.cross_input.copy_((x * self.anchor[:, None, None]).flatten(0, 2))
        pre = self.factor_self(self.states).view(b, s, k, h) + self.factor_anchor(self.anchor)[:, None, None]
        pre = pre + self.factor_cross(self.cross_input).view(b, s, k, h)
        self.factor_hidden.copy_(F.silu(pre).flatten(0, 2))
        edges = self.edges(self.factor_hidden).view(b, s, k, 2, -1)
        out_vector, in_vector = F.normalize(edges, dim=-1, eps=self.vector_eps).unbind(-2)
        anchor_out = F.normalize(self.anchor_out(self.anchor), dim=-1, eps=self.vector_eps)
        first = (anchor_out[:, None] * in_vector[:, 0]).sum(-1).float() * self.logit_scale
        pairs = torch.matmul(out_vector[:, :-1], in_vector[:, 1:].transpose(-1, -2)).float() * self.logit_scale
        return torch.cat((first[:, None, None].expand(-1, 1, k, -1), pairs), dim=1)


class PreparedLatticeSelector:
    """Restrict each proposal to the lattice row selected by its realized predecessor."""

    def __init__(self, weights, config, embedding, hidden, logits, anchor, *, session, name, workspace, buffers, layer_buffers):
        b, block, width = hidden.shape
        self.top_k, self.slots = config.text["dflash_config"]["lilicorr_candidate_topk"], block - 1
        self.ids = buffers.zeros((b, self.slots, self.top_k), torch.int64)
        self.logp = buffers.zeros(self.ids.shape, torch.float32)
        self.pass_hidden = buffers.zeros((b, self.slots, width), torch.bfloat16)
        self.embedded = buffers.zeros((b, self.slots, self.top_k, width), torch.bfloat16)
        self.lookup = PreparedEmbedding(embedding, self.ids.flatten(), self.embedded.view(-1, width),
            session=session, name=name + ".embedding")
        self.lattice = PreparedLattice(weights, config, self.embedded, self.logp, self.pass_hidden, anchor,
            session=session, name=name + ".lattice", workspace=workspace, buffers=buffers, layer_buffers=layer_buffers)
        self.scores = buffers.zeros((b, self.slots, self.top_k, self.top_k), torch.float32)
        self.rows = torch.arange(b, device=hidden.device)

    def prepare(self, hidden, logits, anchor, anchor_valid):
        values = logits[:, 1:].float()
        unary, ids = values.topk(self.top_k, dim=-1, sorted=True)
        self.ids.copy_(ids)
        self.logp.copy_(unary - values.logsumexp(-1, keepdim=True))
        self.pass_hidden.copy_(hidden[:, 1:])
        self.lookup()
        self.scores.copy_(self.lattice(self.embedded, self.logp, self.pass_hidden, anchor, anchor_valid))

    def __call__(self, logits, predecessor, position):
        step = position - 1
        previous = (self.ids[:, step - 1] == predecessor[:, None]).int().argmax(-1) if step else torch.zeros_like(predecessor)
        selected = self.scores[self.rows, step, previous]
        logits.copy_(torch.full_like(logits, -torch.inf).scatter(1, self.ids[:, step], selected))
