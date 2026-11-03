# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared b12x embedding lookup with fixed output and device live counts."""

import torch
from b12x.preparation import PreparedCall
from b12x.sequence import embedding

from ignition.preparation.tuning import declare
from .collectives import PreparedChunkedAllGather


class PreparedEmbedding:
    """Gather token rows into caller storage and zero inactive output rows.

    Token IDs and the optional live count remain device inputs during graph
    replay. The fixed b12x kernel has no workspace or tuning candidates.
    """

    def __init__(self, weight, ids, output, *, session, name, num_rows=None):
        self.weight, self.ids, self.output, self.num_rows = weight, ids, output, num_rows
        self.arguments = {"out": output, "num_rows": num_rows}
        self.plan = embedding.plan(embedding.query_from_call(weight, ids, **self.arguments), device=weight.device)
        def prepare(state):
            prime_ids = torch.zeros_like(ids)
            arguments = {**self.arguments, "num_rows": None if num_rows is None else torch.zeros_like(num_rows)}
            return PreparedCall(run=lambda: state.run(weight, prime_ids, **arguments))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))

    def __call__(self):
        if self.num_rows is not None:
            self.output.zero_()
        return embedding.run(self.weight, self.ids, plan=self.plan, **self.arguments)


class PreparedShardedEmbedding:
    """Gather column-sharded vocabulary rows through a prepared collective."""

    def __init__(self, weight, ids, local_output, output, *, runtime, coordinator, session, name, num_rows=None):
        self.lookup = PreparedEmbedding(weight, ids, local_output, session=session, name=name + ".lookup",
                                        num_rows=num_rows)
        self.gather = PreparedChunkedAllGather(local_output, output, runtime=runtime, coordinator=coordinator,
                                               session=session, name=name + ".gather")

    def __call__(self):
        return self.gather(self.lookup())
