# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Share serial candidate fixtures without retaining them after a tuning race."""

from weakref import ref


def candidate_inputs(factory):
    """Acquire one fixture whose lifetime belongs to the live candidate calls.

    b12x runs each candidate's producer and invocation in stream order. Calls
    can share mutable operands when the producer restores every input and the
    kernel fully writes its output. Each call must retain the returned owner.
    """
    reference = None

    def acquire():
        nonlocal reference
        value = None if reference is None else reference()
        if value is None:
            value = factory()
            reference = ref(value)
        return value

    return acquire
