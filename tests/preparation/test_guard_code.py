# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist guard source while binding each process's closure values."""

import subprocess
import sys

from torch._dynamo import guards

from ignition.preparation.aot import _guard_code_cache


def test_cached_guard_source_binds_fresh_closures_in_another_process(tmp_path):
    parts = ["len(L['values']) == expected", "L['enabled'] == enabled"]
    expected = guards.build_guard_function(parts, 'expected, enabled')
    original = guards.build_guard_function
    with _guard_code_cache(tmp_path):
        assert guards.build_guard_function(parts, 'expected, enabled') == expected
    assert guards.build_guard_function is original

    subprocess.run([sys.executable, '-c', '''
import sys
from pathlib import Path
from torch._dynamo import guards
from ignition.preparation.aot import _guard_code_cache

def forbid_build(*args, **kwargs):
    raise AssertionError('cached guard source must not be regenerated')

guards.build_guard_function = forbid_build
with _guard_code_cache(Path(sys.argv[1])):
    _, source = guards.build_guard_function(
        ["len(L['values']) == expected", "L['enabled'] == enabled"], 'expected, enabled')
scope = {}
exec(source, scope)
for size, enabled in ((3, True), (1, False)):
    check = scope['___make_guard_fn'](size, enabled)
    assert check(dict(values=list(range(size)), enabled=enabled))
    assert not check(dict(values=list(range(size + 1)), enabled=enabled))
    assert not check(dict(values=list(range(size)), enabled=not enabled))
assert guards.build_guard_function is forbid_build
''', str(tmp_path)], check=True)
