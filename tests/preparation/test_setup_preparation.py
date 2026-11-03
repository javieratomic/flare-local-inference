# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preparation setup installs locked dependencies with the native build policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("setup_preparation", ROOT / "scripts/setup_preparation.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def test_setup_runs_locked_sync(monkeypatch):
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert setup.main() == 0
    assert len(calls) == 2
    command = calls[0][0][0]
    assert command[:3] == ["uv", "sync", "--locked"]
    assert calls[0][1]["env"]["BUILD_NVEP"] == "0"
    assert calls[1][0][0] == ["uv", "run", "--no-sync", "python", "-m", "ignition.patches"]
    assert calls[1][1] == {"cwd": ROOT, "check": True}


@pytest.mark.parametrize("error", [OSError("uv unavailable"), subprocess.CalledProcessError(1, "uv")])
@pytest.mark.parametrize("failure_call", [1, 2])
def test_failed_sync_or_patch_reports_error(monkeypatch, capsys, error, failure_call):
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise error

    monkeypatch.setattr(setup.subprocess, "run", fail)
    assert setup.main() == 1
    assert calls == failure_call
    assert str(error) in capsys.readouterr().err
