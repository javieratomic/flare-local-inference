"""Offline gather profiles survive package loading and native startup selection."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.environ.get("FLARE_RUN_TP_CUDA_TESTS") != "1",
                   reason="requires four assigned GPUs and FLARE_RUN_TP_CUDA_TESTS=1")
def test_exported_gather_profiles_and_native_replay(tmp_path):
    import torch

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign four GPUs explicitly"
    assert torch.cuda.device_count() == 4
    root = Path(__file__).resolve().parents[2]
    target = Path(os.environ.get("CARGO_TARGET_DIR", root / "helios/target"))
    subprocess.run(["cargo", "build", "-j", "4", "--manifest-path", str(root / "helios/Cargo.toml"),
        "--no-default-features", "--features", "runtime-package,test-tools", "--bin", "test_fixed_runtime_package"],
        cwd=root, check=True, timeout=600)
    exported = subprocess.run([sys.executable, str(root / "helios/tests/gather_package_fixture.py"), str(tmp_path)],
        cwd=root, text=True, capture_output=True, timeout=600)
    assert exported.returncode == 0, exported.stdout + exported.stderr
    assert "all four preparation ranks exited" in exported.stdout
    replay = subprocess.run([str(target / "debug/test_fixed_runtime_package"), str(tmp_path), "--tp4-collective"],
        cwd=root, text=True, capture_output=True, timeout=180)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert "native TP4 collective package replay passed after Python exit" in replay.stdout
