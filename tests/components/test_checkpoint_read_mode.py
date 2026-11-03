# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""RAM-backed checkpoint shards use a loader that supports their filesystem."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ignition.components import weights


@pytest.mark.parametrize('devices, expected', [([(8, 1)], 'default'), ([(0, 42)], 'bounce'),
                                             ([(8, 1), (0, 42)], 'bounce')])
@pytest.mark.parametrize('fallback', ('auto', 'bounce'))
@pytest.mark.skipif(os.name != 'posix', reason='Linux mount device identities require POSIX device numbers')
def test_reader_uses_resolved_filesystems_for_mixed_checkpoint_shards(monkeypatch, devices, expected, fallback):
    monkeypatch.setattr(weights, 'sys', SimpleNamespace(platform='linux'))
    monkeypatch.setattr(weights, '_read_mode', lambda: fallback)
    monkeypatch.setattr(Path, 'read_text', lambda path:
        '10 1 8:1 / / rw - ext4 /dev/nvme0n1 rw\n'
        '11 1 0:42 / /dev/shm rw - tmpfs tmpfs rw\n')
    paths = [SimpleNamespace(stat=lambda device=os.makedev(*device): SimpleNamespace(st_dev=device))
             for device in devices]
    assert weights._checkpoint_read_mode(paths) == (fallback if expected == 'default' else expected)


@pytest.mark.skipif(os.environ.get('FLARE_RUN_CUDA_TESTS') != '1', reason='requires an assigned CUDA GPU')
@pytest.mark.parametrize('dtype', ('bfloat16', 'uint8', 'float32'))
def test_tmpfs_checkpoint_preserves_strided_tensor_bytes(dtype):
    import tempfile
    from types import MappingProxyType

    import torch
    from safetensors.torch import save_file

    from ignition.preparation.checkpoint import IndexedCheckpoint, _read_header

    if not Path('/dev/shm').is_dir():
        pytest.skip('requires Linux shared memory')
    with tempfile.TemporaryDirectory(prefix='flare-checkpoint-', dir='/dev/shm') as directory:
        root = Path(directory)
        source = torch.arange(17 * 65, dtype=torch.float32).reshape(17, 65).to(getattr(torch, dtype))
        save_file({'weight': source}, str(root / 'weights.safetensors'))
        shard, tensors = _read_header(root / 'weights.safetensors', 'weights.safetensors')
        manifest = IndexedCheckpoint(root, MappingProxyType(tensors), (shard,))
        with weights.DirectWeightLoader(manifest, 'cuda:0') as loader:
            actual = loader.load_tensor('weight', extent=(slice(2, 13), slice(3, 62)))
            assert loader._session.read_mode == 'bounce'
            assert loader._session.torch_copy_bytes == 0
        torch.testing.assert_close(actual.cpu(), source[2:13, 3:62], rtol=0, atol=0)
