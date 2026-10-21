"""Rank zero admits each peer rank once, from its own host, running rank zero's sources."""

import threading

import pytest

from ignition.preparation import host_ranks
from ignition.preparation.host_ranks import HostRendezvous, knock


def topology(boot):
    return {"host": boot, "boot_id": boot, "rdma": [],
            "gpu": {"name": "NVIDIA GB10", "uuid": f"GPU-{boot}", "total_bytes": 1 << 37,
                    "free_bytes": 1 << 36, "capability": [12, 1]}}


def rendezvous(world=2):
    owner = HostRendezvous("127.0.0.1:0", world, identity="same", topology=topology("zero"))
    return owner, f"127.0.0.1:{owner.listener.getsockname()[1]}"


def join(address, rank, world=2, *, identity="same", boot=None, results=None):
    try:
        sockets = knock(address, rank, world, identity=identity, topology=topology(boot or f"peer{rank}"),
                        timeout=10)
        if results is not None:
            results.append(sockets)
    except RuntimeError as error:
        if results is not None:
            results.append(error)


@pytest.mark.parametrize("world", [3, 4])
def test_admits_every_peer_rank_with_its_executor_connection(world):
    owner, address = rendezvous(world)
    results = []
    peers = [threading.Thread(target=join, args=(address, rank, world), kwargs={"results": results})
             for rank in reversed(range(1, world))]
    for peer in peers:
        peer.start()
    topologies = owner.accept(timeout=10)
    for peer in peers:
        peer.join()
    expected = ("zero", *(f"peer{rank}" for rank in range(1, world)))
    assert tuple(item["boot_id"] for item in topologies) == expected
    assert owner.hosts == expected
    assert all(isinstance(result, tuple) for result in results)
    group = host_ranks.HostRankGroup.__new__(host_ranks.HostRankGroup)
    group.rendezvous = owner
    assert len(group.connections) == len(group.controls) == world - 1


@pytest.mark.parametrize("world", [0, 1, 17, 2.0, True])
def test_rendezvous_rejects_rank_counts_that_helios_cannot_load(world):
    with pytest.raises(ValueError, match="two to sixteen"):
        HostRendezvous("127.0.0.1:0", world, identity="same", topology=topology("zero"))


@pytest.mark.parametrize("arguments, reason", [
    ({"identity": "other"}, "source identity"),
    ({"world": 4}, "world size"),
])
def test_rejects_peers_that_differ_from_rank_zero(arguments, reason):
    owner, address = rendezvous()
    results = []
    peer = threading.Thread(target=join, args=(address, 1), kwargs={**arguments, "results": results})
    peer.start()
    with pytest.raises(TimeoutError):
        owner.accept(timeout=2)
    peer.join()
    assert isinstance(results[0], RuntimeError) and reason in str(results[0])


def test_rejects_a_second_process_for_an_admitted_rank():
    owner, address = rendezvous(4)
    failures = []

    def admit():
        try:
            owner.accept(timeout=3)
        except TimeoutError as error:
            failures.append(error)

    accepting = threading.Thread(target=admit)
    accepting.start()
    results = []
    join(address, 1, 4, results=results)
    join(address, 1, 4, results=results)
    accepting.join()
    assert isinstance(results[0], tuple) and failures
    assert isinstance(results[1], RuntimeError) and "already joined" in str(results[1])


def test_requires_one_rank_per_host():
    owner, address = rendezvous()
    peer = threading.Thread(target=join, args=(address, 1), kwargs={"boot": "zero"})
    peer.start()
    with pytest.raises(ValueError, match="different host"):
        owner.accept(timeout=10)
    peer.join()


def test_finds_the_interface_holding_an_address():
    assert host_ranks.interface_of("127.0.0.1") == "lo"
    with pytest.raises(RuntimeError):
        host_ranks.interface_of("203.0.113.7")


def test_multi_host_preparation_requires_a_hub_checkpoint(tmp_path):
    from types import SimpleNamespace

    owner, _ = rendezvous()
    owner.listener.close()
    with pytest.raises(ValueError, match="hub cache"):
        owner.prepare(tmp_path / "local-model", None, None,
                      checkpoint=SimpleNamespace(repository="org/model", revision="rev"))


def test_local_checkpoint_source_must_match_its_declared_directory(tmp_path):
    from types import SimpleNamespace

    owner, _ = rendezvous()
    owner.listener.close()
    with pytest.raises(ValueError, match="declared source"):
        owner.prepare(tmp_path / "source", None, None,
                      checkpoint=SimpleNamespace(repository=str(tmp_path / "other"), revision="path-id"))


def test_export_manifest_transfers_only_absent_files(tmp_path, monkeypatch):
    import socket
    from pathlib import Path
    from ignition.preparation.package_file import write_sources, read_sources
    from ignition.preparation.package_snapshot import BlobStore, ExportOptions
    from ignition.preparation.runtime_package_pb2 import ModelIndex, BundleIndex, Span

    peer = BlobStore(tmp_path / 'peer-cache')
    name = peer.put(8, lambda: [b'weights!'])
    weights = peer.root / name
    code = tmp_path / 'code.bin'
    code.write_bytes(b'kernel')
    index = ModelIndex(version=4, ranks=[BundleIndex(resources={
        'weights': Span(offset=0, nbytes=8), 'kernel': Span(offset=4096, nbytes=6)})])
    manifest = tmp_path / 'rank.json'
    write_sources(manifest, index, [(0, weights, 4096, 8), (4096, code, 0, 6)])
    settings = ExportOptions(tmp_path / 'coordinator')
    requests = []
    send = host_ranks.send_object

    def observe(sock, message):
        if isinstance(message, list):
            requests.append(message)
        send(sock, message)

    monkeypatch.setattr(host_ranks, 'send_object', observe)
    for iteration in range(2):
        directory = tmp_path / f'received-{iteration}'
        directory.mkdir()
        failures = []
        receiver, sender = socket.socketpair()
        receiver.settimeout(5)
        sender.settimeout(5)
        def transmit():
            try:
                host_ranks.send_export_sources(sender, manifest, {})
            except Exception as error:
                failures.append(error)
        thread = threading.Thread(target=transmit)
        thread.start()
        try:
            description = host_ranks.receive_object(receiver)['sources']
            host_ranks.receive_export_sources(receiver, description, directory / 'rank.json',
                ExportOptions(settings.cache, no_weights=iteration == 1))
            _, locate = read_sources(directory / 'rank.json')
            location, offset, count = locate(index.ranks[0].resources['weights'])[0]
            assert Path(location).read_bytes()[offset:offset + count] == b'weights!'
        finally:
            thread.join(5)
            receiver.close()
            sender.close()
        assert not failures and not thread.is_alive()
    assert requests[0] == [str(weights), str(code)]
    assert requests[1] == [str(code)]


def test_export_transfers_more_sources_than_the_descriptor_limit(tmp_path):
    import socket
    import subprocess
    import sys
    from pathlib import Path
    from ignition.preparation.package_file import write_sources, read_sources
    from ignition.preparation.package_snapshot import ExportOptions
    from ignition.preparation.runtime_package_pb2 import ModelIndex, BundleIndex, Span

    pytest.importorskip('resource')
    payload, resources = [], {}
    for item in range(256):
        source = tmp_path / f'global-{item}.bin'
        source.write_bytes(item.to_bytes(4, 'little'))
        payload.append((item * 4096, source, 0, 4))
        resources[str(item)] = Span(offset=item * 4096, nbytes=4)
    manifest = tmp_path / 'rank.json'
    index = ModelIndex(version=4, ranks=[BundleIndex(resources=resources)])
    write_sources(manifest, index, payload)
    destination = tmp_path / 'received.json'
    receiver, sender = socket.socketpair()
    receiver.settimeout(10)
    code = """
import resource, socket, sys
from ignition.preparation.host_ranks import send_export_sources
resource.setrlimit(resource.RLIMIT_NOFILE, (64, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
with socket.socket(fileno=int(sys.argv[1])) as control:
    control.settimeout(10)
    send_export_sources(control, sys.argv[2], {})
"""
    child = subprocess.Popen([sys.executable, '-c', code, str(sender.fileno()), str(manifest)],
                             pass_fds=(sender.fileno(),), stderr=subprocess.PIPE)
    sender.close()
    try:
        description = host_ranks.receive_object(receiver)['sources']
        host_ranks.receive_export_sources(receiver, description, destination, ExportOptions(tmp_path / 'cache'))
        _, locate = read_sources(destination)
        for item, span in index.ranks[0].resources.items():
            path, offset, size = locate(span)[0]
            assert Path(path).read_bytes()[offset:offset + size] == int(item).to_bytes(4, 'little')
        _, errors = child.communicate(timeout=10)
        assert child.returncode == 0, errors.decode()
    finally:
        receiver.close()
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


@pytest.mark.parametrize('change', ['replace', 'truncate'])
def test_export_rejects_source_changes_before_transfer(tmp_path, change):
    import socket
    from ignition.preparation.package_file import write_sources
    from ignition.preparation.runtime_package_pb2 import ModelIndex, BundleIndex, Span

    source = tmp_path / 'global.bin'
    source.write_bytes(b'original')
    manifest = tmp_path / 'rank.json'
    index = ModelIndex(version=4, ranks=[BundleIndex(resources={'global': Span(nbytes=8)})])
    write_sources(manifest, index, [(0, source, 0, 8)])
    receiver, sender = socket.socketpair()
    receiver.settimeout(5)
    sender.settimeout(5)
    failures = []
    def transmit():
        try:
            host_ranks.send_export_sources(sender, manifest, {})
        except Exception as error:
            failures.append(error)
    thread = threading.Thread(target=transmit)
    thread.start()
    try:
        host_ranks.receive_object(receiver)
        if change == 'replace':
            replacement = tmp_path / 'replacement'
            replacement.write_bytes(b'original')
            replacement.replace(source)
        else:
            source.write_bytes(b'short')
        host_ranks.send_object(receiver, [str(source)])
        thread.join(5)
        assert not thread.is_alive()
        assert len(failures) == 1
        assert isinstance(failures[0], ValueError)
        assert 'changed before transfer' in str(failures[0])
    finally:
        receiver.close()
        sender.close()
        thread.join(5)


def test_cluster_export_combines_received_manifests_into_signed_snapshot(tmp_path, monkeypatch):
    import socket
    from types import SimpleNamespace
    from helios.tests.signing import publisher
    from ignition.preparation import runtime_package, memory_inventory
    from ignition.preparation.package_file import write_sources, read_index, PackageFiles
    from ignition.preparation.package_snapshot import BlobStore, read_at
    from ignition.preparation.runtime_package_pb2 import ModelIndex, BundleIndex, Span

    def sources(rank, path):
        store = BlobStore(tmp_path / f'cache-{rank}')
        contents = f'weights-{rank}'.encode()
        blob = store.put(len(contents), lambda: [contents])
        code = path.parent / f'code-{rank}.bin'
        code.write_bytes(b'kernel')
        bundle = BundleIndex(resources={'weights': Span(nbytes=len(contents)), 'kernel': Span(offset=4096, nbytes=6)})
        bundle.allocations.add(name='weights', file='weights')
        bundle.modules.add(name='kernel', file='kernel')
        bundle.roce.world, bundle.roce.rank = 2, rank
        write_sources(path, ModelIndex(version=4, ranks=[bundle]),
                      [(0, store.root / blob, 4096, len(contents)), (4096, code, 0, 6)])
        return SimpleNamespace(memory_inventory={})

    monkeypatch.setattr(runtime_package, 'export_model', lambda model, images, storage, path, **kwargs: sources(0, path))
    monkeypatch.setattr(memory_inventory, 'print_inventory', lambda *args, **kwargs: None)
    peer_path = tmp_path / 'peer.json'
    sources(1, peer_path)
    receiver, sender = socket.socketpair()
    receiver.settimeout(5)
    sender.settimeout(5)
    errors = []
    def peer():
        try:
            assert host_ranks.receive_object(sender) == ('export', {})
            host_ranks.send_export_sources(sender, peer_path, {})
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=peer)
    thread.start()
    group = host_ranks.HostRankGroup.__new__(host_ranks.HostRankGroup)
    group._started = group._closed = False
    group.checkpoint_ring = None
    group.model = None
    group.rendezvous = SimpleNamespace(world=2, peers={1: SimpleNamespace(control=receiver)})
    output = tmp_path / 'model.safetensors'
    try:
        group.export(output, options={}, export_id='snapshot-cluster', images=None, storage=None, publisher=publisher())
    finally:
        thread.join(5)
        receiver.close()
        sender.close()
    assert not errors and not thread.is_alive()
    index, start = read_index(output)
    assert index.cluster.world == 2 and index.cluster.export_id == 'snapshot-cluster'
    with output.open('rb') as primary, PackageFiles(output, primary, start, 0) as files:
        for rank in range(2):
            span = index.ranks[rank].resources['weights']
            source, base, _ = files.get(span.filename)
            assert read_at(source, span.nbytes, base + span.offset) == f'weights-{rank}'.encode()
