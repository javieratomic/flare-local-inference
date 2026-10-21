"""Join tensor-parallel ranks on separate hosts through rank zero's rendezvous.

Each host starts its own rank process. Rank zero listens on an address every
peer can reach, such as a DGX Spark's RoCE fabric address. A peer knocks on
that rendezvous with its rank, the world size, its source identity, and its
host topology. Rank zero admits every rank once and serves authenticated
checkpoint metadata and exact routed byte ranges while each peer prepares its
own shard. Ranks form a Gloo group through a TCP store on rank zero, which
collectives use to exchange their RDMA endpoints.

Every peer holds two TCP connections to rank zero. The control connection
carries the handshake, the preparation assignment, memory reports and the serve
command, and stays open for the group's lifetime: losing it ends the process on
the other side, because a rank cannot complete collectives without its peers.
The executor connection carries Helios's per-step coordination, or rank zero's
steps under Ignition's diagnostic direct execution (`ignition serve --direct`).
"""

import fcntl
import hashlib
import json
import os
import pickle
import secrets
import select
import shutil
import socket
import struct
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from .source_identity import source_identity

PROTOCOL = 4
# Peers wait here while rank zero compiles the complete serving catalog.
_PREPARATION_TIMEOUT = timedelta(hours=2)
_HANDSHAKE_SECONDS = 30
_MAX_MESSAGE = 1 << 30
_SHUTDOWN_SECONDS = 120
_JSON, _PICKLE = b"J", b"P"
_COPY_BLOCK = 16 << 20
_EXPORT_PUBLICATION_LOCK = threading.Lock()


def _send(sock, kind, payload):
    sock.sendall(kind + struct.pack("<Q", len(payload)) + payload)


def _receive_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(min(count - len(data), 1 << 20))
        if not chunk:
            raise ConnectionError("rank rendezvous connection closed")
        data += chunk
    return bytes(data)


def _receive(sock, kind):
    header = _receive_exact(sock, 9)
    size = struct.unpack("<Q", header[1:])[0]
    if header[:1] != kind or size > _MAX_MESSAGE:
        raise ConnectionError("unexpected rank rendezvous message")
    return _receive_exact(sock, size)


def send_json(sock, value):
    _send(sock, _JSON, json.dumps(value, sort_keys=True).encode())


def receive_json(sock):
    return json.loads(_receive(sock, _JSON))


def send_object(sock, value):
    _send(sock, _PICKLE, pickle.dumps(value))


def receive_object(sock):
    """Unpickle a message from rank zero, which every peer was told to trust."""
    return pickle.loads(_receive(sock, _PICKLE))


def _export_workers(function, items, connections):
    """Abort every stream before joining workers when one export transfer fails."""
    from concurrent.futures import ThreadPoolExecutor
    import sys

    failure, lock = [], threading.Lock()

    def run(item):
        try:
            return function(item)
        except BaseException as error:
            with lock:
                if not failure:
                    failure.append(error)
                    # Closing control sockets can make the rank lifetime watcher
                    # exit this process before the caller reports the exception.
                    print(traceback.format_exc(), file=sys.stderr, flush=True)
                    for connection in connections:
                        try:
                            connection.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
            raise

    with ThreadPoolExecutor(max_workers=len(items), thread_name_prefix='flare-export') as pool:
        try:
            return list(pool.map(run, items))
        except BaseException:
            # A lower-numbered worker may wake with a secondary disconnect first.
            if failure:
                raise failure[0]
            raise


def send_export_sources(control, path, inventory, *, channels=None):
    """Send a rank manifest, then the content files requested by the coordinator."""
    from .checkpoint_source import BLOCK_BYTES, file_identity

    document = json.loads(Path(path).read_text())
    identities, metadata = {}, {}
    # A rank can export thousands of cubins. Hold only one descriptor per worker,
    # and reject replacement or mutation between the metadata scan and transfer.
    for _, name, _, _ in document['payload']:
        if name in metadata:
            continue
        with Path(name).open('rb') as source:
            identity = file_identity(source.fileno())
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
            if file_identity(source.fileno()) != identity:
                raise ValueError('rank export source changed during metadata scan')
        identities[name] = identity
        metadata[name] = {'size': identity[2], 'sha256': digest,
                          'blob': Path(name).name == digest + '.safetensors'}
    send_object(control, {'sources': {'document': document, 'files': metadata}, 'memory_inventory': inventory})
    requested = receive_object(control)
    if (not isinstance(requested, list) or any(not isinstance(n, str) for n in requested) or
            len(requested) != len(set(requested)) or any(n not in metadata for n in requested)):
        raise ValueError('invalid requested rank export sources')

    def transmit(stream):
        channel = None if channels is None else channels[stream]
        names = requested if channel is None else requested[stream::len(channels)]
        total = 0
        for name in names:
            with Path(name).open('rb') as source:
                if file_identity(source.fileno()) != identities[name]:
                    raise ValueError('rank export source changed before transfer')
                size = metadata[name]['size']
                if channel is None:
                    if control.sendfile(source, count=size) != size:
                        raise ConnectionError('rank export source ended before its declared size')
                else:
                    for offset in range(0, size, BLOCK_BYTES):
                        block = source.read(min(BLOCK_BYTES, size - offset))
                        if len(block) != min(BLOCK_BYTES, size - offset):
                            raise ConnectionError('rank export source ended before its declared size')
                        channel.send('export-data', block, name=name, offset=offset)
                if file_identity(source.fileno()) != identities[name]:
                    raise ValueError('rank export source changed during transfer')
                total += size
        if channel is not None:
            stats = dict(files=len(names), payload_bytes=total)
            channel.send('export-finished', stats=stats)
            message, payload = channel.receive('export-finished')
            if payload or message.get('stats') != stats:
                raise ValueError('rank export completion accounting differs')

    if channels is None:
        transmit(0)
    else:
        _export_workers(transmit, list(range(len(channels))), [c.connection for c in channels])


def receive_export_sources(control, description, destination, settings, *, channels=None):
    """Reuse verified local blobs and receive only absent files into private storage."""
    import tempfile
    from .package_snapshot import BlobStore
    from .package_file import read_sources
    from .checkpoint_source import BLOCK_BYTES

    document, metadata = description['document'], description['files']
    store = BlobStore(settings.cache)
    targets, requested = {}, []
    for name, info in metadata.items():
        digest, size = info['sha256'], info['size']
        if type(info['blob']) is not bool:
            raise ValueError('invalid rank payload storage kind')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid rank payload digest')
        if type(size) is not int or not 0 < size <= 64 << 30:
            raise ValueError('rank payload exceeds its size limit')
        target = (store.root / (digest + '.safetensors') if info['blob'] else
                  destination.parent / (digest + '.source'))
        targets[name] = target
        # Another rank may publish the same content. Wait until its private
        # hard link is removed, since unlink changes the inode's ctime.
        with _EXPORT_PUBLICATION_LOCK:
            exists = target.exists()
        if exists:
            store.verify(target, bytes.fromhex(digest), size)
        else:
            if info['blob'] and settings.no_weights:
                raise ValueError('--no-weights requires the peer stored-data blob in the coordinator cache')
            requested.append(name)
    send_object(control, requested)
    started = time.monotonic()

    def receive_file(name, channel):
        info, target = metadata[name], targets[name]
        descriptor, temporary = tempfile.mkstemp(prefix='.rank-blob-', dir=target.parent)
        partial = Path(temporary)
        try:
            digest = hashlib.sha256()
            with os.fdopen(descriptor, 'wb') as output:
                remaining = info['size']
                while remaining:
                    if channel is None:
                        block = _receive_exact(control, min(_COPY_BLOCK, remaining))
                    else:
                        message, block = channel.receive('export-data')
                        if (message.get('name') != name or message.get('offset') != info['size'] - remaining or
                                len(block) != min(BLOCK_BYTES, remaining)):
                            raise ValueError('rank export payload range differs')
                    output.write(block)
                    digest.update(block)
                    remaining -= len(block)
                output.flush()
                os.fsync(output.fileno())
                if info['blob']:
                    os.fchmod(output.fileno(), 0o444)
            if digest.hexdigest() != info['sha256']:
                raise ValueError('rank payload digest mismatch')
            with _EXPORT_PUBLICATION_LOCK:
                try:
                    os.link(partial, target)
                    published = True
                except FileExistsError:
                    published = False
                finally:
                    partial.unlink(missing_ok=True)
            if not published:
                store.verify(target, bytes.fromhex(info['sha256']), info['size'])
        finally:
            partial.unlink(missing_ok=True)

    def receive_stream(stream):
        channel = None if channels is None else channels[stream]
        names = requested if channel is None else requested[stream::len(channels)]
        for name in names:
            receive_file(name, channel)
        if channel is not None:
            stats = dict(files=len(names), payload_bytes=sum(metadata[n]['size'] for n in names))
            message, payload = channel.receive('export-finished')
            if payload or message.get('stats') != stats:
                raise ValueError('rank export completion accounting differs')
            channel.send('export-finished', stats=stats)

    if channels is None:
        receive_stream(0)
    else:
        _export_workers(receive_stream, list(range(len(channels))), [c.connection for c in channels])
    for item in document['payload']:
        item[1] = str(targets[item[1]])
    destination.write_text(json.dumps(document, separators=(',', ':')))
    read_sources(destination)
    seconds = time.monotonic() - started
    size = sum(metadata[n]['size'] for n in requested)
    return dict(payload_bytes=size, payload_seconds=seconds, payload_gbps=size * 8 / seconds / 1e9,
                files=len(requested), streams=1 if channels is None else len(channels),
                transport='rendezvous-tcp' if channels is None else 'agent-ring-tcp')


def _rdma_devices():
    devices = []
    for device in sorted(Path("/sys/class/infiniband").glob("*")):
        try:
            if "ACTIVE" in (device / "ports" / "1" / "state").read_text():
                devices.append(device.name)
        except OSError:
            continue
    return devices


def local_topology(gpu):
    """What a rank reports about its host: identity, its GPU, and its active RDMA devices."""
    return {"host": socket.gethostname(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "gpu": {"name": gpu.name, "uuid": gpu.uuid, "total_bytes": gpu.total_bytes,
                    "free_bytes": gpu.free_bytes, "capability": list(gpu.capability)},
            "rdma": _rdma_devices()}


def topology_gpu(topology, index):
    from ignition.planning import Gpu

    gpu = topology["gpu"]
    return Gpu(index, gpu["name"], gpu["uuid"], gpu["total_bytes"], gpu["free_bytes"],
               tuple(gpu["capability"]))


def interface_of(address):
    """Name of the local interface that holds an IPv4 address, for Gloo's device selection."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _, name in socket.if_nameindex():
            try:
                packed = fcntl.ioctl(probe.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))  # SIOCGIFADDR
            except OSError:
                continue
            if socket.inet_ntoa(packed[20:24]) == address:
                return name
    finally:
        probe.close()
    raise RuntimeError(f"no local interface holds {address}")


def _use_interface(address):
    # Gloo binds its sockets to the interface that reaches rank zero.
    os.environ["GLOO_SOCKET_IFNAME"] = interface_of(address)


def parse_address(text):
    host, colon, port = text.rpartition(":")
    if not colon or not port.isdecimal():
        raise ValueError(f"rank rendezvous address must be IPv4:PORT; got {text!r}")
    socket.inet_aton(host)
    return host, int(port)


def _watch_until_closed(sockets, finished, reason, *, cleanup=None):
    """Exit this process when another rank's connection closes before coordinated shutdown."""
    def watch():
        poller = select.poll()
        for sock in sockets:
            poller.register(sock.fileno(), select.POLLRDHUP | select.POLLHUP | select.POLLERR)
        while True:
            events = poller.poll(200)
            if finished.is_set():
                return
            if events:
                print(f"flare: {reason}", flush=True)
                if cleanup is not None:
                    cleanup()
                os._exit(1)

    thread = threading.Thread(target=watch, name="flare-rank-lifetime", daemon=True)
    thread.start()
    return thread


@dataclass
class _Peer:
    rank: int
    topology: dict
    control: socket.socket
    executor: socket.socket | None = None


class HostRendezvous:
    """Rank zero's listener: admits one process per peer rank, each on its own host."""

    def __init__(self, address, world, *, identity, topology, checkpoint_token_file="~/.config/flare/agent.token",
                 checkpoint_ring=None):
        # Helios loads multi-host packages of two through sixteen ranks; recipes
        # declare which of those tensor-parallel sizes they prepare.
        if type(world) is not int or not 2 <= world <= 16:
            raise ValueError("multi-host tensor parallelism requires two to sixteen ranks")
        self.address, self.world = parse_address(address), world
        self.identity, self.topology = identity, topology
        self.checkpoint_token_file = checkpoint_token_file
        self.checkpoint_ring = checkpoint_ring
        self.session = secrets.token_hex(16)
        self.listener = socket.create_server(self.address, reuse_port=False)
        self.peers = {}
        self.store = None
        # Set once coordinated shutdown begins: peers then exit and close their connections.
        self.stopping = threading.Event()

    @property
    def hosts(self):
        """Boot identity of every rank's host, in rank order."""
        return tuple(topology["boot_id"] for topology in self.topologies)

    @property
    def topologies(self):
        return (self.topology, *(self.peers[rank].topology for rank in range(1, self.world)))

    def accept(self, *, timeout=None, report=None):
        """Wait until every peer rank has knocked and opened its executor connection."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while len(self.peers) < self.world - 1 or any(peer.executor is None for peer in self.peers.values()):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                missing = [rank for rank in range(1, self.world)
                           if rank not in self.peers or self.peers[rank].executor is None]
                raise TimeoutError(f"tensor-parallel ranks {missing} did not join {self.address[0]}:{self.address[1]}")
            self.listener.settimeout(remaining)
            try:
                sock, origin = self.listener.accept()
            except TimeoutError:
                continue
            try:
                self._admit(sock, origin, report)
            except (OSError, ValueError, ConnectionError) as error:
                if report is not None:
                    report(f"rejected rank connection from {origin[0]}: {error}")
                sock.close()
        self.listener.close()
        boots = self.hosts
        if len(set(boots)) != len(boots):
            raise ValueError("every multi-host tensor-parallel rank must run on a different host")
        return self.topologies

    def _admit(self, sock, origin, report):
        sock.settimeout(_HANDSHAKE_SECONDS)
        hello = receive_json(sock)
        rank = hello.get("rank")

        def reject(reason):
            send_json(sock, {"accepted": False, "reason": reason})
            raise ValueError(reason)

        if hello.get("protocol") != PROTOCOL:
            reject(f"rank protocol {hello.get('protocol')} differs from {PROTOCOL}")
        if type(rank) is not int or not 1 <= rank < self.world:
            reject(f"rank {rank!r} is outside 1..{self.world - 1}")
        if hello.get("world") != self.world:
            reject(f"world size {hello.get('world')} differs from rank zero's {self.world}")
        if hello.get("channel") == "control":
            if hello.get("identity") != self.identity:
                reject("source identity differs from rank zero's; every host must run the same checkout")
            if rank in self.peers:
                reject(f"rank {rank} has already joined")
            self.peers[rank] = _Peer(rank, hello["topology"], sock)
            send_json(sock, {"accepted": True, "session": self.session})
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if report is not None:
                report(f"rank {rank} joined from {hello['topology']['host']} ({origin[0]})")
        elif hello.get("channel") == "executor":
            peer = self.peers.get(rank)
            if hello.get("session") != self.session or peer is None or peer.executor is not None:
                reject(f"rank {rank} executor connection does not match an admitted control connection")
            peer.executor = sock
            send_json(sock, {"accepted": True})
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        else:
            reject(f"unknown rank channel {hello.get('channel')!r}")

    def prepare(self, snapshot, config, prepare, *, checkpoint, progress=None, package_storage=None):
        """Prepare every rank before returning the root-owned group; any failure ends every rank.

        With `package_storage`, every rank prepares into its own package storage for export.
        """
        import torch.distributed as dist

        from ignition.progress import phase

        if dist.is_initialized():
            raise RuntimeError("rank preparation requires exclusive ownership of the control group")
        snapshot = Path(snapshot)
        local = Path(checkpoint.repository)
        if local.is_absolute():
            if snapshot.resolve() != local.resolve():
                raise ValueError(f"local checkpoint directory differs from its declared source: {snapshot}")
        elif snapshot.parent.name != "snapshots" or snapshot.name != checkpoint.revision:
            raise ValueError("multi-host preparation sends peers a Hugging Face checkpoint from the hub cache, "
                             f"not the directory {snapshot}")
        controls = [self.peers[rank].control for rank in range(1, self.world)]
        _watch_until_closed(controls, self.stopping, "a tensor-parallel peer disconnected before shutdown")
        from .checkpoint_source import RootSource, SourceServer, token

        source = RootSource(snapshot, checkpoint)
        readers = []
        ring = None
        try:
            key = token(self.checkpoint_token_file)
            with phase(progress, "Admit checkpoint range readers"):
                if self.checkpoint_ring is not None:
                    from .checkpoint_ring import RingSource

                    ring = RingSource(source, key, self.identity, self.topologies, **self.checkpoint_ring)
                    if progress is not None:
                        progress(f"checkpoint ranges: agent ring TCP, {ring.streams} streams per peer")
                for rank, control in enumerate(controls, 1):
                    send_object(control, ("checkpoint-source", checkpoint))
                    reader = SourceServer(source, control, key, rank, self.identity,
                                          transport=None if ring is None else ring.grant)
                    readers.append(reader)
                    reader.thread.start()
                for reader in readers:
                    reader.wait_ready()
            with phase(progress, "Connect tensor-parallel ranks"):
                _use_interface(self.address[0])
                # Peers connect only after receiving the store's port, so rank zero does not wait for them here.
                self.store = dist.TCPStore(self.address[0], 0, self.world, True, timeout=_PREPARATION_TIMEOUT,
                                           wait_for_workers=False)
                assignment = {"store": (self.address[0], self.store.port), "config": config, "prepare": prepare,
                              "packaged": package_storage is not None}
                for control in controls:
                    send_object(control, assignment)
                dist.init_process_group("gloo", store=self.store, rank=0, world_size=self.world,
                                        timeout=_PREPARATION_TIMEOUT)
            model = prepare(snapshot, config, rank=0, group=dist.group.WORLD, progress=progress,
                            **({} if package_storage is None else {"package_storage": package_storage}))
            model.manifest()
            # Loading owns each socket's reader until its authenticated end marker is acknowledged.
            for reader in readers:
                reader.join()
            if ring is not None:
                ring.join()
            with phase(progress, "Wait for tensor-parallel ranks", total=len(controls)) as stage:
                reports = []
                for control in controls:
                    message = receive_object(control)
                    if "report" not in message:
                        raise RuntimeError(f"peer preparation failed: {message}")
                    reports.append(message["report"])
                    stage.update(advance=1)
            group = HostRankGroup(self, model, reports,
                                  checkpoint_ring=ring if package_storage is not None else None)
            if package_storage is not None:
                ring = None  # The group owns the authenticated channels through export.
            return group
        except BaseException:  # noqa: BLE001 - Preparation interruption must end every rank.
            traceback.print_exc()
            print("flare: terminating tensor-parallel rank group: preparation failed", flush=True)
            os._exit(1)
        finally:
            if ring is not None:
                ring.close()
            source.close()

    def finish(self):
        """Wait for every peer to report its exit, then release the rendezvous."""
        import torch.distributed as dist

        self.stopping.set()
        deadline = time.monotonic() + _SHUTDOWN_SECONDS
        failures = []
        for rank in range(1, self.world):
            peer = self.peers[rank]
            peer.control.settimeout(max(1.0, deadline - time.monotonic()))
            try:
                message = receive_object(peer.control)
                if message.get("exit") != 0:
                    failures.append(f"rank {rank}: {message}")
            except (OSError, ConnectionError, EOFError) as error:
                failures.append(f"rank {rank}: {error}")
        for peer in self.peers.values():
            for sock in (peer.control, peer.executor):
                if sock is not None:
                    sock.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        self.store = None
        if failures:
            raise RuntimeError(f"tensor-parallel peers did not exit cleanly: {failures}")


class HostRankGroup:
    """Root model and its remote peers through native coordinated shutdown."""

    def __init__(self, rendezvous, model, reports, *, checkpoint_ring=None):
        self.rendezvous, self.model = rendezvous, model
        self.memory_report = {"world_size": rendezvous.world, "ranks": [model.memory_report, *reports]}
        self._started = self._closed = False
        self.checkpoint_ring = checkpoint_ring

    def _close_checkpoint_ring(self):
        if self.checkpoint_ring is not None:
            self.checkpoint_ring.close()
            self.checkpoint_ring = None

    @property
    def controls(self):
        return [self.rendezvous.peers[rank].control for rank in range(1, self.rendezvous.world)]

    @property
    def connections(self):
        return [self.rendezvous.peers[rank].executor for rank in range(1, self.rendezvous.world)]

    def manifest(self):
        return self.model.manifest()

    def export(self, path, *, options, export_id, images, storage, publisher, progress=None, shard_size=64 << 30):
        """Combine rank manifests, receiving only content absent from the coordinator cache."""
        import tempfile
        from .package_snapshot import _OPTIONS, ExportOptions, export_options
        from .package_file import combine
        from .runtime_package import export_model
        from .memory_inventory import print_inventory

        if self._started or self._closed:
            raise RuntimeError("prepared rank group has already been consumed or closed")
        if publisher is None:
            raise ValueError('final model publication requires a signing publisher')
        settings = _OPTIONS.get() or ExportOptions(Path(path).parent / '.flare-cache')
        for control in self.controls:
            send_object(control, ("export", options))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.rank-sources-', dir=Path(path).parent) as temporary:
            stage = Path(temporary)
            roots = [stage / f'rank-{rank}.json' for rank in range(self.rendezvous.world)]
            with export_options(settings.cache, no_weights=settings.no_weights):
                package = export_model(self.model, images, storage, roots[0], options=options,
                                       progress=progress, sources=True)
                def receive_rank(rank):
                    control = self.controls[rank - 1]
                    reply = receive_object(control)
                    if 'sources' not in reply:
                        raise RuntimeError(f'rank {rank} export failed: {reply}')
                    channels = (None if self.checkpoint_ring is None else
                                [self.checkpoint_ring.readers[rank, stream].channel
                                 for stream in range(self.checkpoint_ring.streams)])
                    report = receive_export_sources(control, reply['sources'], roots[rank], settings, channels=channels)
                    if progress is not None:
                        progress(f"received rank {rank} export: {report['payload_bytes'] / (1 << 30):.2f} GiB "
                                 f"via {report['transport']} ({report['streams']} streams, "
                                 f"{report['payload_seconds']:.2f}s, {report['payload_gbps']:.2f} Gbit/s)")
                    return reply['memory_inventory']

                connections = self.controls + ([] if self.checkpoint_ring is None else
                                               [r.connection for r in self.checkpoint_ring.readers.values()])
                try:
                    inventories = [package.memory_inventory, *_export_workers(
                        receive_rank, list(range(1, self.rendezvous.world)), connections)]
                finally:
                    self._close_checkpoint_ring()
                combine(path, roots, [], publisher=publisher, shard_size=shard_size,
                        cluster=(self.rendezvous.world, export_id))
            print_inventory(inventories, console=getattr(progress, 'console', None))

    def start_server(self, *, bind="127.0.0.1:8000", template_defaults=None, options=None):
        if self._started or self._closed:
            raise RuntimeError("prepared rank group has already been consumed or closed")
        try:
            self._close_checkpoint_ring()
            for control in self.controls:
                send_object(control, "serve")
            self._started = True
            return self.model.start_server(bind=bind, template_defaults=template_defaults,
                                           peers=self.connections, options=options)
        except BaseException:  # noqa: BLE001 - Startup failure can strand peer collectives.
            traceback.print_exc()
            print("flare: terminating tensor-parallel rank group: native startup failed", flush=True)
            os._exit(1)

    def start_direct(self, *, eager):
        """Hand every peer to Ignition's diagnostic step loop; returns the executor connections that carry its steps.

        Rank zero then sends each step it decides over these connections, and
        every rank executes the same steps in order. Rank zero sends the loop's
        stop message before closing the group.
        """
        if self._started or self._closed:
            raise RuntimeError("prepared rank group has already been consumed or closed")
        self._close_checkpoint_ring()
        for control in self.controls:
            send_object(control, ("direct", {"eager": eager}))
        self._started = True
        return self.connections

    def close(self):
        if self._closed:
            return
        self.rendezvous.stopping.set()
        try:
            self._close_checkpoint_ring()
            if not self._started:
                for control in self.controls:
                    send_object(control, "close")
            self.model.close()
            self.rendezvous.finish()
            self._closed = True
        except BaseException:  # noqa: BLE001 - Failed drain requires process-level reclamation.
            traceback.print_exc()
            print("flare: terminating tensor-parallel rank group: cleanup could not confirm drain", flush=True)
            os._exit(1)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()


def knock(address, rank, world, *, identity, topology, timeout=600.0, report=None):
    """Join rank zero's rendezvous, retrying until it listens; returns the control and executor sockets."""
    host, port = parse_address(address)
    deadline = time.monotonic() + timeout

    def connect(hello):
        while True:
            try:
                sock = socket.create_connection((host, port), timeout=_HANDSHAKE_SECONDS)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_json(sock, hello)
        reply = receive_json(sock)
        if not reply.get("accepted"):
            sock.close()
            raise RuntimeError(f"rank zero rejected rank {rank}: {reply.get('reason')}")
        sock.settimeout(None)
        return sock, reply

    control, reply = connect({"protocol": PROTOCOL, "channel": "control", "rank": rank, "world": world,
                              "identity": identity, "topology": topology})
    executor, _ = connect({"protocol": PROTOCOL, "channel": "executor", "rank": rank, "world": world,
                           "session": reply["session"]})
    if report is not None:
        report(f"joined rank zero at {host}:{port} as rank {rank} of {world}")
    return control, executor


def serve_peer(control, executor, rank, world, *, hub_cache, progress=None, export_staging=None,
               package_cache=None,
               checkpoint_token_file="~/.config/flare/agent.token", identity=None):
    """Prepare one peer from owned checkpoint metadata and requested source ranges, report
    memory, and serve until coordinated drain.

    Any failure exits the process; rank zero then ends the group.
    """
    import signal

    finished = threading.Event()
    owned_sources = []
    _watch_until_closed([control], finished, "rank zero disconnected before shutdown",
                        cleanup=lambda: [source.close() for source in owned_sources])
    # Rank zero coordinates shutdown; a terminal interrupt must not end a peer mid-drain.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    source = source_scope = None
    try:
        from .checkpoint_source import RemoteSource, token

        command = receive_object(control)
        if not (isinstance(command, tuple) and command[0] == "checkpoint-source"):
            raise RuntimeError(f"rank zero sent {command!r} before its checkpoint")
        _, checkpoint = command
        source = RemoteSource(control, token(checkpoint_token_file), rank,
                              source_identity() if identity is None else identity, checkpoint,
                              directory=export_staging, on_created=owned_sources.append)
        source_scope = source.activate()
        snapshot = source_scope.__enter__()
        assignment = receive_object(control)
        config, prepare = assignment["config"], assignment["prepare"]
        store_host, store_port = assignment["store"]
        _use_interface(control.getsockname()[0])
        import torch.distributed as dist

        store = dist.TCPStore(store_host, store_port, world, False, timeout=_PREPARATION_TIMEOUT)
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world, timeout=_PREPARATION_TIMEOUT)
        recorder = storage = None
        if assignment.get("packaged"):
            import torch

            from .module_images import ModuleImages
            from .package_storage import PackageStorage

            recorder = ModuleImages()
            with recorder:
                torch.cuda.set_device(0)
                storage = PackageStorage(rank=rank, world=world, device=0)
                with storage:
                    model = prepare(snapshot, config, rank=rank, group=dist.group.WORLD, progress=progress,
                                    package_storage=storage)
        else:
            model = prepare(snapshot, config, rank=rank, group=dist.group.WORLD, progress=progress)
        source.finish(keep_transport=bool(assignment.get("packaged")))
        model.memory_report["checkpoint_source"] = source.report()
        if progress is not None:
            report = source.report()
            progress(f"received {source.stats['payload_bytes'] / (1 << 30):.2f} GiB of checkpoint ranges "
                     f"via {report['transport']} ({source.streams} streams, "
                     f"{report['payload_seconds']:.2f}s, {report['payload_gbps']:.2f} Gbit/s)")
        model.manifest()
        send_object(control, {"report": model.memory_report})
        command = receive_object(control)
        if isinstance(command, tuple) and command[0] == "export":
            import tempfile

            from .runtime_package import export_model

            _, options = command
            if export_staging is not None:
                Path(export_staging).mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="flare-rank-", dir=export_staging))
            try:
                from .package_snapshot import export_options
                path = directory / f"rank-{rank}.json"
                if progress is not None:
                    progress(f"exporting rank {rank}")
                # The peer retains its own prepared blobs; rank zero requests only missing files.
                with export_options(package_cache or Path.home() / '.cache/flare/packages'):
                    package = export_model(model, recorder, storage, path, options=options,
                                           bundle_only=True, sources=True)
                send_export_sources(control, path, package.memory_inventory,
                                    channels=[channel for channel, _ in source.channel_stats] or None)
                source.close_transport()
            finally:
                shutil.rmtree(directory, ignore_errors=True)
            command = receive_object(control)
        source.close_transport()
        if command == "serve":
            if progress is not None:
                progress("serving")
            model.serve_rank(executor, rank=rank, world=world)
            if progress is not None:
                progress("drained")
        elif isinstance(command, tuple) and command[0] == "direct":
            from ignition.direct.executor import serve_peer as serve_direct

            if progress is not None:
                progress("executing rank zero's direct diagnostic steps")
            serve_direct(model, executor, **command[1])
            if progress is not None:
                progress("drained")
        elif command != "close":
            raise RuntimeError("invalid prepared rank handoff")
        executor.close()
        model.close()
        if storage is not None:
            for region in storage.host_regions:
                region.close()
        dist.destroy_process_group()
        source_scope.__exit__(None, None, None)
        source_scope = None
        source.close()
        finished.set()
        send_object(control, {"exit": 0})
        control.close()
    except BaseException:  # noqa: BLE001 - Never unwind CUDA resources around a stranded collective.
        traceback.print_exc()
        if source_scope is not None:
            source_scope.__exit__(None, None, None)
        if source is not None:
            source.close()
        try:
            send_object(control, {"exit": 1, "error": traceback.format_exc(limit=3)})
        except OSError:
            pass
        os._exit(1)
