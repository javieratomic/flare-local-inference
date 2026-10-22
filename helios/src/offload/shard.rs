//! Disk records of a model whose ranks run as separate processes on separate hosts.
//!
//! Each host keeps the record files of its own rank in its own record directory. A file holds
//! one rank's planes and row in the ordinary record format, so every host's files have the same
//! names and lengths. Rank zero's [`DiskStore`](super::disk_store::DiskStore) indexes, sizes,
//! evicts, and plans every record; [`Sharded`] runs each of its transport jobs on rank zero's own
//! [`Staged`] transport and on every peer's, through a dedicated connection to each peer, and
//! reports the job only after every rank finished it. A fetch therefore completes only once every
//! rank's GPU holds its pages and row. A store that some rank failed removes the file from the
//! ranks that wrote it before the failure is reported, so no host keeps a record the index lacks.
//!
//! Before serving, the ranks agree on the records they scanned: rank zero keeps a record only
//! when every rank holds a file with the same header and keeps its row only when every rank
//! holds the row. Every rank then deletes or truncates its files to match, so the index rank zero
//! builds describes every host's directory.

use std::{
    collections::HashMap,
    io::{ErrorKind, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use blake2::{Blake2b, Digest, digest::consts::U16};

use crate::{
    Error, Result,
    cache::{PageId, PrefixKey},
    execution::RankStream,
    scheduler::StateSlotId,
};

use super::{
    InboundTicket, JobId, Prepared, PublishedCheckpoint, Shared, Transport,
    disk::DiskThread,
    disk_store::{DiskTransport, FetchSegment, Staged, TransportEvent},
    geometry::invalid,
    record::{self, HEADER_BYTES, Header, Scanned},
    transfer::{self, Transfers},
};

/// Peers connect to rank zero's record listener within this time.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(60);
/// The scan agreement and the final close are answered within this time.
const REPLY_TIMEOUT: Duration = Duration::from_secs(120);
/// Jobs ids at or above this belong to [`Sharded`] itself, not to the disk store.
const INTERNAL_JOBS: u64 = 1 << 63;
/// A frame larger than this is a protocol error; a fetch of a long chain stays far below it.
const FRAME_LIMIT: usize = 64 << 20;
const IDLE: Duration = Duration::from_millis(1);
const BUSY: Duration = Duration::from_micros(100);

const HELLO: u8 = 1;
const SUMMARY: u8 = 2;
const AGREED: u8 = 3;
const STORE: u8 = 4;
const FETCH: u8 = 5;
const TRUNCATE: u8 = 6;
const UNLINK: u8 = 7;
const CLOSE: u8 = 8;
const EVENT: u8 = 9;
const CLOSED: u8 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Stored,
    Fetched,
    Truncated,
    Unlinked,
}

impl Kind {
    fn code(self) -> u8 {
        self as u8
    }

    fn decode(code: u8) -> Result<Self> {
        Ok(match code {
            0 => Self::Stored,
            1 => Self::Fetched,
            2 => Self::Truncated,
            3 => Self::Unlinked,
            _ => return Err(invalid("unknown prefix record event kind")),
        })
    }

    fn of(event: &TransportEvent) -> (Self, JobId, Result<u64>) {
        match event {
            TransportEvent::Stored { id, result } => (Self::Stored, *id, copy(result)),
            TransportEvent::Fetched { id, result } => (Self::Fetched, *id, copy(result)),
            TransportEvent::Truncated { id, result } => {
                (Self::Truncated, *id, copy(result).map(|()| 0))
            }
            TransportEvent::Unlinked { id, result } => {
                (Self::Unlinked, *id, copy(result).map(|()| 0))
            }
        }
    }

    fn event(self, id: JobId, result: Result<u64>) -> TransportEvent {
        match self {
            Self::Stored => TransportEvent::Stored { id, result },
            Self::Fetched => TransportEvent::Fetched { id, result },
            Self::Truncated => TransportEvent::Truncated {
                id,
                result: result.map(drop),
            },
            Self::Unlinked => TransportEvent::Unlinked {
                id,
                result: result.map(drop),
            },
        }
    }
}

fn copy<T: Copy>(result: &Result<T>) -> Result<T> {
    match result {
        Ok(value) => Ok(*value),
        Err(error) => Err(error.clone()),
    }
}

// ---------------------------------------------------------------------------------------------
// Framing

struct Writer(Vec<u8>);

impl Writer {
    fn new(tag: u8) -> Self {
        Self(vec![tag, 0, 0, 0, 0])
    }

    fn u8(&mut self, value: u8) -> &mut Self {
        self.0.push(value);
        self
    }

    fn u32(&mut self, value: u32) -> &mut Self {
        self.0.extend(value.to_le_bytes());
        self
    }

    fn u64(&mut self, value: u64) -> &mut Self {
        self.0.extend(value.to_le_bytes());
        self
    }

    fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.0.extend(value);
        self
    }

    fn text(&mut self, value: &str) -> &mut Self {
        self.u32(value.len() as u32).bytes(value.as_bytes())
    }

    fn key(&mut self, key: &PrefixKey) -> &mut Self {
        self.u64(key.consumed_tokens).bytes(&key.digest)
    }

    fn header(&mut self, header: &Header) -> Result<&mut Self> {
        Ok(self.bytes(&header.encode()?))
    }

    fn finish(mut self) -> Vec<u8> {
        let length = (self.0.len() - 5) as u32;
        self.0[1..5].copy_from_slice(&length.to_le_bytes());
        self.0
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.0.len() < count {
            return Err(invalid("truncated prefix record message"));
        }
        let (head, rest) = self.0.split_at(count);
        self.0 = rest;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn text(&mut self) -> Result<&'a str> {
        let length = self.u32()? as usize;
        std::str::from_utf8(self.take(length)?)
            .map_err(|_| invalid("prefix record message text is not UTF-8"))
    }

    fn key(&mut self) -> Result<PrefixKey> {
        Ok(PrefixKey {
            consumed_tokens: self.u64()?,
            digest: self.take(16)?.try_into().unwrap(),
        })
    }

    fn header(&mut self) -> Result<Header> {
        Header::decode(self.take(HEADER_BYTES)?)
    }

    /// A count of items that each occupy at least `item` bytes of what remains.
    fn count(&mut self, item: usize) -> Result<usize> {
        let count = self.u32()? as usize;
        if count.saturating_mul(item) > self.0.len() {
            return Err(invalid("prefix record message count exceeds its payload"));
        }
        Ok(count)
    }

    fn end(&self) -> Result<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(invalid("prefix record message has trailing bytes"))
        }
    }
}

/// One end of a record connection. Reads never block; a write waits for socket space.
struct Link {
    stream: TcpStream,
    received: Vec<u8>,
    /// Set once the connection failed or the other end closed it.
    broken: Option<Error>,
}

impl Link {
    fn new(stream: TcpStream) -> Result<Self> {
        stream
            .set_nodelay(true)
            .and_then(|()| stream.set_nonblocking(true))
            .map_err(|error| record::io_failure("configure a prefix record connection", error))?;
        Ok(Self {
            stream,
            received: Vec::new(),
            broken: None,
        })
    }

    fn fail(&mut self, error: Error) -> Error {
        self.broken.get_or_insert(error).clone()
    }

    fn send(&mut self, frame: &[u8]) -> Result<()> {
        if let Some(error) = &self.broken {
            return Err(error.clone());
        }
        let mut sent = 0;
        let deadline = Instant::now() + REPLY_TIMEOUT;
        while sent < frame.len() {
            match self.stream.write(&frame[sent..]) {
                Ok(0) => {
                    return Err(self.fail(invalid("prefix record connection closed")));
                }
                Ok(count) => sent += count,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err(self.fail(invalid("prefix record connection stalled")));
                    }
                    std::thread::sleep(BUSY);
                }
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => {
                    return Err(self.fail(record::io_failure(
                        "send on a prefix record connection",
                        error,
                    )));
                }
            }
        }
        Ok(())
    }

    /// The next complete frame, if one has arrived.
    fn receive(&mut self) -> Result<Option<(u8, Vec<u8>)>> {
        if let Some(error) = &self.broken {
            return Err(error.clone());
        }
        let mut buffer = [0u8; 64 << 10];
        loop {
            if self.received.len() >= 5 {
                let length = u32::from_le_bytes(self.received[1..5].try_into().unwrap()) as usize;
                if length > FRAME_LIMIT {
                    return Err(self.fail(invalid("prefix record frame exceeds its limit")));
                }
                if self.received.len() >= 5 + length {
                    let tag = self.received[0];
                    let payload = self.received[5..5 + length].to_vec();
                    self.received.drain(..5 + length);
                    return Ok(Some((tag, payload)));
                }
            }
            match self.stream.read(&mut buffer) {
                Ok(0) => return Err(self.fail(invalid("prefix record connection closed"))),
                Ok(count) => self.received.extend_from_slice(&buffer[..count]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(None),
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) => {
                    return Err(self.fail(record::io_failure(
                        "receive on a prefix record connection",
                        error,
                    )));
                }
            }
        }
    }

    /// Wait up to `timeout` for a frame tagged `tag`.
    fn expect(&mut self, tag: u8, timeout: Duration) -> Result<Vec<u8>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some((received, payload)) = self.receive()? {
                if received != tag {
                    return Err(self.fail(invalid("unexpected prefix record message")));
                }
                return Ok(payload);
            }
            if Instant::now() >= deadline {
                return Err(self.fail(invalid("prefix record peer did not answer in time")));
            }
            std::thread::sleep(IDLE);
        }
    }
}

/// A record file name as rank zero sends it: a plain record or temporary name, never a path.
fn file(directory: &Path, name: &str) -> Result<PathBuf> {
    let plain = name
        .strip_prefix('.')
        .and_then(|n| n.strip_suffix(".tmp"))
        .unwrap_or(name);
    if record::parse_file_name(plain).is_none() {
        return Err(invalid(
            "prefix record message names an invalid record file",
        ));
    }
    Ok(directory.join(name))
}

fn name(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("prefix record path has no file name"))
}

// ---------------------------------------------------------------------------------------------
// Rank zero's transport

/// One job's completion on every rank: index zero is rank zero.
struct Pending {
    kind: Kind,
    results: Vec<Option<Result<u64>>>,
    /// A store's record file, removed from the ranks that wrote it when another rank failed.
    path: Option<PathBuf>,
    /// The store a cleanup unlink belongs to, with the failure it reports.
    cleanup: Option<(JobId, Error)>,
}

/// Rank zero's transport for a model whose ranks run as separate processes: rank zero's own
/// staged transport plus one connection per peer rank, ranks 1 onward.
pub struct Sharded {
    local: Staged,
    peers: Vec<Link>,
    pending: HashMap<(Kind, JobId), Pending>,
    events: Vec<TransportEvent>,
    next_internal: u64,
    closed: bool,
}

impl Sharded {
    fn new(local: Staged, peers: Vec<TcpStream>) -> Result<Self> {
        Ok(Self {
            local,
            peers: peers.into_iter().map(Link::new).collect::<Result<_>>()?,
            pending: HashMap::new(),
            events: Vec::new(),
            next_internal: INTERNAL_JOBS,
            closed: false,
        })
    }

    fn ranks(&self) -> usize {
        self.peers.len() + 1
    }

    /// Begin tracking a job rank zero accepted, and send it to every peer.
    fn begin(&mut self, kind: Kind, id: JobId, path: Option<PathBuf>, frame: &[u8]) {
        let mut results = vec![None; self.ranks()];
        for (index, peer) in self.peers.iter_mut().enumerate() {
            if let Err(error) = peer.send(frame) {
                results[index + 1] = Some(Err(error));
            }
        }
        self.pending.insert(
            (kind, id),
            Pending {
                kind,
                results,
                path,
                cleanup: None,
            },
        );
    }

    fn record(&mut self, kind: Kind, id: JobId, rank: usize, result: Result<u64>) {
        let Some(pending) = self.pending.get_mut(&(kind, id)) else {
            return;
        };
        if pending.results[rank].is_none() {
            pending.results[rank] = Some(result);
        }
        if pending.results.iter().all(Option::is_some) {
            let pending = self.pending.remove(&(kind, id)).unwrap();
            self.finish(id, pending);
        }
    }

    fn finish(&mut self, id: JobId, pending: Pending) {
        let failure = pending
            .results
            .iter()
            .enumerate()
            .find_map(|(rank, result)| match result {
                Some(Err(error)) => Some(invalid(format!("rank {rank}: {error}"))),
                _ => None,
            });
        if let Some((store, error)) = pending.cleanup {
            self.events.push(TransportEvent::Stored {
                id: store,
                result: Err(error),
            });
            return;
        }
        let Some(error) = failure else {
            let result = pending.results[0].as_ref().map_or(Ok(0), copy);
            self.events.push(pending.kind.event(id, result));
            return;
        };
        let wrote: Vec<usize> = pending
            .results
            .iter()
            .enumerate()
            .filter(|(_, result)| matches!(result, Some(Ok(_))))
            .map(|(rank, _)| rank)
            .collect();
        match (pending.kind, pending.path) {
            (Kind::Stored, Some(path)) if !wrote.is_empty() => self.remove(&wrote, path, id, error),
            (kind, _) => self.events.push(kind.event(id, Err(error))),
        }
    }

    /// Unlink a failed store's file on the ranks that wrote it, then report the failure.
    fn remove(&mut self, ranks: &[usize], path: PathBuf, store: JobId, error: Error) {
        let frame = match name(&path) {
            Ok(name) => {
                let mut writer = Writer::new(UNLINK);
                writer.u64(self.next_internal).text(name);
                writer.finish()
            }
            Err(_) => {
                self.events.push(TransportEvent::Stored {
                    id: store,
                    result: Err(error),
                });
                return;
            }
        };
        let id = JobId(self.next_internal);
        self.next_internal += 1;
        let mut results: Vec<Option<Result<u64>>> = vec![Some(Ok(0)); self.ranks()];
        for &rank in ranks {
            let sent = if rank == 0 {
                self.local.unlink(id, path.clone())
            } else {
                self.peers[rank - 1].send(&frame)
            };
            results[rank] = sent.err().map(Err);
        }
        let pending = Pending {
            kind: Kind::Unlinked,
            results,
            path: None,
            cleanup: Some((store, error)),
        };
        if pending.results.iter().all(Option::is_some) {
            self.finish(id, pending);
        } else {
            self.pending.insert((Kind::Unlinked, id), pending);
        }
    }

    fn absorb_local(&mut self) {
        let mut events = Vec::new();
        self.local.poll(&mut events);
        for event in events {
            let (kind, id, result) = Kind::of(&event);
            self.record(kind, id, 0, result);
        }
    }

    fn absorb_peers(&mut self) {
        for index in 0..self.peers.len() {
            let rank = index + 1;
            loop {
                let received = self.peers[index].receive();
                match received {
                    Ok(None) => break,
                    Ok(Some((EVENT, payload))) => match decode_event(&payload) {
                        Ok((kind, id, result)) => self.record(kind, id, rank, result),
                        Err(error) => {
                            self.peers[index].fail(error);
                        }
                    },
                    Ok(Some(_)) => {
                        self.peers[index].fail(invalid("unexpected prefix record message"));
                    }
                    Err(error) => {
                        self.abandon(rank, error);
                        break;
                    }
                }
            }
        }
    }

    /// Fail every job still waiting on a rank whose connection broke.
    fn abandon(&mut self, rank: usize, error: Error) {
        let waiting: Vec<(Kind, JobId)> = self
            .pending
            .iter()
            .filter(|(_, pending)| pending.results[rank].is_none())
            .map(|(key, _)| *key)
            .collect();
        for (kind, id) in waiting {
            self.record(kind, id, rank, Err(error.clone()));
        }
    }

    fn accept(&self) -> Result<()> {
        if self.closed {
            return Err(invalid("the sharded record transport is closed"));
        }
        Ok(())
    }
}

fn encode_event(kind: Kind, id: JobId, result: &Result<u64>) -> Vec<u8> {
    let mut writer = Writer::new(EVENT);
    writer.u8(kind.code()).u64(id.0);
    match result {
        Ok(value) => writer.u8(1).u64(*value),
        Err(error) => writer.u8(0).text(&error.to_string()),
    };
    writer.finish()
}

fn decode_event(payload: &[u8]) -> Result<(Kind, JobId, Result<u64>)> {
    let mut reader = Reader(payload);
    let kind = Kind::decode(reader.u8()?)?;
    let id = JobId(reader.u64()?);
    let result = match reader.u8()? {
        1 => Ok(reader.u64()?),
        0 => Err(invalid(reader.text()?.to_owned())),
        _ => return Err(invalid("invalid prefix record event result")),
    };
    reader.end()?;
    Ok((kind, id, result))
}

impl DiskTransport for Sharded {
    fn kind(&self) -> Transport {
        Transport::Staged
    }

    fn store(
        &mut self,
        id: JobId,
        header: Header,
        temporary: PathBuf,
        path: PathBuf,
        checkpoint: &PublishedCheckpoint,
    ) -> Result<()> {
        self.accept()?;
        let tail = header.tail_pages as usize;
        let chain = checkpoint.chain_pages.len();
        if tail > chain {
            return Err(invalid(format!(
                "record {} holds {tail} tail pages but the published chain has only {chain}",
                record::file_name(&header.key)
            )));
        }
        let pages = &checkpoint.chain_pages[chain - tail..];
        let mut writer = Writer::new(STORE);
        writer
            .u64(id.0)
            .header(&header)?
            .text(name(&temporary)?)
            .text(name(&path)?)
            .u32(checkpoint.slot.0)
            .u32(pages.len() as u32);
        for page in pages {
            writer.u32(page.0);
        }
        let frame = writer.finish();
        self.local.store_pages(
            id,
            header,
            temporary,
            path.clone(),
            pages.to_vec(),
            checkpoint.slot,
        )?;
        self.begin(Kind::Stored, id, Some(path), &frame);
        Ok(())
    }

    fn fetch(&mut self, id: JobId, plan: Vec<FetchSegment>, ticket: &InboundTicket) -> Result<()> {
        self.accept()?;
        let mut writer = Writer::new(FETCH);
        writer
            .u64(id.0)
            .u32(ticket.slot.0)
            .u32(ticket.pages.len() as u32);
        for page in &ticket.pages {
            writer.u32(page.0);
        }
        writer.u32(plan.len() as u32);
        for segment in &plan {
            writer
                .text(name(&segment.path)?)
                .header(&segment.header)?
                .u32(segment.first_page)
                .u32(segment.pages)
                .u8(segment.row as u8);
        }
        let frame = writer.finish();
        self.local.fetch(id, plan, ticket)?;
        self.begin(Kind::Fetched, id, None, &frame);
        Ok(())
    }

    fn truncate(&mut self, id: JobId, path: PathBuf, length: u64) -> Result<()> {
        self.accept()?;
        let mut writer = Writer::new(TRUNCATE);
        writer.u64(id.0).text(name(&path)?).u64(length);
        let frame = writer.finish();
        self.local.truncate(id, path, length)?;
        self.begin(Kind::Truncated, id, None, &frame);
        Ok(())
    }

    fn unlink(&mut self, id: JobId, path: PathBuf) -> Result<()> {
        self.accept()?;
        let mut writer = Writer::new(UNLINK);
        writer.u64(id.0).text(name(&path)?);
        let frame = writer.finish();
        self.local.unlink(id, path)?;
        self.begin(Kind::Unlinked, id, None, &frame);
        Ok(())
    }

    fn poll(&mut self, events: &mut Vec<TransportEvent>) {
        if !self.closed {
            self.absorb_local();
            self.absorb_peers();
        }
        events.append(&mut self.events);
    }

    /// Close every peer's transport and rank zero's, collecting the completions each reaches.
    fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        let close = Writer::new(CLOSE).finish();
        for peer in &mut self.peers {
            let _ = peer.send(&close);
        }
        let local = self.local.close();
        self.absorb_local();
        let deadline = Instant::now() + transfer::CLOSE_TIMEOUT + REPLY_TIMEOUT;
        let mut failure = local.err();
        for index in 0..self.peers.len() {
            let rank = index + 1;
            let result = loop {
                match self.peers[index].receive() {
                    Ok(Some((EVENT, payload))) => match decode_event(&payload) {
                        Ok((kind, id, result)) => self.record(kind, id, rank, result),
                        Err(error) => break Err(self.peers[index].fail(error)),
                    },
                    Ok(Some((CLOSED, payload))) => break decode_closed(&payload),
                    Ok(Some(_)) => {
                        break Err(
                            self.peers[index].fail(invalid("unexpected prefix record message"))
                        );
                    }
                    Ok(None) if Instant::now() >= deadline => {
                        break Err(self.peers[index]
                            .fail(invalid("prefix record peer did not close in time")));
                    }
                    Ok(None) => std::thread::sleep(IDLE),
                    Err(error) => break Err(error),
                }
            };
            if let Err(error) = result {
                self.abandon(rank, error.clone());
                failure.get_or_insert(invalid(format!("rank {rank}: {error}")));
            }
        }
        self.closed = true;
        match failure {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

fn decode_closed(payload: &[u8]) -> Result<()> {
    let mut reader = Reader(payload);
    let result = match reader.u8()? {
        1 => Ok(()),
        0 => Err(invalid(reader.text()?.to_owned())),
        _ => return Err(invalid("invalid prefix record close result")),
    };
    reader.end()?;
    result
}

/// Rank zero's disk transport over its prepared staging resources and its peer connections.
pub(super) fn transport(
    staging: Option<super::slab::SlabPool>,
    transfers: Arc<Mutex<Transfers>>,
    peers: Vec<TcpStream>,
) -> Result<Box<dyn DiskTransport>> {
    let local = Staged::new(staging, transfers, DiskThread::start("flare-offload-disk")?);
    Ok(Box::new(Sharded::new(local, peers)?))
}

// ---------------------------------------------------------------------------------------------
// A peer's record service

/// A peer rank's record service: its staged transport, driven by rank zero over one connection.
pub struct Service {
    thread: Option<JoinHandle<Result<()>>>,
}

impl Service {
    /// Wait for rank zero to close the service, or for its connection to end.
    pub fn join(mut self) -> Result<()> {
        match self.thread.take().map(JoinHandle::join) {
            Some(Ok(result)) => result,
            Some(Err(_)) => Err(invalid("the prefix record service panicked")),
            None => Ok(()),
        }
    }
}

struct Peer {
    staged: Staged,
    directory: PathBuf,
    fingerprint: [u8; 16],
    link: Link,
    /// Retains the record directory through every job.
    _lock: record::DirectoryLock,
}

impl Peer {
    /// Rank zero's header with this rank's own layout fingerprint.
    fn own(&self, mut header: Header) -> Header {
        header.fingerprint = self.fingerprint;
        header
    }

    fn dispatch(&mut self, tag: u8, payload: &[u8]) -> Result<Option<(Kind, JobId, Error)>> {
        let mut reader = Reader(payload);
        let id = JobId(reader.u64()?);
        if tag == STORE {
            let header = self.own(reader.header()?);
            let temporary = file(&self.directory, reader.text()?)?;
            let path = file(&self.directory, reader.text()?)?;
            let slot = StateSlotId(reader.u32()?);
            let count = reader.count(4)?;
            let pages = (0..count)
                .map(|_| reader.u32().map(PageId))
                .collect::<Result<Vec<_>>>()?;
            reader.end()?;
            let result = self
                .staged
                .store_pages(id, header, temporary, path, pages, slot);
            return Ok(result.err().map(|error| (Kind::Stored, id, error)));
        }
        if tag == FETCH {
            let slot = StateSlotId(reader.u32()?);
            let count = reader.count(4)?;
            let pages = (0..count)
                .map(|_| reader.u32().map(PageId))
                .collect::<Result<Vec<_>>>()?;
            let segments = reader.count(HEADER_BYTES)?;
            let mut plan = Vec::with_capacity(segments);
            for _ in 0..segments {
                let path = file(&self.directory, reader.text()?)?;
                let header = self.own(reader.header()?);
                plan.push(FetchSegment {
                    path,
                    header,
                    first_page: reader.u32()?,
                    pages: reader.u32()?,
                    row: reader.u8()? != 0,
                });
            }
            reader.end()?;
            // The key names no record on this side; the transport uses the ids and the slot.
            let ticket = InboundTicket {
                key: plan
                    .last()
                    .map(|segment| segment.header.key)
                    .ok_or_else(|| invalid("a prefix record fetch names no record"))?,
                pages,
                slot,
            };
            let result = self.staged.fetch(id, plan, &ticket);
            return Ok(result.err().map(|error| (Kind::Fetched, id, error)));
        }
        if tag == TRUNCATE {
            let path = file(&self.directory, reader.text()?)?;
            let length = reader.u64()?;
            reader.end()?;
            let result = self.staged.truncate(id, path, length);
            return Ok(result.err().map(|error| (Kind::Truncated, id, error)));
        }
        if tag == UNLINK {
            let path = file(&self.directory, reader.text()?)?;
            reader.end()?;
            let result = self.staged.unlink(id, path);
            return Ok(result.err().map(|error| (Kind::Unlinked, id, error)));
        }
        Err(invalid("unexpected prefix record message"))
    }

    fn forward(&mut self) -> Result<()> {
        let mut events = Vec::new();
        self.staged.poll(&mut events);
        for event in &events {
            let (kind, id, result) = Kind::of(event);
            self.link.send(&encode_event(kind, id, &result))?;
        }
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        loop {
            let mut progressed = false;
            loop {
                match self.link.receive() {
                    Ok(None) => break,
                    Ok(Some((CLOSE, _))) => {
                        let closed = self.staged.close();
                        self.forward()?;
                        let mut writer = Writer::new(CLOSED);
                        match &closed {
                            Ok(()) => writer.u8(1),
                            Err(error) => writer.u8(0).text(&error.to_string()),
                        };
                        self.link.send(&writer.finish())?;
                        return closed;
                    }
                    Ok(Some((tag, payload))) => {
                        progressed = true;
                        if let Some((kind, id, error)) = self.dispatch(tag, &payload)? {
                            self.link.send(&encode_event(kind, id, &Err(error)))?;
                        }
                    }
                    Err(error) => {
                        // Rank zero is gone: finish what this rank started and stop.
                        let _ = self.staged.close();
                        return Err(error);
                    }
                }
            }
            self.forward()?;
            let (outbound, inbound) = self.staged.in_flight();
            std::thread::sleep(if progressed || outbound + inbound > 0 {
                BUSY
            } else {
                IDLE
            });
        }
    }
}

fn serve(prepared: Prepared, link: Link) -> Result<()> {
    let Prepared {
        resolved,
        fingerprint,
        geometry,
        contexts,
        staging_pool,
        directory_lock,
        ..
    } = prepared;
    let directory = resolved
        .directory
        .ok_or_else(|| invalid("a peer's prefix record service needs a record directory"))?;
    let lock = directory_lock
        .ok_or_else(|| invalid("a peer's prefix record service needs its directory lock"))?;
    let pool = staging_pool
        .as_ref()
        .ok_or_else(|| invalid("a peer's prefix record service needs a staging pool"))?;
    let transfers = Arc::new(Mutex::new(Transfers::start(&contexts, geometry, pool)?));
    let staged = Staged::new(
        staging_pool,
        transfers,
        DiskThread::start("flare-offload-disk")?,
    );
    let mut peer = Peer {
        staged,
        directory,
        fingerprint,
        link,
        _lock: lock,
    };
    peer.run()
}

// ---------------------------------------------------------------------------------------------
// Joining the ranks before serving

/// What a rank's scan holds, as the agreement compares it.
fn summary(scanned: &[Scanned]) -> Result<Vec<u8>> {
    let mut writer = Writer::new(SUMMARY);
    writer.u32(scanned.len() as u32);
    for record in scanned {
        writer
            .key(&record.header.key)
            .bytes(&shape(&record.header)?)
            .u8(record.row as u8);
    }
    Ok(writer.finish())
}

/// Digest of a header without the fields that differ between ranks' files of one record.
fn shape(header: &Header) -> Result<[u8; 16]> {
    let mut header = header.clone();
    header.fingerprint = [0; 16];
    header.row_checksum = 0;
    let mut hash = Blake2b::<U16>::new();
    hash.update(header.encode()?);
    Ok(hash.finalize().into())
}

fn decode_summary(payload: &[u8]) -> Result<HashMap<PrefixKey, ([u8; 16], bool)>> {
    let mut reader = Reader(payload);
    let count = reader.count(41)?;
    let mut records = HashMap::with_capacity(count);
    for _ in 0..count {
        let key = reader.key()?;
        let shape: [u8; 16] = reader.take(16)?.try_into().unwrap();
        records.insert(key, (shape, reader.u8()? != 0));
    }
    reader.end()?;
    Ok(records)
}

/// The records every rank holds with equal headers, each with its row when every rank has it,
/// closed under parents: a record whose parent is not agreed is dropped too.
fn agree(
    scanned: &[Scanned],
    peers: &[HashMap<PrefixKey, ([u8; 16], bool)>],
) -> Result<HashMap<PrefixKey, bool>> {
    let mut agreed = HashMap::new();
    for record in scanned {
        let shape = shape(&record.header)?;
        let mut row = record.row;
        let everywhere = peers.iter().all(|peer| match peer.get(&record.header.key) {
            Some((peer_shape, peer_row)) if *peer_shape == shape => {
                row &= peer_row;
                true
            }
            _ => false,
        });
        if everywhere {
            agreed.insert(record.header.key, row);
        }
    }
    let parents: HashMap<PrefixKey, Option<PrefixKey>> = scanned
        .iter()
        .map(|record| (record.header.key, record.header.parent))
        .collect();
    loop {
        let orphans: Vec<PrefixKey> = agreed
            .keys()
            .filter(|key| parents[key].is_some_and(|parent| !agreed.contains_key(&parent)))
            .copied()
            .collect();
        if orphans.is_empty() {
            return Ok(agreed);
        }
        for key in orphans {
            agreed.remove(&key);
        }
    }
}

fn encode_agreed(agreed: &HashMap<PrefixKey, bool>) -> Vec<u8> {
    let mut writer = Writer::new(AGREED);
    writer.u32(agreed.len() as u32);
    let mut keys: Vec<_> = agreed.iter().collect();
    keys.sort_by_key(|(key, _)| (key.consumed_tokens, key.digest));
    for (key, row) in keys {
        writer.key(key).u8(*row as u8);
    }
    writer.finish()
}

fn decode_agreed(payload: &[u8]) -> Result<HashMap<PrefixKey, bool>> {
    let mut reader = Reader(payload);
    let count = reader.count(25)?;
    let mut agreed = HashMap::with_capacity(count);
    for _ in 0..count {
        let key = reader.key()?;
        agreed.insert(key, reader.u8()? != 0);
    }
    reader.end()?;
    Ok(agreed)
}

/// Delete the files of records outside the agreement and truncate rows it does not keep.
fn retain(prepared: &mut Prepared, agreed: &HashMap<PrefixKey, bool>) -> Result<()> {
    let mut kept = Vec::with_capacity(agreed.len());
    for mut record in std::mem::take(&mut prepared.scanned) {
        match agreed.get(&record.header.key) {
            None => match std::fs::remove_file(&record.path) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(record::io_failure(
                        format!("delete record file {}", record.path.display()),
                        error,
                    ));
                }
            },
            Some(&row) => {
                if record.row && !row {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(&record.path)
                        .and_then(|file| file.set_len(record.header.row_segment))
                        .map_err(|error| {
                            record::io_failure(
                                format!("truncate record file {}", record.path.display()),
                                error,
                            )
                        })?;
                    record.row = false;
                    record.bytes = record.header.row_segment;
                }
                kept.push(record);
            }
        }
    }
    prepared
        .disk_budget
        .set_checkpoint(kept.iter().map(|record| record.bytes).sum());
    prepared.scanned = kept;
    Ok(())
}

/// Whether this rank serves a disk store its peers can mirror; a memory tier keeps records in
/// one process's memory, which other hosts cannot reach.
fn mirrored(prepared: &Prepared) -> Result<bool> {
    if prepared.memory_pool.is_some() {
        return Err(invalid(
            "a multi-host package keeps host prefix records on disk only; set --prefix-cache-memory 0",
        ));
    }
    Ok(prepared.directory_lock.is_some())
}

fn nonce() -> [u8; 16] {
    let mut hash = Blake2b::<U16>::new();
    hash.update(std::process::id().to_le_bytes());
    hash.update(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .to_le_bytes(),
    );
    let mut random = [0u8; 16];
    if let Ok(mut source) = std::fs::File::open("/dev/urandom") {
        let _ = source.read_exact(&mut random);
    }
    hash.update(random);
    hash.finalize().into()
}

/// Rank zero, after its peers joined and before serving: agree with every peer on whether a disk
/// store exists, connect a record connection to each, and agree on the scanned records. Returns
/// a summary for the startup report when the ranks share a store.
pub fn join_rank_zero(store: Option<&Shared>, peers: &mut [RankStream]) -> Result<Option<String>> {
    use crate::execution::rendezvous;

    let mut prepared = store.map(Shared::lock);
    let prepared = prepared.as_mut().and_then(|guard| guard.as_mut());
    let enabled = match &prepared {
        Some(prepared) => mirrored(prepared)?,
        None => false,
    };
    let local_ip = match peers.first() {
        Some(RankStream::Tcp(stream)) => stream
            .local_addr()
            .map_err(|error| record::io_failure("inspect the rendezvous connection", error))?
            .ip(),
        _ => {
            return Err(invalid(
                "a multi-host prefix store requires TCP rank connections",
            ));
        }
    };
    let listener = if enabled {
        Some(
            TcpListener::bind((local_ip, 0))
                .map_err(|error| record::io_failure("listen for prefix record peers", error))?,
        )
    } else {
        None
    };
    let token = nonce();
    let mut offer = vec![enabled as u8];
    if let Some(listener) = &listener {
        let address = listener
            .local_addr()
            .map_err(|error| record::io_failure("inspect the prefix record listener", error))?;
        offer.extend(token);
        offer.extend(address.to_string().as_bytes());
    }
    let replies = rendezvous::share_records(peers, offer)?;
    let peer_enabled: Vec<bool> = replies[1..].iter().map(|reply| reply == &[1]).collect();
    if let Some(rank) = peer_enabled.iter().position(|&peer| peer != enabled) {
        return Err(invalid(format!(
            "rank {} {} a disk prefix store and rank zero {}; give every rank the same --prefix-cache-disk settings",
            rank + 1,
            if enabled { "lacks" } else { "has" },
            if enabled { "has one" } else { "does not" },
        )));
    }
    let (Some(listener), Some(prepared)) = (listener, prepared) else {
        return Ok(None);
    };
    let world = peers.len() + 1;
    let mut links: Vec<Option<Link>> = (1..world).map(|_| None).collect();
    listener
        .set_nonblocking(true)
        .map_err(|error| record::io_failure("configure the prefix record listener", error))?;
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    while links.iter().any(Option::is_none) {
        let stream = match listener.accept() {
            Ok((stream, _)) => stream,
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(invalid(
                        "peer ranks did not connect their prefix record connections",
                    ));
                }
                std::thread::sleep(IDLE);
                continue;
            }
            Err(error) => {
                return Err(record::io_failure("accept a prefix record peer", error));
            }
        };
        let mut link = Link::new(stream)?;
        let Ok(hello) = link.expect(HELLO, CONNECT_TIMEOUT) else {
            continue;
        };
        let mut reader = Reader(&hello);
        let (Ok(offered), Ok(rank)) = (reader.take(16), reader.u32()) else {
            continue;
        };
        let rank = rank as usize;
        if offered != token || !(1..world).contains(&rank) || links[rank - 1].is_some() {
            continue;
        }
        links[rank - 1] = Some(link);
    }
    let mut links: Vec<Link> = links.into_iter().map(Option::unwrap).collect();
    let summaries = links
        .iter_mut()
        .map(|link| decode_summary(&link.expect(SUMMARY, REPLY_TIMEOUT)?))
        .collect::<Result<Vec<_>>>()?;
    let agreed = agree(&prepared.scanned, &summaries)?;
    let frame = encode_agreed(&agreed);
    for link in &mut links {
        link.send(&frame)?;
    }
    let scanned = prepared.scanned.len();
    retain(prepared, &agreed)?;
    prepared.shards = links.into_iter().map(|link| link.stream).collect();
    Ok(Some(format!(
        "{} of {scanned} records on every rank",
        prepared.scanned.len()
    )))
}

/// A peer, at the point where rank zero runs [`join_rank_zero`]: connect this rank's record
/// connection, apply the scan agreement, and start the service rank zero drives.
pub fn join_peer(
    store: Option<&Shared>,
    stream: &mut RankStream,
    rank: usize,
    world: usize,
) -> Result<Option<Service>> {
    use crate::execution::rendezvous;

    let enabled = match store
        .map(Shared::lock)
        .as_ref()
        .and_then(|guard| guard.as_ref())
    {
        Some(prepared) => mirrored(prepared)?,
        None => false,
    };
    let records = rendezvous::exchange_record(stream, &[enabled as u8], world)?;
    let offer = &records[0];
    let zero = offer.first() == Some(&1);
    if zero != enabled {
        return Err(invalid(format!(
            "rank zero {} a disk prefix store and rank {rank} {}; give every rank the same --prefix-cache-disk settings",
            if zero { "has" } else { "lacks" },
            if enabled { "has one" } else { "does not" },
        )));
    }
    if !enabled {
        return Ok(None);
    }
    if records[1..].iter().any(|record| record != &[1]) {
        return Err(invalid(
            "another peer rank lacks a disk prefix store; give every rank the same --prefix-cache-disk settings",
        ));
    }
    let (token, address) = offer[1..]
        .split_at_checked(16)
        .ok_or_else(|| invalid("rank zero's prefix record offer is malformed"))?;
    let address = std::str::from_utf8(address)
        .map_err(|_| invalid("rank zero's prefix record address is malformed"))?;
    let connected = TcpStream::connect_timeout(
        &address
            .parse()
            .map_err(|_| invalid("rank zero's prefix record address is malformed"))?,
        CONNECT_TIMEOUT,
    )
    .map_err(|error| {
        record::io_failure(
            format!("connect to rank zero's prefix records at {address}"),
            error,
        )
    })?;
    let mut link = Link::new(connected)?;
    let mut hello = Writer::new(HELLO);
    hello.bytes(token).u32(rank as u32);
    link.send(&hello.finish())?;
    let shared = store.ok_or_else(|| invalid("a peer's prefix store vanished"))?;
    let mut prepared = shared
        .take()
        .ok_or_else(|| invalid("a peer's prefix store was already opened"))?;
    link.send(&summary(&prepared.scanned)?)?;
    let agreed = decode_agreed(&link.expect(AGREED, REPLY_TIMEOUT)?)?;
    retain(&mut prepared, &agreed)?;
    let thread = std::thread::Builder::new()
        .name("flare-offload-peer".into())
        .spawn(move || serve(prepared, link))
        .map_err(|error| invalid(format!("cannot start the prefix record service: {error}")))?;
    Ok(Some(Service {
        thread: Some(thread),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offload::geometry::tests::geometry;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn directory(test: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("flare-shard-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    /// A staged transport without a staging pool: its disk thread runs truncates and unlinks.
    fn staged() -> Staged {
        let transfers = Transfers::stopped(&[geometry(64, &[("kv", 4096)], 4096)]).unwrap();
        Staged::new(
            None,
            Arc::new(Mutex::new(transfers)),
            DiskThread::start("flare-shard-test").unwrap(),
        )
    }

    fn connected() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        (client, listener.accept().unwrap().0)
    }

    fn record_file(directory: &Path, tokens: u64) -> PathBuf {
        let path = directory.join(record::file_name(&key(tokens, 0x5a)));
        std::fs::write(&path, vec![1u8; 8192]).unwrap();
        path
    }

    /// Poll until `count` events arrived, failing after a few seconds.
    fn events(transport: &mut Sharded, count: usize) -> Vec<TransportEvent> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut events = Vec::new();
        while events.len() < count {
            assert!(Instant::now() < deadline, "only {events:?} arrived");
            transport.poll(&mut events);
            std::thread::sleep(IDLE);
        }
        events
    }

    /// Poll for a while and require that nothing was reported.
    fn quiet(transport: &mut Sharded) {
        let mut events = Vec::new();
        for _ in 0..50 {
            transport.poll(&mut events);
            std::thread::sleep(IDLE);
        }
        assert!(events.is_empty(), "{events:?}");
    }

    fn reply(peer: &mut Link) -> (u8, Vec<u8>) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(frame) = peer.receive().unwrap() {
                return frame;
            }
            assert!(Instant::now() < deadline, "no frame arrived");
            std::thread::sleep(IDLE);
        }
    }

    #[test]
    fn rank_zero_reports_a_job_after_every_rank_and_unlinks_partial_stores() {
        let directory = directory("zero");
        let (zero, peer) = connected();
        let mut peer = Link::new(peer).unwrap();
        let mut transport = Sharded::new(staged(), vec![zero]).unwrap();

        // Rank zero's unlink finishes first; the job waits for the peer.
        let first = record_file(&directory, 128);
        transport.unlink(JobId(1), first.clone()).unwrap();
        let (tag, payload) = reply(&mut peer);
        let mut reader = Reader(&payload);
        assert_eq!((tag, reader.u64().unwrap()), (UNLINK, 1));
        assert_eq!(reader.text().unwrap(), name(&first).unwrap());
        quiet(&mut transport);
        assert!(!first.exists());
        peer.send(&encode_event(Kind::Unlinked, JobId(1), &Ok(0)))
            .unwrap();
        assert!(matches!(
            events(&mut transport, 1).as_slice(),
            [TransportEvent::Unlinked {
                id: JobId(1),
                result: Ok(())
            }]
        ));

        // A peer's failure fails the job and names the rank.
        let second = record_file(&directory, 256);
        transport.truncate(JobId(2), second.clone(), 4096).unwrap();
        assert_eq!(reply(&mut peer).0, TRUNCATE);
        peer.send(&encode_event(
            Kind::Truncated,
            JobId(2),
            &Err(invalid("disk full")),
        ))
        .unwrap();
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::Truncated {
                    id: JobId(2),
                    result: Err(error),
                },
            ] => {
                assert!(error.to_string().contains("rank 1: "), "{error}");
                assert!(error.to_string().contains("disk full"), "{error}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(std::fs::metadata(&second).unwrap().len(), 4096);

        // Rank zero wrote a record the peer failed to write: rank zero removes its file and
        // only then reports the store as failed. The peer, which wrote nothing, is not asked.
        let third = record_file(&directory, 384);
        transport.pending.insert(
            (Kind::Stored, JobId(3)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Ok(8192)), None],
                path: Some(third.clone()),
                cleanup: None,
            },
        );
        transport.record(Kind::Stored, JobId(3), 1, Err(invalid("no space")));
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::Stored {
                    id: JobId(3),
                    result: Err(error),
                },
            ] => {
                assert!(error.to_string().contains("no space"), "{error}");
            }
            other => panic!("{other:?}"),
        }
        assert!(!third.exists());

        // The peer wrote a record rank zero failed to write: the peer is asked to remove it,
        // with a job id outside the disk store's, and the failure waits for its answer.
        let fourth = record_file(&directory, 512);
        transport.pending.insert(
            (Kind::Stored, JobId(4)),
            Pending {
                kind: Kind::Stored,
                results: vec![Some(Err(invalid("short write"))), None],
                path: Some(fourth.clone()),
                cleanup: None,
            },
        );
        transport.record(Kind::Stored, JobId(4), 1, Ok(8192));
        let (tag, payload) = reply(&mut peer);
        let mut reader = Reader(&payload);
        let internal = reader.u64().unwrap();
        assert_eq!(tag, UNLINK);
        assert!(internal >= INTERNAL_JOBS);
        assert_eq!(reader.text().unwrap(), name(&fourth).unwrap());
        quiet(&mut transport);
        peer.send(&encode_event(Kind::Unlinked, JobId(internal), &Ok(0)))
            .unwrap();
        match events(&mut transport, 1).as_slice() {
            [
                TransportEvent::Stored {
                    id: JobId(4),
                    result: Err(error),
                },
            ] => {
                assert!(error.to_string().contains("short write"), "{error}");
            }
            other => panic!("{other:?}"),
        }

        // A broken connection fails every job still waiting on that rank.
        let fifth = record_file(&directory, 640);
        transport.unlink(JobId(5), fifth).unwrap();
        assert_eq!(reply(&mut peer).0, UNLINK);
        drop(peer);
        assert!(matches!(
            events(&mut transport, 1).as_slice(),
            [TransportEvent::Unlinked {
                id: JobId(5),
                result: Err(_)
            }]
        ));
        assert!(
            transport
                .unlink(JobId(6), record_file(&directory, 768))
                .is_ok()
        );
        assert!(matches!(
            events(&mut transport, 1).as_slice(),
            [TransportEvent::Unlinked {
                id: JobId(6),
                result: Err(_)
            }]
        ));
        assert!(transport.close().is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn peer_service_mirrors_jobs_into_its_own_directory_until_closed() {
        let zero_directory = directory("mirror-zero");
        let peer_directory = directory("mirror-peer");
        let (zero, peer) = connected();
        let lock = record::DirectoryLock::acquire(&peer_directory).unwrap();
        let service_directory = peer_directory.clone();
        let service = std::thread::spawn(move || {
            Peer {
                staged: staged(),
                directory: service_directory,
                fingerprint: [3; 16],
                link: Link::new(peer).unwrap(),
                _lock: lock,
            }
            .run()
        });
        let mut transport = Sharded::new(staged(), vec![zero]).unwrap();
        let (ours, theirs) = (
            record_file(&zero_directory, 128),
            record_file(&peer_directory, 128),
        );
        let (kept_ours, kept_theirs) = (
            record_file(&zero_directory, 256),
            record_file(&peer_directory, 256),
        );
        transport.unlink(JobId(1), ours.clone()).unwrap();
        transport
            .truncate(JobId(2), kept_ours.clone(), 4096)
            .unwrap();
        let mut events = events(&mut transport, 2);
        events.sort_by_key(|event| Kind::of(event).1);
        assert!(matches!(
            events.as_slice(),
            [
                TransportEvent::Unlinked {
                    id: JobId(1),
                    result: Ok(())
                },
                TransportEvent::Truncated {
                    id: JobId(2),
                    result: Ok(())
                },
            ]
        ));
        assert!(!ours.exists() && !theirs.exists());
        assert_eq!(std::fs::metadata(&kept_ours).unwrap().len(), 4096);
        assert_eq!(std::fs::metadata(&kept_theirs).unwrap().len(), 4096);

        transport.close().unwrap();
        service.join().unwrap().unwrap();
        std::fs::remove_dir_all(zero_directory).unwrap();
        std::fs::remove_dir_all(peer_directory).unwrap();
    }

    fn key(tokens: u64, seed: u8) -> PrefixKey {
        PrefixKey {
            consumed_tokens: tokens,
            digest: [seed; 16],
        }
    }

    fn scanned(key: PrefixKey, parent: Option<PrefixKey>, row: bool, fingerprint: u8) -> Scanned {
        let geometry = vec![geometry(64, &[("kv", 4096)], 4096)];
        let mut header = Header::new(
            &geometry,
            [7; 32],
            [fingerprint; 16],
            key,
            parent,
            [0; 16],
            2,
            UNIX_EPOCH + Duration::from_secs(1_000_000),
        )
        .unwrap();
        header.row_checksum = fingerprint as u64;
        Scanned {
            bytes: if row {
                header.length_with_row()
            } else {
                header.row_segment
            },
            path: PathBuf::from(record::file_name(&key)),
            header,
            row,
            modified: SystemTime::now(),
        }
    }

    #[test]
    fn agreement_keeps_records_every_rank_holds_and_drops_their_orphans() {
        let (a, b, c, d) = (key(128, 1), key(256, 2), key(384, 3), key(128, 4));
        let zero = vec![
            scanned(a, None, true, 0),
            scanned(b, Some(a), true, 0),
            scanned(c, Some(b), true, 0),
            scanned(d, None, true, 0),
        ];
        // Rank one lacks b's row and holds no d; rank two lacks d's row too.
        let one = vec![
            scanned(a, None, true, 1),
            scanned(b, Some(a), false, 1),
            scanned(c, Some(b), true, 1),
        ];
        let two = vec![
            scanned(a, None, true, 2),
            scanned(b, Some(a), true, 2),
            scanned(c, Some(b), true, 2),
            scanned(d, None, false, 2),
        ];
        let peers = [one, two]
            .iter()
            .map(|records| decode_summary(&summary(records).unwrap()[5..]).unwrap())
            .collect::<Vec<_>>();
        let agreed = agree(&zero, &peers).unwrap();
        assert_eq!(agreed, HashMap::from([(a, true), (b, false), (c, true)]));
        let decoded = decode_agreed(&encode_agreed(&agreed)[5..]).unwrap();
        assert_eq!(decoded, agreed);

        // A peer whose header for `a` differs drops `a` and, with it, every descendant.
        let mut other = scanned(a, None, true, 1);
        other.header.bucket = [9; 16];
        let peers = [decode_summary(&summary(&[other]).unwrap()[5..]).unwrap()];
        assert!(agree(&zero, &peers).unwrap().is_empty());
    }

    #[test]
    fn events_and_peer_file_names_round_trip_and_reject_paths() {
        for result in [Ok(42u64), Err(invalid("disk full"))] {
            let frame = encode_event(Kind::Fetched, JobId(9), &result);
            let (kind, id, decoded) = decode_event(&frame[5..]).unwrap();
            assert_eq!((kind, id), (Kind::Fetched, JobId(9)));
            match (result, decoded) {
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                (Err(a), Err(b)) => assert!(b.to_string().contains(&a.to_string())),
                other => panic!("{other:?}"),
            }
        }
        let directory = Path::new("/records");
        let key = key(4096, 0xab);
        let plain = record::file_name(&key);
        assert_eq!(file(directory, &plain).unwrap(), directory.join(&plain));
        let temporary = record::temporary_name(&key);
        assert_eq!(
            file(directory, &temporary).unwrap(),
            directory.join(&temporary)
        );
        for name in [
            "../etc/passwd",
            "/abs.ckpt",
            "x.ckpt",
            ".tmp",
            "",
            &format!("a/{plain}"),
        ] {
            assert!(file(directory, name).is_err(), "{name:?}");
        }
    }
}
