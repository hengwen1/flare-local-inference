//! Framed activation transport between adjacent pipeline stages.
//!
//! A boundary link carries the control and data frames for one adjacent pair of
//! stages. Frames are small, fixed-layout records so the runtime can reserve an
//! activation slot before a producer writes it and release that slot only after
//! the consumer's device copy has completed.

use std::{
    io::{self, Read, Write},
    net::{TcpListener, TcpStream},
};

/// Largest activation payload accepted by a boundary link.
pub const MAX_PAYLOAD_BYTES: usize = 64 << 20;

const HEADER_BYTES: usize = 13;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameKind {
    /// Producer reserves a boundary slot for `fence`.
    Prepare,
    /// Consumer acknowledges that the slot and its device buffer are available.
    Ready,
    /// Producer publishes one activation payload.
    Data,
    /// Consumer releases a slot after its copy has completed.
    Release,
    /// Last stage reports that the request's sampled result is committed.
    Commit,
    /// Either side abandons an in-flight request.
    Cancel,
    /// The sender will send no further frames.
    Drain,
    /// The sender failed; the payload is a UTF-8 description.
    Error,
}

impl FrameKind {
    fn from_byte(byte: u8) -> io::Result<Self> {
        Ok(match byte {
            0 => Self::Prepare,
            1 => Self::Ready,
            2 => Self::Data,
            3 => Self::Release,
            4 => Self::Commit,
            5 => Self::Cancel,
            6 => Self::Drain,
            7 => Self::Error,
            _ => return Err(invalid("unknown pipeline frame kind")),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Frame {
    pub kind: FrameKind,
    pub fence: u64,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn new(kind: FrameKind, fence: u64) -> Self {
        Self {
            kind,
            fence,
            payload: Vec::new(),
        }
    }

    pub fn with_payload(kind: FrameKind, fence: u64, payload: impl Into<Vec<u8>>) -> Self {
        Self {
            kind,
            fence,
            payload: payload.into(),
        }
    }

    fn encode_header_into(&self, out: &mut Vec<u8>, payload_len: usize) -> io::Result<()> {
        let length = u32::try_from(payload_len)
            .map_err(|_| invalid("pipeline frame payload exceeds the wire limit"))?;
        out.clear();
        out.reserve(HEADER_BYTES + payload_len);
        out.extend_from_slice(&length.to_be_bytes());
        out.push(self.kind as u8);
        out.extend_from_slice(&self.fence.to_be_bytes());
        Ok(())
    }
}

/// A blocking, full-duplex boundary link.
///
/// The producer sends `Prepare`/`Data`/`Cancel`/`Drain`; the consumer sends
/// `Ready`/`Release`/`Commit`/`Cancel`/`Drain`. Both directions share one TCP
/// connection and are independently framed, so acknowledgements cannot overtake
/// the frames they acknowledge.
pub struct BoundaryLink {
    stream: TcpStream,
    max_payload: usize,
    write_buffer: Vec<u8>,
}

impl BoundaryLink {
    pub fn new(stream: TcpStream) -> io::Result<Self> {
        stream.set_nodelay(true)?;
        stream.set_nonblocking(false)?;
        Ok(Self {
            stream,
            max_payload: MAX_PAYLOAD_BYTES,
            write_buffer: Vec::new(),
        })
    }

    pub fn connect(address: impl std::net::ToSocketAddrs) -> io::Result<Self> {
        Self::new(TcpStream::connect(address)?)
    }

    pub fn accept(listener: &TcpListener) -> io::Result<Self> {
        let (stream, _) = listener.accept()?;
        Self::new(stream)
    }

    /// Adopts a TCP rank stream as a pipeline boundary link.
    pub fn from_rank_stream(stream: super::RankStream) -> io::Result<Self> {
        match stream {
            super::RankStream::Tcp(stream) => Self::new(stream),
            #[cfg(unix)]
            super::RankStream::Unix(_) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "pipeline boundaries require TCP rank streams",
            )),
        }
    }

    pub fn set_max_payload(&mut self, max_payload: usize) {
        self.max_payload = max_payload;
    }

    pub fn send(&mut self, frame: &Frame) -> io::Result<()> {
        self.send_parts(frame.kind, frame.fence, &frame.payload)
    }

    /// Sends a frame from caller-owned payload storage without copying it into a
    /// temporary frame.
    pub fn send_parts(&mut self, kind: FrameKind, fence: u64, payload: &[u8]) -> io::Result<()> {
        if payload.len() > self.max_payload {
            return Err(invalid(
                "pipeline frame payload exceeds the configured limit",
            ));
        }
        let frame = Frame {
            kind,
            fence,
            payload: Vec::new(),
        };
        frame.encode_header_into(&mut self.write_buffer, payload.len())?;
        self.write_buffer.extend_from_slice(payload);
        self.stream.write_all(&self.write_buffer)
    }

    pub fn receive(&mut self) -> io::Result<Frame> {
        let mut payload = Vec::new();
        let (kind, fence) = self.receive_into(&mut payload)?;
        Ok(Frame {
            kind,
            fence,
            payload,
        })
    }

    /// Receives one frame into caller-owned payload storage. The buffer is
    /// resized to the payload length and retains its capacity across frames.
    pub fn receive_into(&mut self, payload: &mut Vec<u8>) -> io::Result<(FrameKind, u64)> {
        let (kind, fence, length) = self.receive_header()?;
        self.receive_payload(payload, length)?;
        Ok((kind, fence))
    }

    /// Reads one frame header, leaving its payload unread so the caller can
    /// choose where to receive it.
    pub fn receive_header(&mut self) -> io::Result<(FrameKind, u64, usize)> {
        let mut header = [0u8; HEADER_BYTES];
        self.stream.read_exact(&mut header)?;
        let length = u32::from_be_bytes(header[..4].try_into().unwrap()) as usize;
        if length > self.max_payload {
            return Err(invalid(
                "pipeline frame payload exceeds the configured limit",
            ));
        }
        let kind = FrameKind::from_byte(header[4])?;
        let fence = u64::from_be_bytes(header[5..].try_into().unwrap());
        Ok((kind, fence, length))
    }

    /// Reads the payload announced by [`Self::receive_header`].
    pub fn receive_payload(&mut self, payload: &mut Vec<u8>, length: usize) -> io::Result<()> {
        if length > self.max_payload {
            return Err(invalid(
                "pipeline frame payload exceeds the configured limit",
            ));
        }
        payload.resize(length, 0);
        self.stream.read_exact(payload)
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Fixed-capacity reservation state for one boundary's activation slots.
///
/// A fence is reserved before the producer writes its payload and released only
/// after the consumer's copy has completed. Slots may be released out of order;
/// completion ordering remains the scheduler's responsibility.
#[derive(Debug)]
pub struct SlotRing {
    slots: Vec<Option<u64>>,
    free: usize,
}

impl SlotRing {
    pub fn new(slots: usize) -> Self {
        assert!(
            slots > 0,
            "a boundary requires at least one activation slot"
        );
        Self {
            slots: vec![None; slots],
            free: slots,
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn free(&self) -> usize {
        self.free
    }

    /// Reserves the first free slot for `fence`, or returns `None` when the ring
    /// is full or already holds that fence.
    pub fn reserve(&mut self, fence: u64) -> Option<usize> {
        if self.slots.iter().any(|slot| *slot == Some(fence)) {
            return None;
        }
        let index = self.slots.iter().position(Option::is_none)?;
        self.slots[index] = Some(fence);
        self.free -= 1;
        Some(index)
    }

    /// Releases `fence`'s slot, returning whether it was reserved.
    pub fn release(&mut self, fence: u64) -> bool {
        let Some(index) = self.slots.iter().position(|slot| *slot == Some(fence)) else {
            return false;
        };
        self.slots[index] = None;
        self.free += 1;
        true
    }

    pub fn is_reserved(&self, fence: u64) -> bool {
        self.slots.contains(&Some(fence))
    }
}

/// One activation payload received into a reserved boundary slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundaryData {
    pub fence: u64,
    pub slot: usize,
    pub nbytes: usize,
}

/// A consumer-visible boundary event other than a payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoundaryEvent {
    Data(BoundaryData),
    Cancel { fence: u64 },
    Drain,
}

/// Producer half of one boundary.
///
/// `publish` waits for the consumer's slot acknowledgement before sending the
/// activation, so the consumer never receives data it has no room to retain.
pub struct BoundaryProducer {
    link: BoundaryLink,
    scratch: Vec<u8>,
}

impl BoundaryProducer {
    pub fn new(link: BoundaryLink) -> Self {
        Self {
            link,
            scratch: Vec::new(),
        }
    }

    /// Reserves the consumer's next slot and publishes one activation payload.
    pub fn publish(&mut self, fence: u64, payload: &[u8]) -> io::Result<()> {
        self.publish_prepared(fence, &[], payload)
    }

    /// Publishes one activation with a stage control payload in its `PREPARE`
    /// frame. The activation itself follows in `DATA`.
    pub fn publish_prepared(
        &mut self,
        fence: u64,
        prepare: &[u8],
        payload: &[u8],
    ) -> io::Result<()> {
        self.link.send_parts(FrameKind::Prepare, fence, prepare)?;
        loop {
            let (kind, reply, length) = self.link.receive_header()?;
            self.link.receive_payload(&mut self.scratch, length)?;
            match kind {
                FrameKind::Ready if reply == fence => break,
                FrameKind::Cancel if reply == fence => {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "pipeline boundary was cancelled",
                    ));
                }
                FrameKind::Error => return Err(remote_error(&self.scratch)),
                _ => return Err(unexpected("producer", kind, reply)),
            }
        }
        self.link.send_parts(FrameKind::Data, fence, payload)
    }

    /// Waits until the consumer has copied the activation out of its slot.
    pub fn wait_release(&mut self, fence: u64) -> io::Result<()> {
        loop {
            let (kind, reply, length) = self.link.receive_header()?;
            self.link.receive_payload(&mut self.scratch, length)?;
            match kind {
                FrameKind::Release if reply == fence => return Ok(()),
                FrameKind::Cancel if reply == fence => {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "pipeline boundary was cancelled",
                    ));
                }
                FrameKind::Error => return Err(remote_error(&self.scratch)),
                _ => return Err(unexpected("producer", kind, reply)),
            }
        }
    }

    /// Waits for the final stage's committed completion payload.
    pub fn wait_commit(&mut self, fence: u64, payload: &mut Vec<u8>) -> io::Result<()> {
        loop {
            let (kind, reply, length) = self.link.receive_header()?;
            self.link.receive_payload(payload, length)?;
            match kind {
                FrameKind::Commit if reply == fence => return Ok(()),
                FrameKind::Cancel if reply == fence => {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "pipeline boundary was cancelled",
                    ));
                }
                FrameKind::Error => return Err(remote_error(payload)),
                _ => return Err(unexpected("producer", kind, reply)),
            }
        }
    }

    pub fn cancel(&mut self, fence: u64) -> io::Result<()> {
        self.link.send_parts(FrameKind::Cancel, fence, &[])
    }

    pub fn drain(&mut self) -> io::Result<()> {
        self.link.send_parts(FrameKind::Drain, 0, &[])
    }
}

/// Consumer half of one boundary, retaining its activation-slot reservations.
pub struct BoundaryConsumer {
    link: BoundaryLink,
    slots: SlotRing,
    scratch: Vec<u8>,
}

impl BoundaryConsumer {
    pub fn new(link: BoundaryLink, slots: usize) -> Self {
        Self {
            link,
            slots: SlotRing::new(slots),
            scratch: Vec::new(),
        }
    }

    /// Waits for one prepared activation and receives its payload directly into
    /// `payload`. The returned slot remains reserved until `release` or `commit`.
    pub fn receive_into(&mut self, payload: &mut Vec<u8>) -> io::Result<BoundaryEvent> {
        self.receive_prepared_into(&mut Vec::new(), payload)
    }

    /// Receives one activation and its `PREPARE` control payload. The control
    /// payload is copied into `prepare`; the activation payload is received
    /// directly into `payload`.
    pub fn receive_prepared_into(
        &mut self,
        prepare: &mut Vec<u8>,
        payload: &mut Vec<u8>,
    ) -> io::Result<BoundaryEvent> {
        let mut prepared = None;
        loop {
            let (kind, fence, length) = self.link.receive_header()?;
            match kind {
                FrameKind::Prepare => {
                    self.link.receive_payload(prepare, length)?;
                    if prepared.is_some() {
                        return Err(invalid("pipeline boundary prepared two fences at once"));
                    }
                    let Some(slot) = self.slots.reserve(fence) else {
                        let message = format!("pipeline boundary ring is full at fence {fence}");
                        self.link
                            .send_parts(FrameKind::Error, fence, message.as_bytes())?;
                        return Err(invalid(&message));
                    };
                    prepared = Some((fence, slot));
                    self.link.send_parts(FrameKind::Ready, fence, &[])?;
                }
                FrameKind::Data => {
                    let Some((prepared_fence, slot)) = prepared.take() else {
                        return Err(invalid("pipeline boundary received data without prepare"));
                    };
                    if prepared_fence != fence {
                        return Err(invalid("pipeline boundary data fence differs from prepare"));
                    }
                    self.link.receive_payload(payload, length)?;
                    return Ok(BoundaryEvent::Data(BoundaryData {
                        fence,
                        slot,
                        nbytes: length,
                    }));
                }
                FrameKind::Cancel => {
                    self.link.receive_payload(&mut self.scratch, length)?;
                    if prepared
                        .take()
                        .is_some_and(|(prepared, _)| prepared == fence)
                    {
                        self.slots.release(fence);
                    }
                    return Ok(BoundaryEvent::Cancel { fence });
                }
                FrameKind::Drain => {
                    self.link.receive_payload(&mut self.scratch, length)?;
                    if let Some((prepared, _)) = prepared.take() {
                        self.slots.release(prepared);
                    }
                    return Ok(BoundaryEvent::Drain);
                }
                FrameKind::Error => {
                    self.link.receive_payload(&mut self.scratch, length)?;
                    if let Some((prepared, _)) = prepared.take() {
                        self.slots.release(prepared);
                    }
                    return Err(remote_error(&self.scratch));
                }
                _ => return Err(unexpected("consumer", kind, fence)),
            }
        }
    }

    /// Reports a local stage failure to the upstream producer.
    pub fn fail(&mut self, message: &str) -> io::Result<()> {
        self.link
            .send_parts(FrameKind::Error, 0, message.as_bytes())
    }

    /// Releases a consumed slot and acknowledges it to the producer.
    pub fn release(&mut self, fence: u64) -> io::Result<()> {
        self.complete(fence, FrameKind::Release, &[])
    }

    /// Releases a committed slot and reports the request's final result upstream.
    pub fn commit(&mut self, fence: u64) -> io::Result<()> {
        self.commit_with(fence, &[])
    }

    /// Releases a committed slot and sends the final-stage completion payload.
    pub fn commit_with(&mut self, fence: u64, payload: &[u8]) -> io::Result<()> {
        self.complete(fence, FrameKind::Commit, payload)
    }

    fn complete(&mut self, fence: u64, kind: FrameKind, payload: &[u8]) -> io::Result<()> {
        if !self.slots.release(fence) {
            return Err(invalid("pipeline boundary released an unreserved slot"));
        }
        self.link.send_parts(kind, fence, payload)
    }
}

fn remote_error(payload: &[u8]) -> io::Error {
    io::Error::new(
        io::ErrorKind::Other,
        format!("pipeline peer failed: {}", String::from_utf8_lossy(payload)),
    )
}

fn unexpected(side: &str, kind: FrameKind, fence: u64) -> io::Error {
    invalid(&format!(
        "pipeline {side} received unexpected {kind:?} for fence {fence}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    fn pair() -> (BoundaryLink, BoundaryLink) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || BoundaryLink::connect(address).unwrap());
        let server = BoundaryLink::accept(&listener).unwrap();
        (client.join().unwrap(), server)
    }

    #[test]
    fn frames_round_trip_in_both_directions() {
        let (mut producer, mut consumer) = pair();
        producer
            .send(&Frame::with_payload(FrameKind::Prepare, 7, [8u8, 0, 0, 0]))
            .unwrap();
        producer
            .send(&Frame::with_payload(FrameKind::Data, 7, [1u8, 2, 3, 4]))
            .unwrap();
        consumer.send(&Frame::new(FrameKind::Ready, 7)).unwrap();
        consumer
            .send(&Frame::with_payload(FrameKind::Release, 7, []))
            .unwrap();

        let prepare = consumer.receive().unwrap();
        assert_eq!(prepare.kind, FrameKind::Prepare);
        assert_eq!(prepare.fence, 7);
        assert_eq!(prepare.payload, [8, 0, 0, 0]);
        assert_eq!(consumer.receive().unwrap().payload, [1, 2, 3, 4]);
        assert_eq!(producer.receive().unwrap().kind, FrameKind::Ready);
        assert_eq!(producer.receive().unwrap().kind, FrameKind::Release);
    }

    #[test]
    fn error_frames_carry_utf8_descriptions() {
        let (mut producer, mut consumer) = pair();
        producer
            .send(&Frame::with_payload(
                FrameKind::Error,
                9,
                b"stage 1 copy failed".to_vec(),
            ))
            .unwrap();
        let frame = consumer.receive().unwrap();
        assert_eq!(frame.kind, FrameKind::Error);
        assert_eq!(frame.fence, 9);
        assert_eq!(
            std::str::from_utf8(&frame.payload).unwrap(),
            "stage 1 copy failed"
        );
    }

    #[test]
    fn consumer_failures_are_reported_to_the_producer() {
        let (producer, consumer) = pair();
        let mut producer = BoundaryProducer::new(producer);
        let mut consumer = BoundaryConsumer::new(consumer, 1);
        let reporting = std::thread::spawn(move || producer.link.receive().unwrap());
        consumer.fail("stage 1 copy failed").unwrap();
        let frame = reporting.join().unwrap();
        assert_eq!(frame.kind, FrameKind::Error);
        assert_eq!(
            std::str::from_utf8(&frame.payload).unwrap(),
            "stage 1 copy failed"
        );
    }

    #[test]
    fn slot_ring_reserves_and_releases_activation_slots() {
        let mut ring = SlotRing::new(2);
        assert_eq!(ring.reserve(4), Some(0));
        assert_eq!(ring.reserve(5), Some(1));
        assert_eq!(ring.reserve(4), None);
        assert_eq!(ring.reserve(6), None);
        assert_eq!(ring.free(), 0);
        assert!(ring.is_reserved(4));
        assert!(ring.release(4));
        assert!(!ring.release(4));
        assert_eq!(ring.reserve(6), Some(0));
        assert_eq!(ring.free(), 0);
        assert!(ring.release(5));
        assert!(ring.release(6));
        assert_eq!(ring.free(), 2);
    }

    #[test]
    fn producer_and_consumer_exchange_one_activation() {
        let (producer, consumer) = pair();
        let mut producer = BoundaryProducer::new(producer);
        let mut consumer = BoundaryConsumer::new(consumer, 2);
        let (done, received) = mpsc::channel();
        let publishing = std::thread::spawn(move || {
            producer.publish(7, b"activation").unwrap();
            producer.wait_release(7).unwrap();
            done.send(()).unwrap();
        });

        let mut payload = Vec::new();
        assert_eq!(
            consumer.receive_into(&mut payload).unwrap(),
            BoundaryEvent::Data(BoundaryData {
                fence: 7,
                slot: 0,
                nbytes: 10,
            })
        );
        assert_eq!(payload, b"activation");
        consumer.release(7).unwrap();
        received.recv_timeout(Duration::from_secs(2)).unwrap();
        publishing.join().unwrap();
    }

    #[test]
    fn cancelled_preparation_releases_the_consumer_slot() {
        let (producer, consumer) = pair();
        let mut producer = BoundaryProducer::new(producer);
        let mut consumer = BoundaryConsumer::new(consumer, 1);
        let cancelling = std::thread::spawn(move || {
            producer
                .link
                .send_parts(FrameKind::Prepare, 3, &[])
                .unwrap();
            let (kind, fence) = producer.link.receive_into(&mut Vec::new()).unwrap();
            assert_eq!((kind, fence), (FrameKind::Ready, 3));
            producer.cancel(3).unwrap();
        });

        let mut payload = Vec::new();
        assert_eq!(
            consumer.receive_into(&mut payload).unwrap(),
            BoundaryEvent::Cancel { fence: 3 }
        );
        assert_eq!(consumer.slots.free(), 1);
        cancelling.join().unwrap();
    }

    #[test]
    fn prepare_payload_is_delivered_with_the_activation() {
        let (producer, consumer) = pair();
        let mut producer = BoundaryProducer::new(producer);
        let mut consumer = BoundaryConsumer::new(consumer, 1);
        let publishing = std::thread::spawn(move || {
            producer
                .link
                .send_parts(FrameKind::Prepare, 5, b"stage request")
                .unwrap();
            let (kind, fence) = producer.link.receive_into(&mut Vec::new()).unwrap();
            assert_eq!((kind, fence), (FrameKind::Ready, 5));
            producer
                .link
                .send_parts(FrameKind::Data, 5, b"activation")
                .unwrap();
        });

        let mut prepare = Vec::new();
        let mut payload = Vec::new();
        assert!(matches!(
            consumer
                .receive_prepared_into(&mut prepare, &mut payload)
                .unwrap(),
            BoundaryEvent::Data(BoundaryData { fence: 5, .. })
        ));
        assert_eq!(prepare, b"stage request");
        assert_eq!(payload, b"activation");
        consumer.release(5).unwrap();
        publishing.join().unwrap();
    }

    #[test]
    fn producer_publishes_a_prepare_payload_before_the_activation() {
        let (producer, consumer) = pair();
        let mut producer = BoundaryProducer::new(producer);
        let mut consumer = BoundaryConsumer::new(consumer, 1);
        let publishing = std::thread::spawn(move || {
            producer
                .publish_prepared(7, b"stage request", b"activation")
                .unwrap();
            producer.wait_release(7).unwrap();
        });

        let mut prepare = Vec::new();
        let mut payload = Vec::new();
        assert!(matches!(
            consumer
                .receive_prepared_into(&mut prepare, &mut payload)
                .unwrap(),
            BoundaryEvent::Data(BoundaryData { fence: 7, .. })
        ));
        assert_eq!(prepare, b"stage request");
        assert_eq!(payload, b"activation");
        consumer.release(7).unwrap();
        publishing.join().unwrap();
    }

    #[test]
    fn final_stage_commits_a_completion_payload() {
        let (producer, consumer) = pair();
        let mut producer = BoundaryProducer::new(producer);
        let mut consumer = BoundaryConsumer::new(consumer, 1);
        let publishing = std::thread::spawn(move || {
            producer
                .publish_prepared(9, b"request", b"activation")
                .unwrap();
            let mut payload = Vec::new();
            producer.wait_commit(9, &mut payload).unwrap();
            payload
        });

        let mut prepare = Vec::new();
        let mut activation = Vec::new();
        assert!(matches!(
            consumer
                .receive_prepared_into(&mut prepare, &mut activation)
                .unwrap(),
            BoundaryEvent::Data(BoundaryData { fence: 9, .. })
        ));
        consumer.commit_with(9, b"completion").unwrap();
        assert_eq!(publishing.join().unwrap(), b"completion");
    }

    #[test]
    fn full_ring_reports_an_error_without_stealing_the_reserved_slot() {
        let (producer, consumer) = pair();
        let mut producer = BoundaryProducer::new(producer);
        let mut consumer = BoundaryConsumer::new(consumer, 1);
        let (reported, result) = mpsc::channel();
        let publishing = std::thread::spawn(move || {
            producer
                .link
                .send_parts(FrameKind::Prepare, 1, &[])
                .unwrap();
            let (kind, fence) = producer.link.receive_into(&mut Vec::new()).unwrap();
            assert_eq!((kind, fence), (FrameKind::Ready, 1));
            producer
                .link
                .send_parts(FrameKind::Data, 1, b"activation")
                .unwrap();
            producer
                .link
                .send_parts(FrameKind::Prepare, 2, &[])
                .unwrap();
            let error = producer.link.receive().unwrap();
            reported
                .send((error.kind, error.fence, error.payload))
                .unwrap();
        });

        let mut payload = Vec::new();
        assert!(matches!(
            consumer.receive_into(&mut payload).unwrap(),
            BoundaryEvent::Data(BoundaryData { fence: 1, .. })
        ));
        assert_eq!(
            consumer.receive_into(&mut payload).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(consumer.slots.free(), 0);
        consumer.release(1).unwrap();
        let (kind, fence, message) = result.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!((kind, fence), (FrameKind::Error, 2));
        assert_eq!(
            std::str::from_utf8(&message).unwrap(),
            "pipeline boundary ring is full at fence 2"
        );
        publishing.join().unwrap();
    }

    #[test]
    fn drain_is_reported_to_the_consumer() {
        let (producer, consumer) = pair();
        let mut producer = BoundaryProducer::new(producer);
        let mut consumer = BoundaryConsumer::new(consumer, 1);
        let draining = std::thread::spawn(move || producer.drain().unwrap());
        let mut payload = Vec::new();
        assert_eq!(
            consumer.receive_into(&mut payload).unwrap(),
            BoundaryEvent::Drain
        );
        draining.join().unwrap();
    }

    #[test]
    fn malformed_headers_are_rejected() {
        let (mut producer, mut consumer) = pair();
        producer
            .stream
            .write_all(&[0, 0, 0, 0, 99, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        assert_eq!(
            consumer.receive().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );

        let (mut producer, mut consumer) = pair();
        consumer.set_max_payload(4);
        producer
            .stream
            .write_all(&[0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        assert_eq!(
            consumer.receive().unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
}
