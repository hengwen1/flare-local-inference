//! Control payload for one pipeline stage execution.
//!
//! Stage zero packs the scheduler's uploads into this record and sends it in a
//! boundary `PREPARE` frame. The boundary activation itself travels in the
//! following `DATA` frame, so large tensors never pass through this encoding.

use std::io;
use std::sync::mpsc;

use super::boundary::{BoundaryConsumer, BoundaryEvent, BoundaryLink, BoundaryProducer};
use super::{ExecutionManifest, FrameCompletion, RankExecutor, Upload};
use crate::{Error, ExecutionError, Result};

const VERSION: u32 = 1;
const MAX_UPLOADS: usize = 4096;
const MAX_NAME_BYTES: usize = 256;
const MAX_UPLOAD_BYTES: u64 = 64 << 20;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageUpload {
    pub buffer: String,
    pub offset: u64,
    pub nbytes: u64,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageRequest {
    pub graph: u32,
    pub uploads: Vec<StageUpload>,
}

impl StageRequest {
    pub fn new(graph: u32, uploads: Vec<StageUpload>) -> Self {
        Self { graph, uploads }
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let uploads = u32::try_from(self.uploads.len())
            .ok()
            .filter(|count| *count as usize <= MAX_UPLOADS)
            .ok_or_else(|| protocol_error("pipeline request has too many uploads"))?;
        let capacity = self.uploads.iter().try_fold(12usize, |total, upload| {
            total
                .checked_add(4)
                .and_then(|total| total.checked_add(upload.buffer.len()))
                .and_then(|total| total.checked_add(24))
                .and_then(|total| total.checked_add(upload.data.len()))
                .ok_or_else(|| protocol_error("pipeline request size overflow"))
        })?;
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(&VERSION.to_be_bytes());
        out.extend_from_slice(&self.graph.to_be_bytes());
        out.extend_from_slice(&uploads.to_be_bytes());
        for upload in &self.uploads {
            validate_upload(upload)?;
            let name_len = u32::try_from(upload.buffer.len())
                .map_err(|_| protocol_error("pipeline upload name is too long"))?;
            out.extend_from_slice(&name_len.to_be_bytes());
            out.extend_from_slice(upload.buffer.as_bytes());
            out.extend_from_slice(&upload.offset.to_be_bytes());
            out.extend_from_slice(&upload.nbytes.to_be_bytes());
            out.extend_from_slice(&(upload.data.len() as u64).to_be_bytes());
            out.extend_from_slice(&upload.data);
        }
        Ok(out)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader { bytes, offset: 0 };
        let version = reader.u32()?;
        if version != VERSION {
            return Err(protocol_error(
                "unsupported pipeline request version; re-export the model package",
            ));
        }
        let graph = reader.u32()?;
        let count = reader.u32()? as usize;
        if count > MAX_UPLOADS {
            return Err(protocol_error("pipeline request has too many uploads"));
        }
        let mut uploads = Vec::with_capacity(count);
        for _ in 0..count {
            let name_len = reader.u32()? as usize;
            if name_len == 0 || name_len > MAX_NAME_BYTES {
                return Err(protocol_error("pipeline upload name length is invalid"));
            }
            let buffer = std::str::from_utf8(reader.bytes(name_len)?)
                .map_err(|_| protocol_error("pipeline upload name is not UTF-8"))?
                .to_string();
            let offset = reader.u64()?;
            let nbytes = reader.u64()?;
            let data_len = reader.u64()?;
            if data_len > MAX_UPLOAD_BYTES {
                return Err(protocol_error("pipeline upload payload is too large"));
            }
            let data_len = usize::try_from(data_len)
                .map_err(|_| protocol_error("pipeline upload payload exceeds this host"))?;
            let data = reader.bytes(data_len)?.to_vec();
            let upload = StageUpload {
                buffer,
                offset,
                nbytes,
                data,
            };
            validate_upload(&upload)?;
            uploads.push(upload);
        }
        if reader.offset != bytes.len() {
            return Err(protocol_error("pipeline request has trailing bytes"));
        }
        Ok(Self { graph, uploads })
    }
}

/// One decoded stage event. The activation payload remains in [`StageWorker::payload`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StageEvent {
    Execute {
        fence: u64,
        slot: usize,
        nbytes: usize,
        request: StageRequest,
    },
    Cancel {
        fence: u64,
    },
    Drain,
}

/// Receives boundary frames and decodes the stage control payload.
pub struct StageWorker {
    input: BoundaryConsumer,
    prepare: Vec<u8>,
    payload: Vec<u8>,
}

impl StageWorker {
    pub fn new(input: BoundaryConsumer) -> Self {
        Self {
            input,
            prepare: Vec::new(),
            payload: Vec::new(),
        }
    }

    pub fn receive(&mut self) -> io::Result<StageEvent> {
        match self
            .input
            .receive_prepared_into(&mut self.prepare, &mut self.payload)?
        {
            BoundaryEvent::Data(data) => {
                let request = StageRequest::decode(&self.prepare).map_err(|error| {
                    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
                })?;
                Ok(StageEvent::Execute {
                    fence: data.fence,
                    slot: data.slot,
                    nbytes: data.nbytes,
                    request,
                })
            }
            BoundaryEvent::Cancel { fence } => Ok(StageEvent::Cancel { fence }),
            BoundaryEvent::Drain => Ok(StageEvent::Drain),
        }
    }

    /// The activation bytes belonging to the most recent `Execute` event.
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn release(&mut self, fence: u64) -> io::Result<()> {
        self.input.release(fence)
    }

    pub fn commit(&mut self, fence: u64) -> io::Result<()> {
        self.input.commit(fence)
    }
}

fn validate_upload(upload: &StageUpload) -> Result<()> {
    if upload.buffer.is_empty() || upload.buffer.len() > MAX_NAME_BYTES {
        return Err(protocol_error("pipeline upload name length is invalid"));
    }
    if upload.data.len() as u64 != upload.nbytes {
        return Err(protocol_error(
            "pipeline upload byte count differs from its payload",
        ));
    }
    if upload.nbytes > MAX_UPLOAD_BYTES {
        return Err(protocol_error("pipeline upload payload is too large"));
    }
    Ok(())
}

fn protocol_error(message: &str) -> Error {
    Error::execution(ExecutionError::State, message)
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, nbytes: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(nbytes)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| protocol_error("pipeline request is truncated"))?;
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(
            self.bytes(4)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(
            self.bytes(8)?.try_into().expect("eight bytes"),
        ))
    }
}

/// One completed frame reported by the final pipeline stage.
pub struct StageCompletion {
    pub fence: u64,
    pub completion: FrameCompletion,
}

fn boundary_error(error: impl std::fmt::Display) -> Error {
    Error::execution(ExecutionError::State, format!("pipeline boundary: {error}"))
}

fn boundary_bytes(boundary: &crate::model::contract::PipelineBoundary) -> Result<usize> {
    let element = match boundary.dtype.as_str() {
        "bf16" | "f16" => 2,
        "f32" => 4,
        other => {
            return Err(Error::execution(
                ExecutionError::State,
                format!("unsupported pipeline boundary dtype {other}"),
            ));
        }
    };
    boundary
        .rows
        .checked_mul(boundary.elements)
        .and_then(|elements| elements.checked_mul(element))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| {
            Error::execution(
                ExecutionError::State,
                "pipeline boundary geometry exceeds host address space",
            )
        })
}

/// Runs one non-first pipeline stage until the input boundary drains.
///
/// The stage uploads each activation into its declared boundary buffer, executes
/// the requested graph, and forwards the next boundary activation. The final
/// stage reports completions through `completions` instead of publishing.
///
/// # Safety
/// The manifest's prepared graphs, buffers, and CUDA context must remain valid
/// for the duration of the call. Stage execution is local; the manifest must not
/// declare tensor-parallel peers.
pub unsafe fn serve_stage(
    manifest: ExecutionManifest,
    input: BoundaryLink,
    input_boundary: crate::model::contract::PipelineBoundary,
    output: Option<(BoundaryLink, crate::model::contract::PipelineBoundary)>,
    completions: mpsc::Sender<StageCompletion>,
) -> Result<()> {
    let input_bytes = boundary_bytes(&input_boundary)?;
    let slots = usize::try_from(input_boundary.slots).map_err(|_| {
        Error::execution(
            ExecutionError::State,
            "pipeline boundary slot count exceeds host address space",
        )
    })?;
    if slots == 0 {
        return Err(Error::execution(
            ExecutionError::State,
            "pipeline boundary requires at least one activation slot",
        ));
    }
    let input_capacity = input_bytes.checked_mul(slots).ok_or_else(|| {
        Error::execution(
            ExecutionError::State,
            "pipeline boundary buffer size exceeds host address space",
        )
    })?;
    if manifest.buffer(&input_boundary.buffer)?.nbytes != input_capacity {
        return Err(Error::execution(
            ExecutionError::State,
            "pipeline boundary geometry differs from the prepared input buffer",
        ));
    }
    let output = output
        .map(|(link, boundary)| {
            boundary_bytes(&boundary).and_then(|bytes| {
                let capacity = bytes.checked_mul(slots).ok_or_else(|| {
                    Error::execution(
                        ExecutionError::State,
                        "pipeline boundary buffer size exceeds host address space",
                    )
                })?;
                if manifest.buffer(&boundary.buffer)?.nbytes != capacity {
                    return Err(Error::execution(
                        ExecutionError::State,
                        "pipeline boundary geometry differs from the prepared output buffer",
                    ));
                }
                if boundary.slots != input_boundary.slots {
                    return Err(Error::execution(
                        ExecutionError::State,
                        "adjacent pipeline boundaries disagree on activation slots",
                    ));
                }
                Ok((link, boundary, bytes))
            })
        })
        .transpose()?;

    let mut worker = StageWorker::new(BoundaryConsumer::new(input, slots));
    let mut executor = unsafe { RankExecutor::new(manifest, None) }?;
    let staging_bytes = output
        .as_ref()
        .map_or(input_bytes, |(_, _, bytes)| input_bytes.max(*bytes));
    let mut host = Vec::with_capacity(slots);
    for _ in 0..slots {
        host.push(executor.host_buffer(staging_bytes)?);
    }
    let mut output =
        output.map(|(link, boundary, bytes)| (BoundaryProducer::new(link), boundary, bytes));

    loop {
        match worker.receive().map_err(boundary_error)? {
            StageEvent::Execute {
                fence,
                slot,
                nbytes,
                request,
            } => {
                if nbytes != input_bytes {
                    return Err(Error::execution(
                        ExecutionError::State,
                        "pipeline activation size differs from the prepared boundary",
                    ));
                }
                let Some(buffer) = host.get_mut(slot) else {
                    return Err(Error::execution(
                        ExecutionError::State,
                        "pipeline activation slot exceeds the prepared boundary",
                    ));
                };
                buffer.bytes_mut()[..nbytes].copy_from_slice(worker.payload());

                let mut uploads = Vec::with_capacity(request.uploads.len() + 1);
                uploads.push(Upload {
                    buffer: input_boundary.buffer.as_str(),
                    offset: slot * input_bytes,
                    bytes: &buffer.bytes()[..nbytes],
                });
                for upload in &request.uploads {
                    let offset = usize::try_from(upload.offset).map_err(|_| {
                        Error::execution(
                            ExecutionError::State,
                            "pipeline upload offset exceeds host address space",
                        )
                    })?;
                    uploads.push(Upload {
                        buffer: upload.buffer.as_str(),
                        offset,
                        bytes: &upload.data,
                    });
                }
                let graph = usize::try_from(request.graph).map_err(|_| {
                    Error::execution(
                        ExecutionError::State,
                        "pipeline stage graph index exceeds host address space",
                    )
                })?;
                if executor.submit(graph, &uploads, false)?.is_none() {
                    return Err(Error::execution(
                        ExecutionError::State,
                        "pipeline stage executor reported backpressure after a completed frame",
                    ));
                }
                let completion = loop {
                    if let Some(completion) = executor.poll()? {
                        break completion;
                    }
                    std::thread::yield_now();
                };

                if let Some((producer, boundary, bytes)) = output.as_mut() {
                    let offset = slot.checked_mul(*bytes).ok_or_else(|| {
                        Error::execution(
                            ExecutionError::State,
                            "pipeline boundary slot offset exceeds host address space",
                        )
                    })?;
                    executor.read_buffer_range(&boundary.buffer, buffer, offset, *bytes)?;
                    producer
                        .publish(fence, &buffer.bytes()[..*bytes])
                        .map_err(boundary_error)?;
                    producer.wait_release(fence).map_err(boundary_error)?;
                } else {
                    completions
                        .send(StageCompletion { fence, completion })
                        .map_err(boundary_error)?;
                }
                worker.release(fence).map_err(boundary_error)?;
            }
            StageEvent::Cancel { .. } => {}
            StageEvent::Drain => break,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> StageRequest {
        StageRequest::new(
            3,
            vec![
                StageUpload {
                    buffer: "token_ids".into(),
                    offset: 8,
                    nbytes: 4,
                    data: vec![1, 2, 3, 4],
                },
                StageUpload {
                    buffer: "page_tables".into(),
                    offset: 0,
                    nbytes: 2,
                    data: vec![5, 6],
                },
            ],
        )
    }

    #[test]
    fn stage_requests_round_trip() {
        let value = request();
        assert_eq!(
            StageRequest::decode(&value.encode().unwrap()).unwrap(),
            value
        );
    }

    #[test]
    fn stage_requests_reject_truncation_and_trailing_bytes() {
        let encoded = request().encode().unwrap();
        for end in 0..encoded.len() {
            assert!(StageRequest::decode(&encoded[..end]).is_err());
        }
        let mut trailing = encoded;
        trailing.push(0);
        assert!(StageRequest::decode(&trailing).is_err());
    }

    #[test]
    fn stage_requests_reject_mismatched_upload_lengths() {
        let mut value = request();
        value.uploads[0].nbytes += 1;
        assert!(value.encode().is_err());

        let mut damaged = request().encode().unwrap();
        damaged[33..41].copy_from_slice(&5u64.to_be_bytes());
        assert!(StageRequest::decode(&damaged).is_err());
    }

    #[test]
    fn stage_requests_reject_unknown_versions() {
        let mut encoded = request().encode().unwrap();
        encoded[..4].copy_from_slice(&2u32.to_be_bytes());
        assert!(StageRequest::decode(&encoded).is_err());
    }

    #[test]
    fn boundary_geometry_matches_the_declared_dtype() {
        let boundary = crate::model::contract::PipelineBoundary {
            producer: 0,
            consumer: 1,
            buffer: "activation".into(),
            dtype: "bf16".into(),
            rows: 8,
            elements: 16,
            slots: 3,
        };
        assert_eq!(boundary_bytes(&boundary).unwrap(), 256);

        let mut half = boundary.clone();
        half.dtype = "f16".into();
        assert_eq!(boundary_bytes(&half).unwrap(), 256);

        let mut float = boundary;
        float.dtype = "f32".into();
        assert_eq!(boundary_bytes(&float).unwrap(), 512);
    }

    #[test]
    fn boundary_geometry_rejects_unknown_dtypes_and_overflow() {
        let mut boundary = crate::model::contract::PipelineBoundary {
            producer: 0,
            consumer: 1,
            buffer: "activation".into(),
            dtype: "fp8".into(),
            rows: 1,
            elements: 1,
            slots: 1,
        };
        assert!(boundary_bytes(&boundary).is_err());

        boundary.dtype = "f32".into();
        boundary.rows = u64::MAX;
        boundary.elements = 2;
        assert!(boundary_bytes(&boundary).is_err());
    }

    #[test]
    fn stage_worker_decodes_control_and_activation_payloads() {
        use super::super::boundary::{BoundaryConsumer, BoundaryLink, FrameKind};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || BoundaryLink::connect(address).unwrap());
        let server = BoundaryLink::accept(&listener).unwrap();
        let mut producer = client.join().unwrap();
        let mut worker = StageWorker::new(BoundaryConsumer::new(server, 1));
        let control = request().encode().unwrap();
        let publishing = std::thread::spawn(move || {
            producer
                .send_parts(FrameKind::Prepare, 11, &control)
                .unwrap();
            let (kind, fence) = producer.receive_into(&mut Vec::new()).unwrap();
            assert_eq!((kind, fence), (FrameKind::Ready, 11));
            producer
                .send_parts(FrameKind::Data, 11, b"activation")
                .unwrap();
            let (kind, fence) = producer.receive_into(&mut Vec::new()).unwrap();
            assert_eq!((kind, fence), (FrameKind::Release, 11));
        });

        match worker.receive().unwrap() {
            StageEvent::Execute {
                fence,
                slot,
                nbytes,
                request: decoded,
            } => {
                assert_eq!((fence, slot, nbytes), (11, 0, 10));
                assert_eq!(decoded, request());
            }
            event => panic!("unexpected stage event: {event:?}"),
        }
        assert_eq!(worker.payload(), b"activation");
        worker.release(11).unwrap();
        publishing.join().unwrap();
    }
}
