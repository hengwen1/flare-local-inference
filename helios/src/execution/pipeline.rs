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

    pub fn commit_with(&mut self, fence: u64, payload: &[u8]) -> io::Result<()> {
        self.input.commit_with(fence, payload)
    }

    pub fn fail(&mut self, message: &str) -> io::Result<()> {
        self.input.fail(message)
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

/// Finds the remote stage graph with the same prepared name as the local graph.
///
/// Preparation gives every stage the same graph names for operations that cross
/// a boundary; stages may declare additional local graphs.
pub fn select_stage_graph(
    stage: &crate::model::contract::PipelineStage,
    name: &str,
) -> Result<u32> {
    stage
        .graphs
        .iter()
        .position(|graph| graph.graph == name)
        .and_then(|index| u32::try_from(index).ok())
        .ok_or_else(|| {
            protocol_error(&format!(
                "pipeline stage {} has no graph named {name}",
                stage.index
            ))
        })
}

/// Copies the uploads a remote stage declares into a control payload.
///
/// Stage zero packs every local input once; this filter keeps the uploads whose
/// buffer names the next stage declares and drops stage-local-only inputs.
pub fn stage_request(
    graph: u32,
    uploads: &[Upload<'_>],
    remote_inputs: &[String],
) -> Result<StageRequest> {
    let mut selected = Vec::new();
    for upload in uploads {
        if !remote_inputs.iter().any(|name| name == upload.buffer) {
            continue;
        }
        selected.push(StageUpload {
            buffer: upload.buffer.to_owned(),
            offset: u64::try_from(upload.offset).map_err(|_| {
                protocol_error("pipeline upload offset exceeds the control payload")
            })?,
            nbytes: u64::try_from(upload.bytes.len()).map_err(|_| {
                protocol_error("pipeline upload byte count exceeds the control payload")
            })?,
            data: upload.bytes.to_vec(),
        });
    }
    Ok(StageRequest {
        graph,
        uploads: selected,
    })
}

/// One stage-zero job waiting for boundary publication.
pub struct StageJob {
    pub fence: u64,
    pub request: StageRequest,
    pub activation: Vec<u8>,
}

/// Serial publisher for one stage-zero boundary.
///
/// The scheduler hands over completed local frames; this worker publishes the
/// control payload and activation, then waits for the final stage's committed
/// completion. Stage zero polls `next_completion` from its normal worker loop.
pub struct StageDispatch {
    jobs: Option<mpsc::SyncSender<StageJob>>,
    completions: mpsc::Receiver<Result<FrameCompletion>>,
}

impl StageDispatch {
    pub fn start(mut producer: BoundaryProducer, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::execution(
                ExecutionError::State,
                "pipeline dispatch requires at least one activation slot",
            ));
        }
        let (jobs, receiver) = mpsc::sync_channel::<StageJob>(capacity);
        let (sender, completions) = mpsc::channel();
        std::thread::Builder::new()
            .name("flare-pipeline-boundary".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    let result = (|| {
                        let prepare = job.request.encode()?;
                        producer
                            .publish_prepared(job.fence, &prepare, &job.activation)
                            .map_err(boundary_error)?;
                        let mut payload = Vec::new();
                        producer
                            .wait_commit(job.fence, &mut payload)
                            .map_err(boundary_error)?;
                        super::stage_completion::decode(job.fence, &payload)
                    })();
                    let failed = result.is_err();
                    if sender.send(result).is_err() || failed {
                        break;
                    }
                }
                let _ = producer.drain();
            })
            .map_err(|error| {
                Error::execution(
                    ExecutionError::State,
                    format!("cannot start pipeline boundary worker: {error}"),
                )
            })?;
        Ok(Self {
            jobs: Some(jobs),
            completions,
        })
    }

    /// Queues one completed local frame without blocking the scheduler.
    pub fn submit(&self, job: StageJob) -> Result<()> {
        let jobs = self.jobs.as_ref().ok_or_else(|| {
            Error::execution(ExecutionError::Closed, "pipeline boundary worker stopped")
        })?;
        jobs.try_send(job).map_err(|error| match error {
            mpsc::TrySendError::Full(_) => {
                Error::execution(ExecutionError::Capacity, "pipeline boundary queue is full")
            }
            mpsc::TrySendError::Disconnected(_) => {
                Error::execution(ExecutionError::Closed, "pipeline boundary worker stopped")
            }
        })
    }

    /// Returns one final-stage completion if the boundary worker finished one.
    pub fn next_completion(&self) -> Option<Result<FrameCompletion>> {
        self.completions.try_recv().ok()
    }
}

impl Drop for StageDispatch {
    fn drop(&mut self) {
        self.jobs.take();
    }
}

fn boundary_error(error: impl std::fmt::Display) -> Error {
    Error::execution(ExecutionError::State, format!("pipeline boundary: {error}"))
}

pub(crate) fn boundary_bytes(boundary: &crate::model::contract::PipelineBoundary) -> Result<usize> {
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
/// stage commits the encoded frame completion upstream instead of publishing.
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

    let result = (|| -> Result<()> {
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
                        if upload.buffer == input_boundary.buffer {
                            return Err(Error::execution(
                                ExecutionError::State,
                                "pipeline request uploads its boundary activation twice",
                            ));
                        }
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
                        worker.release(fence).map_err(boundary_error)?;
                    } else {
                        let payload = super::stage_completion::encode(&completion)?;
                        worker
                            .commit_with(fence, &payload)
                            .map_err(boundary_error)?;
                    }
                }
                StageEvent::Cancel { .. } => {}
                StageEvent::Drain => break,
            }
        }
        Ok(())
    })();
    if let Err(error) = &result {
        let _ = worker.fail(&error.to_string());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair() -> (
        super::super::boundary::BoundaryLink,
        super::super::boundary::BoundaryLink,
    ) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || BoundaryLink::connect(address).unwrap());
        let server = BoundaryLink::accept(&listener).unwrap();
        (client.join().unwrap(), server)
    }

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

    fn completion(fence: u64) -> FrameCompletion {
        FrameCompletion {
            fence: crate::cache::Fence(fence),
            graph_index: 1,
            timing: crate::execution::ExecutionTiming::default(),
            records: [crate::execution::CompletionRecord {
                vector: Some(crate::execution::TokenVector::default()),
                ..Default::default()
            }]
            .into(),
            logprobs: None,
            prompt_logprobs: None,
            proposal_confidence: None,
            speculative_survival: None,
        }
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
    fn dropping_stage_dispatch_drains_the_boundary() {
        let (producer, consumer) = pair();
        let dispatch = StageDispatch::start(BoundaryProducer::new(producer), 1).unwrap();
        let mut consumer = BoundaryConsumer::new(consumer, 1);
        let draining = std::thread::spawn(move || {
            let mut payload = Vec::new();
            matches!(
                consumer.receive_into(&mut payload).unwrap(),
                BoundaryEvent::Drain
            )
        });
        drop(dispatch);
        assert!(draining.join().unwrap());
    }

    #[test]
    fn stage_dispatch_publishes_jobs_and_returns_committed_completions() {
        use std::net::TcpListener;
        use std::time::Duration;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || BoundaryLink::connect(address).unwrap());
        let server = BoundaryLink::accept(&listener).unwrap();
        let client = client.join().unwrap();
        let producer = BoundaryProducer::new(server);
        let mut consumer = BoundaryConsumer::new(client, 1);
        let dispatch = StageDispatch::start(producer, 1).unwrap();

        let payload = crate::execution::stage_completion::encode(&completion(3)).unwrap();
        let receiving = std::thread::spawn(move || {
            let mut prepare = Vec::new();
            let mut activation = Vec::new();
            assert!(matches!(
                consumer
                    .receive_prepared_into(&mut prepare, &mut activation)
                    .unwrap(),
                BoundaryEvent::Data(crate::execution::boundary::BoundaryData { fence: 3, .. })
            ));
            assert_eq!(activation, [7, 8]);
            consumer.commit_with(3, &payload).unwrap();
        });

        dispatch
            .submit(StageJob {
                fence: 3,
                request: StageRequest::new(4, vec![]),
                activation: vec![7, 8],
            })
            .unwrap();
        let decoded = loop {
            if let Some(result) = dispatch.next_completion() {
                break result.unwrap();
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(decoded.fence, crate::cache::Fence(3));
        assert_eq!(decoded.graph_index, 1);
        receiving.join().unwrap();
    }

    #[test]
    fn stage_dispatch_returns_completions_in_job_order() {
        use std::time::Duration;

        let (producer, consumer) = pair();
        let dispatch = StageDispatch::start(BoundaryProducer::new(producer), 2).unwrap();
        let mut consumer = BoundaryConsumer::new(consumer, 2);
        let committing = std::thread::spawn(move || {
            for fence in [4, 5] {
                let mut prepare = Vec::new();
                let mut activation = Vec::new();
                let BoundaryEvent::Data(data) = consumer
                    .receive_prepared_into(&mut prepare, &mut activation)
                    .unwrap()
                else {
                    panic!("expected activation data");
                };
                assert_eq!(data.fence, fence);
                let payload =
                    crate::execution::stage_completion::encode(&completion(fence)).unwrap();
                consumer.commit_with(fence, &payload).unwrap();
            }
        });
        for fence in [4, 5] {
            dispatch
                .submit(StageJob {
                    fence,
                    request: StageRequest::new(4, vec![]),
                    activation: vec![7, 8],
                })
                .unwrap();
        }
        let mut fences = Vec::new();
        while fences.len() < 2 {
            if let Some(result) = dispatch.next_completion() {
                fences.push(result.unwrap().fence.0);
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        assert_eq!(fences, [4, 5]);
        committing.join().unwrap();
    }

    #[test]
    fn remote_stage_graphs_are_selected_by_prepared_name() {
        let stage = crate::model::contract::PipelineStage {
            index: 1,
            first_layer: 8,
            last_layer: 16,
            pages: 1,
            state_rows: 1,
            snapshots: 1,
            graphs: vec![
                crate::model::contract::GraphContract {
                    graph: "prefill".into(),
                    kind: crate::model::contract::GraphOperation::Prefill,
                    inputs: vec![],
                    state: vec![],
                    state_rows_per_request: 0,
                    draft_cursor: None,
                    embedding_tables: vec![],
                    pending_token: None,
                },
                crate::model::contract::GraphContract {
                    graph: "decode".into(),
                    kind: crate::model::contract::GraphOperation::Decode,
                    inputs: vec![],
                    state: vec![],
                    state_rows_per_request: 0,
                    draft_cursor: None,
                    embedding_tables: vec![],
                    pending_token: None,
                },
            ],
            cache: crate::model::contract::CacheContract {
                page_tokens: 16,
                regions: vec![],
                page_tables: vec![],
            },
            inputs: vec![],
            outputs: vec![],
            has_vision: false,
            has_mtp: false,
            has_head: false,
        };
        assert_eq!(select_stage_graph(&stage, "decode").unwrap(), 1);
        assert!(select_stage_graph(&stage, "vision").is_err());
    }

    #[test]
    fn stage_requests_copy_only_remote_inputs() {
        let uploads = [
            Upload {
                buffer: "positions",
                offset: 16,
                bytes: &[1, 2, 3, 4],
            },
            Upload {
                buffer: "stage_zero_only",
                offset: 0,
                bytes: &[9],
            },
        ];
        let request = stage_request(4, &uploads, &["positions".into()]).unwrap();
        assert_eq!(request.graph, 4);
        assert_eq!(
            request.uploads,
            vec![StageUpload {
                buffer: "positions".into(),
                offset: 16,
                nbytes: 4,
                data: vec![1, 2, 3, 4],
            }]
        );
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
