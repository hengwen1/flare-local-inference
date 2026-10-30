//! Completion payload for the final pipeline stage.
//!
//! The final stage owns sampling and produces the frame completion. This codec
//! carries that completion back to stage zero over the reverse boundary link
//! without Python or host-side model code.

use super::{
    CompletionRecord, ExecutionTiming, FrameCompletion, MAX_OUTPUT_TOKENS, MAX_READINESS_INTERVALS,
    MAX_TOP_LOGPROBS, PromptLogprobs, REQUEST_LANES, ReadinessIntervals, TokenLogprobs,
    VECTOR_RECORD_FIELDS,
};
use crate::{Error, ExecutionError, Result, cache::Fence};

const VERSION: u8 = 1;
const MAX_LOGPROB_ROWS: usize = REQUEST_LANES;

fn invalid(message: impl Into<String>) -> Error {
    Error::execution(ExecutionError::Completion, message)
}

fn usize_of(value: u64) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| invalid("stage completion extent exceeds host address space"))
}

fn u32_of(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid("stage completion extent exceeds the wire format"))
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self {
            bytes: Vec::with_capacity(256),
        }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| invalid("stage completion payload is truncated"))?;
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }
}

fn write_logprobs(out: &mut Writer, value: &TokenLogprobs) -> Result<()> {
    if value.top.len() > MAX_TOP_LOGPROBS {
        return Err(invalid(
            "stage completion logprob prefix exceeds its capacity",
        ));
    }
    out.u32(value.token);
    out.f32(value.logprob);
    out.u16(value.top.len() as u16);
    for (token, logprob) in &value.top {
        out.u32(*token);
        out.f32(*logprob);
    }
    Ok(())
}

fn read_logprobs(input: &mut Reader<'_>) -> Result<TokenLogprobs> {
    let token = input.u32()?;
    let logprob = input.f32()?;
    let count = input.u16()? as usize;
    if count > MAX_TOP_LOGPROBS {
        return Err(invalid(
            "stage completion logprob prefix exceeds its capacity",
        ));
    }
    let mut top = Vec::with_capacity(count);
    for _ in 0..count {
        top.push((input.u32()?, input.f32()?));
    }
    Ok(TokenLogprobs {
        token,
        logprob,
        top,
    })
}

/// Encodes one final-stage completion for its `COMMIT` payload.
pub fn encode(completion: &FrameCompletion) -> Result<Vec<u8>> {
    if completion.records.len() > REQUEST_LANES {
        return Err(invalid(
            "stage completion record count exceeds its capacity",
        ));
    }
    let mut out = Writer::new();
    out.u8(VERSION);
    out.u64(completion.fence.0);
    out.u32(u32_of(completion.graph_index)?);
    out.u8(completion.records.len() as u8);
    for record in completion.records.iter() {
        let fields = record.fields();
        if fields.len() != VECTOR_RECORD_FIELDS {
            return Err(invalid(
                "stage completion record width differs from its ABI",
            ));
        }
        for field in fields {
            out.i64(field);
        }
    }

    out.u8(u8::from(completion.timing.gpu_ns[0].is_some()));
    out.u64(completion.timing.gpu_ns[0].unwrap_or(0));
    out.u64(completion.timing.wall_ns);
    out.u8(u8::from(completion.timing.queued));
    match &completion.timing.readiness {
        Some(readiness) => {
            out.u8(1);
            out.u16(readiness.pairs().len() as u16);
            for pair in readiness.pairs() {
                out.u64(pair[0]);
                out.u64(pair[1]);
            }
        }
        None => out.u8(0),
    }

    match &completion.logprobs {
        Some(rows) => {
            if rows.len() > MAX_LOGPROB_ROWS {
                return Err(invalid(
                    "stage completion logprob rows exceed their capacity",
                ));
            }
            out.u8(1);
            out.u8(rows.len() as u8);
            for row in rows {
                for value in row {
                    match value {
                        Some(value) => {
                            out.u8(1);
                            write_logprobs(&mut out, value)?;
                        }
                        None => out.u8(0),
                    }
                }
            }
        }
        None => out.u8(0),
    }

    match &completion.prompt_logprobs {
        Some(rows) => {
            if rows.len() > MAX_LOGPROB_ROWS {
                return Err(invalid(
                    "stage completion prompt rows exceed their capacity",
                ));
            }
            out.u8(1);
            out.u32(u32_of(rows.len())?);
            for value in rows {
                match value {
                    Some(value) => {
                        out.u8(1);
                        write_logprobs(&mut out, &value.value)?;
                        out.u32(value.rank);
                    }
                    None => out.u8(0),
                }
            }
        }
        None => out.u8(0),
    }

    for (present, rows) in [
        (
            completion.proposal_confidence.is_some(),
            completion.proposal_confidence.as_ref(),
        ),
        (
            completion.speculative_survival.is_some(),
            completion.speculative_survival.as_ref(),
        ),
    ] {
        out.u8(u8::from(present));
        if let Some(rows) = rows {
            for value in rows.iter().flatten() {
                out.f32(*value);
            }
        }
    }
    Ok(out.bytes)
}

/// Decodes one final-stage completion and validates its fence.
pub fn decode(expected_fence: u64, bytes: &[u8]) -> Result<FrameCompletion> {
    let mut input = Reader::new(bytes);
    if input.u8()? != VERSION {
        return Err(invalid("unsupported stage completion version"));
    }
    let fence = Fence(input.u64()?);
    if fence.0 != expected_fence {
        return Err(invalid("stage completion fence differs from its frame"));
    }
    let graph_index = usize_of(u64::from(input.u32()?))?;
    let count = input.u8()? as usize;
    if !(1..=REQUEST_LANES).contains(&count) {
        return Err(invalid("stage completion record count is invalid"));
    }
    let record_bytes = input.bytes(count * VECTOR_RECORD_FIELDS * 8)?;
    let records =
        super::transport::decode_records(record_bytes, fence, count, VECTOR_RECORD_FIELDS)?;

    let gpu_present = input.u8()?;
    if gpu_present > 1 {
        return Err(invalid("stage completion GPU timing presence is invalid"));
    }
    let gpu_ns = input.u64()?;
    let wall_ns = input.u64()?;
    let queued = input.u8()?;
    if queued > 1 {
        return Err(invalid("stage completion queue flag is invalid"));
    }
    let readiness = match input.u8()? {
        0 => None,
        1 => {
            let count = input.u16()? as usize;
            if count > MAX_READINESS_INTERVALS {
                return Err(invalid("stage completion readiness count exceeds capacity"));
            }
            let mut pairs = vec![[0u64; 2]; count];
            for pair in &mut pairs {
                pair[0] = input.u64()?;
                pair[1] = input.u64()?;
            }
            Some(ReadinessIntervals::from_pairs(&pairs)?)
        }
        _ => return Err(invalid("stage completion readiness presence is invalid")),
    };
    let timing = ExecutionTiming {
        gpu_ns: [
            (gpu_present == 1).then_some(gpu_ns),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        readiness,
        wall_ns,
        queued: queued == 1,
    };

    let logprobs = match input.u8()? {
        0 => None,
        1 => {
            let rows = input.u8()? as usize;
            if rows > MAX_LOGPROB_ROWS {
                return Err(invalid("stage completion logprob rows exceed capacity"));
            }
            let mut values = Vec::with_capacity(rows);
            for _ in 0..rows {
                let mut row = [const { None }; MAX_OUTPUT_TOKENS];
                for value in &mut row {
                    *value = match input.u8()? {
                        0 => None,
                        1 => Some(read_logprobs(&mut input)?),
                        _ => return Err(invalid("stage completion logprob presence is invalid")),
                    };
                }
                values.push(row);
            }
            Some(values)
        }
        _ => return Err(invalid("stage completion logprob presence is invalid")),
    };

    let prompt_logprobs = match input.u8()? {
        0 => None,
        1 => {
            let count = usize_of(u64::from(input.u32()?))?;
            if count > MAX_LOGPROB_ROWS {
                return Err(invalid("stage completion prompt rows exceed capacity"));
            }
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(match input.u8()? {
                    0 => None,
                    1 => Some(PromptLogprobs {
                        value: read_logprobs(&mut input)?,
                        rank: input.u32()?,
                    }),
                    _ => return Err(invalid("stage completion prompt presence is invalid")),
                });
            }
            Some(values)
        }
        _ => return Err(invalid("stage completion prompt presence is invalid")),
    };

    let proposal_confidence = match input.u8()? {
        0 => None,
        1 => {
            let mut rows = [[0.0f32; MAX_OUTPUT_TOKENS - 1]; REQUEST_LANES];
            for value in rows.iter_mut().flatten() {
                *value = input.f32()?;
            }
            Some(rows)
        }
        _ => return Err(invalid("stage completion confidence presence is invalid")),
    };
    let speculative_survival = match input.u8()? {
        0 => None,
        1 => {
            let mut rows = [[-1.0f32; MAX_OUTPUT_TOKENS - 1]; REQUEST_LANES];
            for value in rows.iter_mut().flatten() {
                *value = input.f32()?;
            }
            Some(rows)
        }
        _ => return Err(invalid("stage completion survival presence is invalid")),
    };
    if input.remaining() != 0 {
        return Err(invalid("stage completion payload has trailing bytes"));
    }
    Ok(FrameCompletion {
        fence,
        graph_index,
        timing,
        records,
        logprobs,
        prompt_logprobs,
        proposal_confidence,
        speculative_survival,
    })
}

#[cfg(test)]
mod tests {
    use super::super::transport::CompletionRecords;
    use super::*;
    use crate::cache::Fence;
    use crate::execution::TokenVector;

    fn record() -> CompletionRecord {
        CompletionRecord {
            generation: 3,
            submission_sequence: 7,
            consumed_tokens: 12,
            generated_tokens: 1,
            token_id: 42,
            status: 1,
            error_code: 0,
            reserved: 0,
            vector: Some(TokenVector {
                emitted_count: 1,
                offered_drafts: 0,
                accepted_drafts: 0,
                copy_offered_mask: 0,
                model_reserved: 0,
                draft_cache_consumed: -1,
                verification_width: 1,
                tokens: [42, -1, -1, -1, -1, -1, -1, -1],
            }),
        }
    }

    fn completion() -> FrameCompletion {
        let mut confidence = [[0.0f32; MAX_OUTPUT_TOKENS - 1]; REQUEST_LANES];
        confidence[0][0] = 0.75;
        let mut survival = [[-1.0f32; MAX_OUTPUT_TOKENS - 1]; REQUEST_LANES];
        survival[0][0] = 1.0;
        FrameCompletion {
            fence: Fence(7),
            graph_index: 3,
            timing: ExecutionTiming {
                gpu_ns: [
                    Some(11),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
                readiness: Some(ReadinessIntervals::from_pairs(&[[4, 2]]).unwrap()),
                wall_ns: 19,
                queued: true,
            },
            records: CompletionRecords::from([record()]),
            logprobs: Some(vec![[
                Some(TokenLogprobs {
                    token: 42,
                    logprob: -0.25,
                    top: vec![(42, -0.25), (7, -1.5)],
                }),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]]),
            prompt_logprobs: Some(vec![Some(PromptLogprobs {
                value: TokenLogprobs {
                    token: 9,
                    logprob: -0.5,
                    top: vec![(9, -0.5)],
                },
                rank: 2,
            })]),
            proposal_confidence: Some(confidence),
            speculative_survival: Some(survival),
        }
    }

    #[test]
    fn completions_round_trip_with_timing_and_logprobs() {
        let value = completion();
        let bytes = encode(&value).unwrap();
        let decoded = decode(7, &bytes).unwrap();
        assert_eq!(decoded.fence, value.fence);
        assert_eq!(decoded.graph_index, value.graph_index);
        assert_eq!(decoded.timing, value.timing);
        assert_eq!(&*decoded.records, &*value.records);
        assert_eq!(decoded.logprobs, value.logprobs);
        assert_eq!(decoded.prompt_logprobs, value.prompt_logprobs);
        assert_eq!(decoded.proposal_confidence, value.proposal_confidence);
        assert_eq!(decoded.speculative_survival, value.speculative_survival);
    }

    #[test]
    fn completions_reject_wrong_fences_and_trailing_bytes() {
        let mut bytes = encode(&completion()).unwrap();
        assert!(decode(8, &bytes).is_err());
        bytes.push(0);
        assert!(decode(7, &bytes).is_err());
    }

    #[test]
    fn completions_reject_unknown_versions_and_truncation() {
        let mut bytes = encode(&completion()).unwrap();
        bytes[0] = 2;
        assert!(decode(7, &bytes).is_err());
        let bytes = encode(&completion()).unwrap();
        assert!(decode(7, &bytes[..bytes.len() - 1]).is_err());
    }
}
