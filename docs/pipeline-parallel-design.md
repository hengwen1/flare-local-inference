# Pipeline parallelism on Jetson Thor

Flare can serve one prepared model across several Jetson Thor modules. Each
stage owns a contiguous decoder-layer range, its weights, and its cache state.
Stage zero owns the frontend, scheduler, embedding, and vision. The last stage
owns the final layers, vocabulary projection, sampling, and the MTP module.
Activations cross stages through prepared boundary buffers and TCP with
preallocated pinned staging.

The design is preparation-time: a package declares its stage layout, boundary
buffers, and per-stage graph catalogs. Serving performs no layer slicing,
runtime compilation, or steady-state allocation. A package is either
tensor-parallel or pipeline-parallel, never both.

## Scope

- One stage per host and GPU. No TP composition. `world = stages`.
- Jetson Thor (SM110) target, with unified host/device memory.
- Text, vision, and MTP speculative decoding.
- TCP boundary transport with BF16 activations and pinned staging.
- In-memory prefix checkpoints. Disk offload shards are deferred and rejected
  explicitly in this version.

## Topology

```text
global rank = stage
stage 0        frontend, scheduler, embedding, vision, layers [0, a)
stage 1..S-2   layers [a, b), local KV/recurrent state
stage S-1      final layers, norm, head, sampling, MTP module
```

Stage zero rank zero serves HTTP. Other stages run a stage-worker loop. The
package contains every stage bundle; stage zero holds the complete package for
distribution, and each host fetches its stage's payload.

## Package contract

The prepared model contract inside `ServingContract` carries a `PipelineContract`:

```proto
message ModelContract {
  // ... existing fields ...
  optional PipelineContract pipeline = 8;
}
message PipelineContract {
  uint32 version = 1;
  uint32 stages = 2;
  uint32 activation_slots = 3;
  uint32 embedding_stage = 4;
  uint32 sampling_stage = 5;
  uint32 mtp_stage = 6;
  repeated PipelineStage stage = 7;
  repeated PipelineBoundary boundary = 8;
}

message PipelineStage {
  uint32 index = 1;
  uint32 first_layer = 2;          // inclusive
  uint32 last_layer = 3;           // exclusive
  uint64 pages = 4;
  uint64 state_rows = 5;
  uint64 snapshots = 6;
  repeated GraphContract graphs = 7;
  CacheContract cache = 8;
  repeated string inputs = 9;
  repeated string outputs = 10;
  bool has_vision = 11;
  bool has_mtp = 12;
  bool has_head = 13;
}

message PipelineBoundary {
  uint32 producer = 1;
  uint32 consumer = 2;             // producer + 1
  string buffer = 3;
  string dtype = 4;                // "bf16", "f16", "f32"
  uint64 rows = 5;
  uint64 elements = 6;
  uint64 slots = 7;
}
```

Prepared model contract version 3 denotes a pipeline package. Version 2
packages remain tensor-parallel or single-stage packages and require no
re-export. Execution ABI 4 remains valid: pipeline stages use the same prepared
graph and buffer manifest as other packages, so no execution ABI bump is needed.

Stage layer ranges cover the decoder exactly: the first stage starts at zero,
each stage starts where the previous ended, and every stage has at least one
layer. Stage zero owns embedding and vision. The last stage owns the head,
sampling, and MTP. Global page, state-row, and snapshot capacity is the minimum
across stages; page and row identifiers remain global while each stage owns its
local storage.

## Preparation

The plan declares the stage count and partition. Automatic partitioning uses
recipe-declared per-layer costs; an explicit `--stage-layers` value is recorded
in the plan. Per-stage reports include weights, state rows, pages, graphs,
scratch, boundary and pinned staging, and headroom. Stage zero accounts for
embedding, vision, and the image feature arena. The last stage accounts for the
head, MTP module, draft head, verification scratch, and completion buffers.

Recipes load only their assigned layers. The partial forward composes the
assigned layers and writes or reads the declared boundary buffer. Unassigned
layers remain meta and are never captured. MTP and vision preparation on SM110
are part of this work; missing SM110 components are added in Ignition rather
than replaced by a fallback.

## Runtime

Stage zero runs the existing server, scheduler, and worker with a pipeline
executor. Remote stages run a stage worker that loads its bundle, applies
uploads, executes its local graph, and forwards the boundary buffer to the next
stage.

Control frames carry job preparation, readiness, commit, cancellation, and
shutdown. A separate data connection carries boundary payloads between
neighboring stages. The last stage returns the completion frame to stage zero.
Boundary buffers form a preallocated ring; a producer cannot reuse a slot until
the consumer releases it after its graph consumed the data. Device-to-pinned,
TCP, pinned-to-device copies use prepared streams and events.

The first boundary implementation is a framed full-duplex TCP link with
`PREPARE`, `READY`, `DATA`, `RELEASE`, `COMMIT`, `CANCEL`, `DRAIN`, and `ERROR`
frames. A producer publishes only after the consumer acknowledges a reserved
slot, and the consumer receives activation bytes directly into caller-owned
pinned storage. `SlotRing` tracks `activation_slots` reservations independently
of completion ordering. A `PREPARE` payload carries the stage graph index and
small uploads; the activation itself follows in `DATA`.

Package loading selects the bundle for this host's cluster rank and exposes its
stage view as the prepared model contract. `NativeModel` retains the global
pipeline layout so the stage worker can validate stage roles and boundary
geometry. Disk prefix-cache offload is rejected for pipeline packages; in-memory
checkpoints remain in scope.

A stage worker receives an activation into pinned storage, uploads it to the
declared boundary buffer at the reserved slot, executes the requested stage
graph, and publishes the next boundary activation. The final stage commits an
encoded frame completion upstream instead of publishing. The completion carries
records, timing, logprobs, proposal confidence, and speculative survival, and
stage zero validates its fence before publishing the request result. This loop
performs no allocation after startup and keeps each stage's prepared buffers and
context alive until drain.

Stage zero remains the only scheduler and packer. It constructs per-stage input
packing from each stage's graph contract and execution manifest, then sends
buffer-name/offset/byte uploads. Remote stages do not run the scheduler.

## MTP

The complete `SPECULATIVE` operation remains one scheduler-visible job. Its
stage segments are:

```text
last stage      PROPOSAL  MTP feedback + draft layer + draft head
                     |
                     v  draft token identifiers
stage 0..S-2    TARGET    target decoder layers over verification rows
last stage      TARGET+COMMIT  final layers, norm, head, target distribution,
                               verification, accepted-state commit, completion,
                               MTP absorption
```

The last stage owns the MTP module, draft head, target residual, draft cache,
and verification state. The proposal phase uses the previous round's saved
target hidden. Stage zero uploads the returned draft identifiers into its
candidate buffer before packing verification rows. The scheduler still selects
one speculative operation and validates one completion; proposal, target, and
commit are internal segments, not independently scheduled operations.

Round `j`'s proposal depends on round `j-1`'s target hidden, so same-request
rounds serialize at the last stage. Independent requests may interleave when
their proposals are ready. The plan balances the last stage's head/MTP cost by
assigning it fewer layers when necessary. Measure proposal, target, commit, and
last-stage idle time before claiming speculative throughput.

## Vision

Vision encoding, image preparation, the feature arena, and image-token spans
stay on stage zero. `VISION` graphs run locally; image features never cross a
stage boundary. Stage zero prefill consumes the features and produces the first
boundary activation. Stage zero's memory plan includes vision weights and
scratch. Multimodal prefix identity and snapshot semantics are unchanged.

## Scheduling

The scheduler gains an in-flight window of `activation_slots`, carried as
`PolicyConfig::max_inflight` and set from the package's pipeline contract:

- Prefill chunks with fixed positions may fill the window.
- Independent decode or speculative batches for different requests may fill it.
- Same-request decode whose pending token is unknown, grammar-constrained
  steps, host stops, state operations, and phase transitions wait for the
  oldest completion.
- Prefill stops at the first required snapshot endpoint; the scheduler drains,
  captures the checkpoint, then continues.
- Completions are delivered in fence order, preserving existing validation.

## Cache and checkpoints

Global page, row, and snapshot identifiers are used by every stage. Snapshot,
restore, reset, and initialize operations fan out to every stage. Publication
waits for every stage's copy fence. Restore validates on every stage before any
stage commits; a rejection suppresses the commit everywhere. Snapshot-slot
reuse is acknowledged by every stage before reuse.

Prefix checkpoints remain in memory in this version. Pipeline packages reject
disk offload configuration explicitly; per-stage offload shards are a later
change.

## Thor constraints

Thor shares one DRAM pool between CPU and GPU. The plan includes boundary device
buffers and pinned staging in the per-stage memory budget. SM110 component
paths apply; stream overlap remains unavailable. The existing 1024-token prefill
chunk fits Thor memory and is the initial serving choice. TCP boundary payloads
are small relative to prefill compute; measure before making throughput claims.

Any stage failure is fatal to the model. Stage zero aborts in-flight jobs, fails
affected requests, stops admission, drains, and exits nonzero. No retry or
degraded execution path exists.

## Validation

- Contract and package tests cover stage coverage, boundary declarations,
  capacity agreement, version rejection, and mixed TP/PP rejection.
- CPU pipeline tests cover ordering, credits, completion routing, cancellation,
  stage failure, and shutdown.
- Two-Thor integration covers text, then vision, then MTP; token/logit
  comparisons against single-stage execution; prefix hits; snapshots;
  cancellation; stage failure; and no steady-state allocation.
- Performance reports cover prefill throughput, decode latency and throughput,
  MTP acceptance, per-stage busy time, activation bytes, and pinned-staging
  memory.
