// Package mempher provides long-term memory for AI agents, backed by
// PostgreSQL 18 and pgvector. It needs no service beyond the database.
//
// # The invariant
//
// Episodes (L0) are an immutable, append-only log and the only source of truth.
// Everything else is a projection derived from them: vector encodings now,
// extracted facts and their validity windows next. Any projection can be
// dropped and rebuilt by replaying L0 through the same deriving code. So an
// episode is never updated or deleted, and only a worker draining the job queue
// writes a projection. That is what makes a bad extractor recoverable.
//
// # Three loops
//
// Write is the hot path: validate, insert one episode, enqueue the work it
// implies, return. No model calls.
//
// Consolidate is offline, batched, idempotent and replayable, and every model
// call lives there. See [Worker].
//
// Read runs its channels in parallel, fuses their rankings with reciprocal rank
// fusion, cuts the result to a token budget, and returns it with the provenance
// of every hit. Its only model call embeds the query.
//
// # The layers
//
// L0 is the episode log: [Memory.Append], [Memory.Recall] over raw content, and
// encode jobs that project it into vectors.
//
// L1 is the facts extracted from it, each with the event-time window over which
// it holds and the episodes it was read from. It is opt-in: configure an
// [Extractor] and appends enqueue [JobKindExtract] alongside the encode job, a
// [Worker] drains it, and [Recollection.Facts] carries the result. Leave it nil
// and none of that happens.
//
// The two layers are not peers, and recall keeps them apart. Episodes are ranked
// against the query and fused; facts are what is true about the scope, returned
// whole and budgeted first.
//
// Forgetting and procedural memory are not implemented.
package mempher
