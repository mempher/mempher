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
// Write is the hot path: validate, insert, enqueue the work it implies, return.
// No model calls. [Memory.AppendBatch] is the same loop for several episodes of
// one scope at once: they take a contiguous block of sequence numbers under one
// lock, share one round trip and one commit, and imply one job per kind rather
// than one per episode -- so the consolidation that follows is batched too.
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
// # Operating it
//
// Everything needed to run a deployment, and nothing needed to use one, lives in
// the ops subpackage: backfilling the work L0 implies but the queue has lost,
// reading and pruning the job queue, and walking the catalogue of scopes. It is
// a separate package so that this one stays the concept.
//
// # Forgetting
//
// Two different things go by that name, and only one of them destroys anything.
//
// A claim the world has moved past has its validity window closed and stays
// answerable, so "where did they live last year" still has an answer. That is
// not forgetting, it is remembering accurately, and nothing is removed for it.
//
// [Memory.Forget] is the other one: erasure, for when the person the memory is
// about asks for it. It removes episodes and every projection derived from them,
// a whole scope at a time or a named few episodes, in one transaction. It is the
// single exception to the append-only invariant, and a narrow one -- rows may be
// deleted from L0 inside a transaction that declares itself, while an episode is
// still never updated and never truncated. A row that exists is still exactly
// what was recorded.
//
// Facts standing on an erased episode are deleted rather than closed, because a
// closed window still says what it said. They are safe to delete for the reason
// every projection is: if the episodes that remain still support the claim, the
// next extraction re-derives it.
//
// Retention, decay and procedural memory are not implemented.
package mempher
