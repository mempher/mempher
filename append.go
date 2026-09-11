// The write loop: what a caller asks for, and what the store applies atomically.

package mempher

import "time"

// MaxAppendBatch is the most episodes [Memory.AppendBatch] takes at once.
//
// It is bounded by what happens next rather than by what the statement could
// manage. A batch becomes one encode job, and a worker satisfies that job with
// one call to [Embedder.EmbedDocuments], so the batch size is a batch size an
// embedding provider has to accept. Sixty-four is inside every current one, and
// it bounds how long an append holds the scope row that orders the log.
const MaxAppendBatch = 64

// AppendRequest describes one episode to record. It is the caller-facing form of
// [NewEpisode]: timestamps may be left zero and are then taken from the [Clock].
type AppendRequest struct {
	// Scope is the partition to append to, created on first use. Required.
	Scope ScopeID
	// Content is the episode text. Required, at most [MaxContentLen] bytes.
	Content string
	// Role is the kind of participant this came from. Required.
	Role Role
	// Actor is who produced it. Optional.
	Actor Actor
	// Source is the system that delivered it. Required.
	Source Source
	// OccurredAt is when the thing happened. Zero means now.
	OccurredAt time.Time
	// Binding is the context to bind. Optional; copied, not retained.
	Binding Binding
}

// AppendCommand is the unit a [Store] applies atomically: episodes plus the jobs
// they imply.
//
// They commit together because they must. An episode written without its encode
// job would be permanently invisible to the semantic channel, and a crash
// between two statements is exactly when that happens.
//
// It carries a slice rather than one episode so that there is one way to write
// to L0 and not two. [Memory.Append] sends a command of length one and
// [Memory.AppendBatch] sends a longer one; a [Store] sees no difference, which
// is what keeps the two entry points from ever disagreeing about sequence
// allocation, job enqueueing or atomicity.
type AppendCommand struct {
	// Episodes are the episodes to insert, in the order they should be
	// sequenced. Required and never empty; every one of them names the same
	// scope.
	Episodes []NewEpisode
	// Jobs are enqueued in the same transaction. Their Episodes field is
	// filled in by the [Store] with the ids it just minted -- all of them, so
	// a batch of episodes yields one job per kind rather than one per
	// episode. That is what lets a [Worker] embed the whole batch in one
	// model call.
	Jobs []NewJob
}

// AppendResult is what an [AppendCommand] produced.
type AppendResult struct {
	// Episodes are the episodes as persisted, with their minted IDs and
	// Seqs, in the order they were given.
	Episodes []Episode
	// Jobs are the jobs enqueued alongside them.
	Jobs []Job
}
