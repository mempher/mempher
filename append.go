// The write loop: what a caller asks for, and what the store applies atomically.

package mempher

import "time"

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

// AppendCommand is the unit a [Store] applies atomically: one episode plus the
// jobs it implies.
//
// They commit together because they must. An episode written without its encode
// job would be permanently invisible to the semantic channel, and a crash
// between two statements is exactly when that happens.
type AppendCommand struct {
	// Episode is the episode to insert.
	Episode NewEpisode
	// Jobs are enqueued in the same transaction. Their Episodes field is
	// filled in by the [Store] with the id it just minted.
	Jobs []NewJob
}

// AppendResult is what an [AppendCommand] produced.
type AppendResult struct {
	// Episode is the episode as persisted, with its minted ID and Seq.
	Episode Episode
	// Jobs are the jobs enqueued alongside it.
	Jobs []Job
}
