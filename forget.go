// Erasure: the one way content leaves L0.

package mempher

import "context"

// MaxForgetEpisodes caps how many episodes one erasure may name.
//
// It bounds the arrays this turns into, and it is a cap rather than a stream
// because erasing is one transaction: a request that would not fit in one is a
// request to erase the scope, or a loop the caller writes and can report on.
const MaxForgetEpisodes = 1000

// ForgetRequest asks for content to be erased: removed from the database, not
// superseded, not hidden.
//
// It is the answer to a request from the person the memory is about, and it is
// the only operation in this library that destroys anything. Everything else
// that sounds like forgetting is not this. A claim the world has moved past has
// its validity window closed and stays answerable; an episode that was
// misunderstood is corrected by appending another one and replaying the
// projection. Both of those are remembering accurately. This is the other thing.
//
// Erasing is idempotent: a scope that is already gone, or an episode id that
// this scope never held, erases nothing and reports so rather than failing. A
// compliance operation that could not be retried safely would be worse than
// useless.
type ForgetRequest struct {
	// Scope is the partition to erase from. Required, and never crossed:
	// erasure is bounded by the same wall as everything else.
	Scope ScopeID
	// Episodes narrows the erasure to those episodes. Empty erases the whole
	// scope, which is what "delete my account" means and the shape most
	// requests take.
	//
	// Naming episodes leaves the scope standing, with holes in its Seq where
	// they were. The scope's last_seq does not move, so no later episode ever
	// reuses one of their numbers.
	Episodes []EpisodeID
}

// ForgetResult reports what an erasure destroyed, table by table.
//
// It is counted rather than merely acknowledged because this is the operation
// whose effect nobody can go back and check: the evidence is what was removed.
type ForgetResult struct {
	// Scope is the partition that was erased from.
	Scope ScopeID
	// Episodes is how many episodes were removed from L0.
	Episodes int
	// Encodings is how many vectors went with them.
	Encodings int
	// Facts is how many facts were removed because an episode they were read
	// from is gone.
	Facts int
	// Extractions is how many extraction markers went, which is what stops a
	// worker from re-reading an episode that no longer exists.
	Extractions int
	// Jobs is how many queued jobs were dropped for naming an erased episode.
	Jobs int
	// ScopeRemoved reports whether the partition itself is gone, which is
	// true only of a whole-scope erasure. A scope id appended to afterwards
	// begins a new log at Seq 1.
	ScopeRemoved bool
}

// Forget erases episodes and everything derived from them.
//
// This is the one operation that breaks the append-only invariant, and it does
// so under one narrow relaxation: rows may be removed from L0 inside a
// transaction that has explicitly declared an erasure. An episode is still never
// updated, and never truncated. A row that exists is still exactly what was
// recorded, which is the half of the invariant worth keeping: erasure removes
// history, it does not rewrite it.
//
// Projections go with it, and they go for good rather than being closed or
// marked. A fact is a claim read out of episodes, so a fact whose provenance
// includes an erased episode is deleted outright: leaving it would leave the
// content standing in the layer that gets put in front of a model, which is
// exactly what was asked to be removed. If the episodes that remain still
// support the claim, the next extraction re-derives it -- which is the
// projection contract paying for itself, and the reason this is safe.
//
// The order is what makes an interrupted erasure recoverable: projections are
// removed before the episodes they came from, so a failure part-way leaves
// derived data missing and its source intact, which re-running repairs. The
// reverse would leave a claim about someone whose episodes are gone, and nothing
// to notice it by. In the one-store deployment the whole erasure is a single
// transaction and the question does not arise.
//
// It makes no model call and does not touch the queue beyond dropping the jobs
// that named an erased episode. A worker already holding one of those jobs when
// the erasure commits finds the episode gone, fails the job, and finds the job
// gone too; it logs both and carries on, which is the right outcome for work
// whose subject no longer exists.
func (m *Memory) Forget(ctx context.Context, req ForgetRequest) (ForgetResult, error) {
	if err := req.Validate(); err != nil {
		return ForgetResult{}, err
	}
	return m.store.Forget(ctx, req)
}
