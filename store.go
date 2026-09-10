// The persistence port.

package mempher

import "context"

// Store is the persistence port: L0, its vector projection, and the retrieval
// primitives over both. The postgres subpackage is the one implementation.
//
// Three rules hold across every method. A Store never reads a clock: every
// timestamp arrives as an argument, which is what makes appends replayable. A
// Store never crosses a scope, and an episode addressed from the wrong scope is
// [ErrNotFound], not an authorization error. And a Store never updates an
// episode; there is deliberately no method that could.
//
// Removing one is the single exception, and it has exactly one method, so that
// the whole of erasure is one thing to find, review and audit. See [Store.Forget].
//
// Retrieval is split per channel rather than exposed as one fused query so that
// [Memory.Recall] can run the channels concurrently and gain a channel later
// without rewriting a single large statement.
type Store interface {
	// Append inserts one episode and the jobs it implies, atomically,
	// creating the scope if this is its first episode.
	Append(ctx context.Context, cmd AppendCommand) (AppendResult, error)

	// Episode returns one episode by id within a scope, or [ErrNotFound].
	Episode(ctx context.Context, scope ScopeID, id EpisodeID) (Episode, error)

	// Replay returns up to limit episodes of a scope in Seq order, starting
	// after afterSeq. It is how a projection is rebuilt from L0.
	Replay(ctx context.Context, scope ScopeID, afterSeq int64, limit int) ([]Episode, error)

	// SearchSemantic returns the nearest episodes by cosine distance over one
	// model's encodings, best first.
	SearchSemantic(ctx context.Context, q SemanticQuery) ([]Candidate, error)

	// SearchLexical returns the best full-text matches, best first.
	SearchLexical(ctx context.Context, q LexicalQuery) ([]Candidate, error)

	// PutEncoding writes one encoding, replacing any existing one for the
	// same episode and model, which is what makes encode jobs safe to retry.
	// It fills the encoding's scope and ingest time from the episode row, and
	// returns [ErrNotFound] if that episode does not exist.
	PutEncoding(ctx context.Context, enc Encoding) error

	// PendingEncodings returns ids of episodes with no encoding for a model,
	// in Seq order.
	PendingEncodings(ctx context.Context, q PendingEncodings) ([]EpisodeID, error)

	// Scopes lists the partitions of L0 in id order, so that work spanning
	// all of them can be walked one scope at a time.
	//
	// It is the catalogue every other method assumes: Seq is only ordered
	// within a scope, so paging through the whole log means paging through
	// each scope in turn, and nothing else can say what the scopes are.
	Scopes(ctx context.Context, q ScopeQuery) ([]Scope, error)

	// Forget erases episodes and every row this store holds that derives
	// from them, and reports what went.
	//
	// It must be atomic, in the way [Store.Append] must: a half-erased scope
	// is not a smaller erasure, it is a failed one. An implementation
	// therefore removes the facts, extraction markers, encodings and jobs in
	// the same transaction as the episodes, whichever of its tables they
	// live in -- and a deployment that keeps L1 in a genuinely separate
	// backend cannot get that from this method and must erase there too.
	//
	// Erasing what is not there is not an error. A scope that never existed,
	// or an episode id it never held, removes nothing and says so.
	Forget(ctx context.Context, req ForgetRequest) (ForgetResult, error)
}
