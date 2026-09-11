// L1: facts extracted from L0, and the port that produces them.

package mempher

import (
	"context"
	"time"
)

// Length limits on the parts of a fact, enforced here and again by CHECK
// constraints.
const (
	// MaxSubjectLen is the longest permitted [Subject], in bytes.
	MaxSubjectLen = 200
	// MaxPredicateLen is the longest permitted [Predicate], in bytes.
	MaxPredicateLen = 100
	// MaxObjectLen is the longest permitted fact object, in bytes.
	//
	// It is smaller than [MaxStatementLen] because the object is part of a
	// fact's temporal primary key and the statement is not. That key is a
	// GiST index tuple, which has a hard size limit, so scope, extractor,
	// subject, predicate and object are sized to fit inside it together with
	// room to spare. Raising this without redoing that arithmetic turns a
	// long object into an insert that fails on the index rather than on a
	// CHECK.
	MaxObjectLen = 1000
	// MaxStatementLen is the longest permitted fact statement, in bytes.
	MaxStatementLen = 4000
	// MaxAssertionsPerExtraction caps how many facts one extraction may
	// assert, so a runaway model cannot fill a scope in a single job.
	MaxAssertionsPerExtraction = 64
	// MaxRetractionsPerExtraction caps how many facts one extraction may
	// close, for the same reason.
	MaxRetractionsPerExtraction = 64
)

// Validity is the event-time window over which a fact holds: when the thing it
// asserts became true in the world, and when it stopped.
//
// It is event time, never system time. "Lived in Paris from 2019 to 2024" is a
// validity window; "we learned this on Tuesday" is [Fact.AssertedAt]. Conflating
// them is the usual way a memory store starts answering questions about the past
// with what it believes today.
//
// The window is half-open, [From, To): a fact valid to noon and one valid from
// noon do not overlap, so a change of state needs no gap and no arbitrary
// tie-break.
type Validity struct {
	// From is when the fact became true. Required: a fact with no beginning
	// cannot be placed on a timeline, and "as long as we have known it" is
	// expressible as the occurrence time of the episode it came from.
	From time.Time
	// To is when it stopped being true. Zero means it is still true, which is
	// the common case and the reason this is not a closed interval.
	To time.Time
}

// IsOpen reports whether the fact is still true, meaning To is unset.
func (v Validity) IsOpen() bool { return v.To.IsZero() }

// Contains reports whether the fact holds at instant t, treating the window as
// half-open.
func (v Validity) Contains(t time.Time) bool {
	if t.Before(v.From) {
		return false
	}
	return v.IsOpen() || t.Before(v.To)
}

// Overlaps reports whether two windows share any instant.
func (v Validity) Overlaps(other Validity) bool {
	if !v.IsOpen() && !other.From.Before(v.To) {
		return false
	}
	if !other.IsOpen() && !v.From.Before(other.To) {
		return false
	}
	return true
}

// Fact is one durable claim derived from L0: the atom of L1.
//
// A fact is a projection, not a source of truth. Every one of them can be
// dropped and rebuilt by replaying the episodes in [Fact.Episodes] through the
// same [Extractor], which is what makes a bad extractor recoverable rather than
// permanent.
//
// Facts are shaped as a triple with a rendered statement beside it. The triple
// is machine identity: it is what makes re-asserting the same claim an upsert
// instead of a duplicate, and what a retraction names. The statement is the
// payload: it is what goes in front of a model and what the lexical index sees.
// Neither serves the other's purpose well, which is why both are stored.
type Fact struct {
	// ID is the uuidv7 surrogate key, minted by the database. It is stable
	// across a change to Valid, so a retraction can name a fact whose window
	// is about to move.
	ID FactID
	// Scope is the partition this fact belongs to.
	Scope ScopeID
	// Extractor identifies what produced it. Facts are keyed by it, so
	// several extractors can coexist and changing one is a backfill.
	Extractor ExtractorID
	// Subject is what the fact is about.
	Subject Subject
	// Predicate is the relation it asserts.
	Predicate Predicate
	// Object is the value of that relation, as text.
	Object string
	// Statement is the fact rendered for a reader: what gets injected into a
	// prompt, and what full-text search matches against.
	Statement string
	// Valid is the event-time window over which the fact holds.
	Valid Validity
	// Confidence is how sure the extractor was, in [0,1].
	Confidence float32
	// Episodes are the episodes this was derived from, in Seq order: the
	// provenance of the claim, and the first thing asked of a fact that looks
	// wrong.
	Episodes []EpisodeID
	// AssertedAt is system time: when this library learned the fact, from the
	// [Clock]. Unlike Valid it says nothing about the world.
	AssertedAt time.Time
	// UpdatedAt is when the row last changed, from the [Clock]. A fact is
	// re-asserted or retracted, never rewritten, so this moves only when its
	// window or confidence does.
	UpdatedAt time.Time
}

// Assertion is a claim an [Extractor] wants recorded. It is the unvalidated,
// id-less form of [Fact]: the extractor knows what is true, and the [FactStore]
// decides whether that is new, a re-assertion, or a conflict.
type Assertion struct {
	// Subject is what the fact is about. Required.
	Subject Subject
	// Predicate is the relation it asserts. Required.
	Predicate Predicate
	// Object is the value of that relation. Required.
	Object string
	// Statement is the fact rendered for a reader. Required: a fact that
	// cannot be shown to a model is not worth storing.
	Statement string
	// Valid is the event-time window. From is required; a zero To means the
	// fact is still true.
	Valid Validity
	// Confidence is how sure the extractor is, in [0,1]. Zero is taken at
	// face value: it means no confidence, not "unset".
	Confidence float32
}

// Retraction closes an existing fact's validity window, recording that the world
// changed rather than that the extractor was wrong.
//
// It is not a delete, and there is deliberately no method that deletes a fact.
// "Moved to Berlin" must leave "lived in Paris until March" standing and
// answerable, because a question about last year still has a right answer.
//
// A mistaken fact is corrected the other way: drop the projection and replay L0
// through a fixed extractor.
type Retraction struct {
	// Fact is the fact to close, named by the id carried on the [Fact] the
	// extractor was given in [ExtractRequest.Known].
	Fact FactID
	// At is the event time from which the fact no longer holds. It must fall
	// at or after the fact's Valid.From, and closes the window as [From, At).
	At time.Time
}

// ExtractRequest is what an [Extractor] is given: new episodes, and what is
// already believed about the scope.
//
// Known is supplied rather than looked up because extraction must stay a pure
// function of its inputs. An extractor that queried the store itself could not
// be replayed, and replay is the only repair L1 has.
type ExtractRequest struct {
	// Scope is the partition being extracted. Never crossed.
	Scope ScopeID
	// Episodes are the new episodes to read, in Seq order and never empty.
	Episodes []Episode
	// Known are the facts currently believed about this scope, so the
	// extractor can recognise a re-assertion and can retract what the new
	// episodes contradict. It may be empty, and is capped by the caller.
	Known []Fact
	// Now is the current instant, from the [Clock]. It is what resolves the
	// relative dates that natural language is made of: "since last Tuesday"
	// has no meaning without it, and reading a wall clock inside the
	// extractor would make replay non-deterministic.
	Now time.Time
}

// ExtractResult is what an [Extractor] returns.
//
// An empty result is a valid and common answer. Most episodes carry no durable
// fact, and an extractor that always finds one is the failure mode this type is
// shaped to make visible.
type ExtractResult struct {
	// Assert are the claims to record.
	Assert []Assertion
	// Retract are the facts the episodes contradict, named from
	// [ExtractRequest.Known].
	Retract []Retraction
}

// Extractor turns episodes into facts. It is the second model-calling port after
// [Embedder], and like it, is only ever called from the consolidation loop:
// nothing on the write path waits for a model.
//
// Implementations must be safe for concurrent use and honour cancellation. They
// must not read a clock, query a [Store], or retain the episodes they are given.
//
// Prefer a deterministic one -- temperature zero, or as close as the provider
// allows. Reading the same episode twice is not a hypothetical: a reclaimed
// lease replays a job, and while the [FactStore] declines to apply an extraction
// of episodes it has already read, a model that answers differently the first
// time it sees a scope and the first time it re-derives one during a backfill
// will assert the same claim over a window that overlaps the old one, which is
// [ErrFactConflict] rather than a silently merged answer.
type Extractor interface {
	// Extract reads the episodes and returns what to assert and retract.
	Extract(ctx context.Context, req ExtractRequest) (ExtractResult, error)

	// Model identifies the extractor precisely enough that changing the
	// model, the prompt or the schema it emits changes this value. Facts are
	// keyed by it, so a change is a backfill rather than a silent mixture of
	// two extractors' output.
	Model() ExtractorID
}

// ExtractCommand is the unit a [FactStore] applies atomically: everything one
// extraction concluded, plus the record that those episodes were read.
//
// They commit together because they must. Facts written without the extraction
// marker would be re-derived on the next pass and re-asserted forever; a marker
// written without its facts would lose them permanently.
type ExtractCommand struct {
	// Scope is the partition to write to. Required.
	Scope ScopeID
	// Extractor is what produced this. Required.
	Extractor ExtractorID
	// Episodes are the episodes that were read, marked extracted whether or
	// not they yielded anything. Required and never empty.
	Episodes []EpisodeID
	// Assert are the claims to record, at most
	// [MaxAssertionsPerExtraction].
	Assert []Assertion
	// Retract are the windows to close, at most
	// [MaxRetractionsPerExtraction].
	Retract []Retraction
	// At is system time for the write, from the [Clock]. A [FactStore] never
	// reads a clock, which is what makes an extraction replayable.
	At time.Time
}

// FactQuery selects facts. It is a lookup rather than a ranked search: L1 is the
// compact, current summary of a scope, and the normal request is "everything
// believed about this scope", not "the ten nearest claims".
type FactQuery struct {
	// Scope restricts the query to one partition. Required.
	Scope ScopeID
	// Extractor selects whose facts to read. Required: mixing two
	// extractors' output would double every claim they agree on.
	Extractor ExtractorID
	// AsOf is the system-time cut: only facts asserted at or before it are
	// visible. It bounds what had been learned, not what was true.
	AsOf time.Time
	// At is the event-time instant the facts must hold at. Zero means now,
	// resolved by the caller's [Clock]. It bounds what was true, not what had
	// been learned.
	At time.Time
	// IncludeClosed keeps facts whose window has ended before At. It is what
	// asks "what was true then", and is off by default because the usual
	// question is "what is true now".
	IncludeClosed bool
	// Subjects, when non-empty, keeps only facts about one of them.
	Subjects []Subject
	// Predicates, when non-empty, keeps only facts asserting one of them.
	Predicates []Predicate
	// MinConfidence drops facts the extractor was less sure of than this.
	// Zero keeps everything.
	MinConfidence float32
	// Text, when non-empty, orders results by full-text relevance to it
	// rather than by subject and predicate. It decides which facts survive
	// Limit when a scope holds more than fit; it never excludes a fact that
	// the filters above admitted.
	Text string
	// Limit caps how many facts are returned. Zero means a store-chosen
	// default.
	Limit int
}

// FactStore is the persistence port for L1. It is separate from [Store] for the
// same reason [Queue] is: one backend normally serves all three, and the
// postgres subpackage does, but a deployment that runs no [Extractor] stores no
// facts and should not have to implement this to compile.
//
// The rules that hold across [Store] hold here too. A FactStore never reads a
// clock, never crosses a scope, and reports a fact addressed from the wrong
// scope as [ErrNotFound] rather than as an authorization error.
//
// It also never deletes. A fact stops applying by having its window closed, and
// a fact that should never have existed is removed by dropping the projection
// and replaying L0 -- not by a DELETE that leaves the log and its derivation
// disagreeing.
type FactStore interface {
	// ApplyExtraction records one extraction atomically: the assertions, the
	// retractions, and the markers saying those episodes were read.
	//
	// It is idempotent, because leasing is at-least-once. Re-asserting an
	// identical claim over an identical window refreshes it in place rather
	// than duplicating it, and re-closing an already-closed window at the
	// same instant does nothing. It returns the facts as persisted.
	ApplyExtraction(ctx context.Context, cmd ExtractCommand) ([]Fact, error)

	// Facts returns the facts matching q, ordered by subject and predicate,
	// or by relevance when [FactQuery.Text] is set.
	Facts(ctx context.Context, q FactQuery) ([]Fact, error)

	// Fact returns one fact by id within a scope, or [ErrNotFound].
	Fact(ctx context.Context, scope ScopeID, id FactID) (Fact, error)
}
