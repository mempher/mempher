package mempher_test

import (
	"errors"
	"testing"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/memphertest"
)

// Erasure through the public API, against a real PostgreSQL 18. It is the one
// operation whose effect nobody can go back and check, so every test here counts
// rows rather than trusting the result it was handed.

// allergy is a rule that finds one durable fact, so a scope has an L1 to erase.
var allergy = memphertest.Rule{
	Match: "hazelnuts",
	Assert: mempher.Assertion{
		Subject: "user", Predicate: "allergic_to", Object: "hazelnuts",
		Statement:  "the user is allergic to hazelnuts",
		Confidence: 0.95,
	},
}

// scopeRows is every table's count for one scope, read in one round trip so a
// test asserts on a consistent picture.
type scopeRows struct {
	Episodes    int
	Encodings   int
	Facts       int
	Extractions int
	Jobs        int
	Catalogue   int
}

func (f *fixture) rowsFor(t *testing.T, scope mempher.ScopeID) scopeRows {
	t.Helper()
	var r scopeRows
	if err := f.pool.QueryRow(t.Context(), `
		SELECT
		    (SELECT count(*) FROM mempher.episodes          WHERE scope_id = $1),
		    (SELECT count(*) FROM mempher.episode_encodings WHERE scope_id = $1),
		    (SELECT count(*) FROM mempher.facts             WHERE scope_id = $1),
		    (SELECT count(*) FROM mempher.fact_extractions  WHERE scope_id = $1),
		    (SELECT count(*) FROM mempher.jobs              WHERE scope_id = $1),
		    (SELECT count(*) FROM mempher.scopes            WHERE id       = $1)`,
		string(scope),
	).Scan(&r.Episodes, &r.Encodings, &r.Facts, &r.Extractions, &r.Jobs, &r.Catalogue); err != nil {
		t.Fatalf("count rows for %q: %v", scope, err)
	}
	return r
}

func TestForgetErasesAScopeCompletely(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergy)
	ctx := t.Context()

	f.append(t, "user:1", "I am allergic to hazelnuts")
	f.append(t, "user:1", "small talk")
	f.append(t, "user:2", "I am allergic to hazelnuts")
	f.drain(t)

	before := f.rowsFor(t, "user:1")
	if before.Episodes != 2 || before.Encodings != 2 || before.Facts != 1 ||
		before.Extractions != 2 || before.Jobs == 0 || before.Catalogue != 1 {
		t.Fatalf("before erasure = %+v, want every table populated", before)
	}

	result, err := f.memory.Forget(ctx, mempher.ForgetRequest{Scope: "user:1"})
	if err != nil {
		t.Fatalf("Forget: %v", err)
	}
	switch {
	case result.Episodes != 2:
		t.Errorf("Episodes = %d, want 2", result.Episodes)
	case result.Encodings != 2:
		t.Errorf("Encodings = %d, want 2", result.Encodings)
	case result.Facts != 1:
		t.Errorf("Facts = %d, want 1", result.Facts)
	case result.Extractions != 2:
		t.Errorf("Extractions = %d, want 2", result.Extractions)
	case result.Jobs != before.Jobs:
		t.Errorf("Jobs = %d, want the %d the scope held", result.Jobs, before.Jobs)
	case !result.ScopeRemoved:
		t.Error("ScopeRemoved = false after erasing a whole scope")
	}

	if after := (f.rowsFor(t, "user:1")); after != (scopeRows{}) {
		t.Errorf("after erasure = %+v, want nothing left anywhere", after)
	}

	// The neighbour is untouched, which is the same wall every other
	// operation respects.
	if other := f.rowsFor(t, "user:2"); other.Episodes != 1 || other.Facts != 1 {
		t.Errorf("scope user:2 = %+v, want its own episode and fact intact", other)
	}

	// And the scope is gone from the catalogue, so a walk no longer visits it.
	scopes, err := f.store.Scopes(ctx, mempher.ScopeQuery{})
	if err != nil {
		t.Fatalf("Scopes: %v", err)
	}
	for _, scope := range scopes {
		if scope.ID == "user:1" {
			t.Error("the erased scope is still in the catalogue")
		}
	}
}

// TestForgetLeavesNothingToRecall is the check that matters to the person who
// asked: the content is not merely unreachable by id, it is not there.
func TestForgetLeavesNothingToRecall(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergy)
	ctx := t.Context()

	f.append(t, "user:1", "I am allergic to hazelnuts")
	f.drain(t)

	recalled, err := f.memory.Recall(ctx, mempher.RecallRequest{
		Scope: "user:1", Query: "any food allergies?",
	})
	if err != nil {
		t.Fatalf("Recall before erasure: %v", err)
	}
	if len(recalled.Episodes) == 0 || len(recalled.Facts) == 0 {
		t.Fatalf("recall before erasure returned %d episodes and %d facts, want both",
			len(recalled.Episodes), len(recalled.Facts))
	}

	if _, err := f.memory.Forget(ctx, mempher.ForgetRequest{Scope: "user:1"}); err != nil {
		t.Fatalf("Forget: %v", err)
	}

	// An erased scope is an empty memory, not a broken one: every channel
	// answers, and answers with nothing.
	recalled, err = f.memory.Recall(ctx, mempher.RecallRequest{
		Scope: "user:1", Query: "any food allergies?",
	})
	if err != nil {
		t.Fatalf("Recall after erasure: %v", err)
	}
	if len(recalled.Episodes) != 0 || len(recalled.Facts) != 0 {
		t.Errorf("recall after erasure returned %d episodes and %d facts, want none",
			len(recalled.Episodes), len(recalled.Facts))
	}
	for _, channel := range recalled.Channels {
		if channel.Err != nil {
			t.Errorf("channel %s failed after erasure: %v", channel.Channel, channel.Err)
		}
	}
}

func TestForgetOneEpisodeLeavesTheScopeStanding(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergy)
	ctx := t.Context()

	first := f.append(t, "user:1", "I am allergic to hazelnuts")
	second := f.append(t, "user:1", "I live in Paris")
	f.drain(t)

	result, err := f.memory.Forget(ctx, mempher.ForgetRequest{
		Scope:    "user:1",
		Episodes: []mempher.EpisodeID{first.ID},
	})
	if err != nil {
		t.Fatalf("Forget: %v", err)
	}
	switch {
	case result.Episodes != 1:
		t.Errorf("Episodes = %d, want 1", result.Episodes)
	case result.Encodings != 1:
		t.Errorf("Encodings = %d, want the one vector that episode had", result.Encodings)
	case result.Facts != 1:
		t.Errorf("Facts = %d, want the fact read out of it", result.Facts)
	case result.ScopeRemoved:
		t.Error("ScopeRemoved = true after erasing one episode of two")
	}

	left := f.rowsFor(t, "user:1")
	if left.Episodes != 1 || left.Catalogue != 1 {
		t.Fatalf("after erasure = %+v, want the other episode and the scope", left)
	}

	// The survivor is untouched, and keeps the Seq it was given.
	got, err := f.store.Episode(ctx, "user:1", second.ID)
	if err != nil {
		t.Fatalf("Episode: %v", err)
	}
	if got.Seq != second.Seq {
		t.Errorf("Seq = %d, want the %d it was allocated", got.Seq, second.Seq)
	}
	if _, err := f.store.Episode(ctx, "user:1", first.ID); !errors.Is(err, mempher.ErrNotFound) {
		t.Errorf("reading the erased episode err = %v, want mempher.ErrNotFound", err)
	}

	// Its number is not reused: the log continues past the hole.
	next := f.append(t, "user:1", "and another thing")
	if next.Seq <= second.Seq {
		t.Errorf("the next episode took Seq %d, want one past %d", next.Seq, second.Seq)
	}
}

// TestForgetTakesAFactItOnlyPartlySupports is the consequence worth stating: a
// fact read from two episodes goes when either of them does, because the claim
// can no longer be shown to come from what remains. If it still does, the next
// extraction says so.
func TestForgetTakesAFactItOnlyPartlySupports(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergy)
	ctx := t.Context()

	first := f.append(t, "user:1", "I am allergic to hazelnuts")
	f.append(t, "user:1", "and to walnuts, come to think of it")
	f.drain(t)

	facts, err := f.store.Facts(ctx, mempher.FactQuery{
		Scope: "user:1", Extractor: f.extractor.Model(),
		AsOf: f.clock.Now(), At: f.clock.Now(),
	})
	if err != nil {
		t.Fatalf("Facts: %v", err)
	}
	if len(facts) != 1 {
		t.Fatalf("%d facts before erasure, want 1", len(facts))
	}

	result, err := f.memory.Forget(ctx, mempher.ForgetRequest{
		Scope:    "user:1",
		Episodes: []mempher.EpisodeID{first.ID},
	})
	if err != nil {
		t.Fatalf("Forget: %v", err)
	}
	if result.Facts != 1 {
		t.Errorf("Facts = %d, want the claim its provenance named", result.Facts)
	}
	if left := f.rowsFor(t, "user:1"); left.Facts != 0 {
		t.Errorf("%d facts left, want none standing on an erased episode", left.Facts)
	}
}

func TestForgetIsIdempotentAndForgiving(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "an episode")
	f.drain(t)

	if _, err := f.memory.Forget(ctx, mempher.ForgetRequest{Scope: "user:1"}); err != nil {
		t.Fatalf("Forget: %v", err)
	}

	// A retried erasure is the normal case, not an edge one.
	again, err := f.memory.Forget(ctx, mempher.ForgetRequest{Scope: "user:1"})
	if err != nil {
		t.Fatalf("Forget again: %v", err)
	}
	if again != (mempher.ForgetResult{Scope: "user:1"}) {
		t.Errorf("erasing an erased scope = %+v, want nothing removed", again)
	}

	// A scope that never existed is the same answer.
	never, err := f.memory.Forget(ctx, mempher.ForgetRequest{Scope: "user:nobody"})
	if err != nil {
		t.Fatalf("Forget an unknown scope: %v", err)
	}
	if never.Episodes != 0 || never.ScopeRemoved {
		t.Errorf("erasing an unknown scope = %+v, want nothing removed", never)
	}
}

// TestForgetNeverCrossesAScope covers the wall every other operation respects:
// an episode named from the wrong scope is not erased, and saying so is not an
// error.
func TestForgetNeverCrossesAScope(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "an episode")
	theirs := f.append(t, "user:2", "someone else's episode")

	result, err := f.memory.Forget(ctx, mempher.ForgetRequest{
		Scope:    "user:1",
		Episodes: []mempher.EpisodeID{theirs.ID},
	})
	if err != nil {
		t.Fatalf("Forget: %v", err)
	}
	if result.Episodes != 0 {
		t.Errorf("Episodes = %d, want nothing: that episode is not in this scope", result.Episodes)
	}
	if other := f.rowsFor(t, "user:2"); other.Episodes != 1 {
		t.Error("an episode was erased from a scope the request did not name")
	}
}

// TestForgetLetsTheScopeIdBeUsedAgain: the id is a name, not a tombstone.
func TestForgetLetsTheScopeIdBeUsedAgain(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "the first life")
	f.append(t, "user:1", "still the first life")
	if _, err := f.memory.Forget(ctx, mempher.ForgetRequest{Scope: "user:1"}); err != nil {
		t.Fatalf("Forget: %v", err)
	}

	fresh := f.append(t, "user:1", "a new log")
	if fresh.Seq != 1 {
		t.Errorf("the new log starts at Seq %d, want 1", fresh.Seq)
	}
}

func TestForgetRejectsBadRequests(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	if _, err := f.memory.Forget(ctx, mempher.ForgetRequest{}); !errors.Is(err, mempher.ErrInvalidScope) {
		t.Errorf("erasing with no scope err = %v, want mempher.ErrInvalidScope", err)
	}
	if _, err := f.memory.Forget(ctx, mempher.ForgetRequest{
		Scope:    "user:1",
		Episodes: []mempher.EpisodeID{{}},
	}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("erasing an unset episode id err = %v, want mempher.ErrInvalidConfig", err)
	}
	too := make([]mempher.EpisodeID, mempher.MaxForgetEpisodes+1)
	for i := range too {
		too[i] = mempher.EpisodeID{1}
	}
	if _, err := f.memory.Forget(ctx, mempher.ForgetRequest{
		Scope: "user:1", Episodes: too,
	}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("erasing more than the cap err = %v, want mempher.ErrInvalidConfig", err)
	}
}
