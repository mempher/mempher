package mempher_test

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/memphertest"
	"github.com/mempher/mempher/ops"
)

// Backfill through the public API, against a real PostgreSQL 18. What it is for
// cannot be faked: every answer it gives is an anti-join between L0 and a
// projection table, and a store that only pretended to hold them would be
// answering its own question.

// operator builds the ops handle these tests drive, over the fixture's own
// store and clock. Everything it needs is what a deployment would give it.
func (f *fixture) operator(t *testing.T, extractor mempher.Extractor) *ops.Ops {
	t.Helper()
	o, err := ops.New(ops.Config{
		Store:     f.store,
		Embedder:  f.embedder,
		Extractor: extractor,
		Clock:     f.clock,
	})
	if err != nil {
		t.Fatalf("ops.New: %v", err)
	}
	return o
}

// countEncodings reports how many vectors exist, which is what a dropped
// projection is measured by.
func (f *fixture) countEncodings(t *testing.T) int {
	t.Helper()
	var n int
	if err := f.pool.QueryRow(t.Context(),
		"SELECT count(*) FROM mempher.episode_encodings").Scan(&n); err != nil {
		t.Fatalf("count encodings: %v", err)
	}
	return n
}

// countJobs reports how many jobs the queue holds in total, which is how
// idempotence is checked without depending on a clock.
func (f *fixture) countJobs(t *testing.T) int {
	t.Helper()
	jobs, err := f.store.Jobs(t.Context(), ops.JobQuery{Limit: 1000})
	if err != nil {
		t.Fatalf("Jobs: %v", err)
	}
	return len(jobs)
}

// TestBackfillRebuildsADroppedProjection is the claim the whole design rests on:
// a projection can be thrown away and replayed out of L0.
func TestBackfillRebuildsADroppedProjection(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	for i := range 5 {
		f.append(t, "user:1", fmt.Sprintf("episode %d", i))
	}
	f.drain(t)
	if got := f.countEncodings(t); got != 5 {
		t.Fatalf("%d encodings after draining, want 5", got)
	}

	// The projection goes. L0 does not: the trigger would refuse.
	if _, err := f.pool.Exec(ctx, "TRUNCATE mempher.episode_encodings"); err != nil {
		t.Fatalf("truncate the encodings: %v", err)
	}

	result, err := f.operator(t, nil).Backfill(ctx, ops.BackfillRequest{})
	if err != nil {
		t.Fatalf("Backfill: %v", err)
	}
	switch {
	case result.Encode.Episodes != 5:
		t.Errorf("Encode.Episodes = %d, want 5", result.Encode.Episodes)
	case result.Encode.Jobs != 1:
		t.Errorf("Encode.Jobs = %d, want 1: five episodes fit in one default batch",
			result.Encode.Jobs)
	case result.Encode.Collapsed != 0:
		t.Errorf("Encode.Collapsed = %d, want 0 on a queue with no outstanding work",
			result.Encode.Collapsed)
	case !result.Done:
		t.Error("Done = false, want the walk to have finished")
	}
	// Nothing was extracted, because this worker has no Extractor.
	if result.Extract.Episodes != 0 {
		t.Errorf("Extract.Episodes = %d on a worker with no Extractor", result.Extract.Episodes)
	}

	f.drain(t)
	if got := f.countEncodings(t); got != 5 {
		t.Errorf("%d encodings after backfilling and draining, want all 5 back", got)
	}
}

// TestBackfillPicksUpAnExtractorConfiguredLater covers the second way work goes
// missing: the episodes arrived before there was anything to extract them.
func TestBackfillPicksUpAnExtractorConfiguredLater(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "I am allergic to hazelnuts")
	f.append(t, "user:1", "small talk")
	f.drain(t)

	// No extract job was ever enqueued: the memory that appended these had no
	// Extractor, so L1 was switched off.
	extractor := memphertest.NewExtractor().On(memphertest.Rule{
		Match: "hazelnuts",
		Assert: mempher.Assertion{
			Subject: "user", Predicate: "allergic_to", Object: "hazelnuts",
			Statement:  "the user is allergic to hazelnuts",
			Confidence: 0.95,
		},
	})
	worker, err := mempher.NewWorker(mempher.WorkerConfig{
		Store: f.store, Embedder: f.embedder, Extractor: extractor,
		Clock: f.clock, ID: "backfill-worker",
	})
	if err != nil {
		t.Fatalf("NewWorker: %v", err)
	}

	result, err := f.operator(t, extractor).Backfill(ctx, ops.BackfillRequest{
		Kinds: []mempher.JobKind{mempher.JobKindExtract},
	})
	if err != nil {
		t.Fatalf("Backfill: %v", err)
	}
	if result.Extract.Episodes != 2 {
		t.Fatalf("Extract.Episodes = %d, want both episodes", result.Extract.Episodes)
	}
	if result.Encode.Episodes != 0 {
		t.Errorf("Encode.Episodes = %d, want none: the kind filter excluded it",
			result.Encode.Episodes)
	}

	if _, err := worker.DrainOnce(ctx); err != nil {
		t.Fatalf("DrainOnce: %v", err)
	}
	facts, err := f.store.Facts(ctx, mempher.FactQuery{
		Scope:     "user:1",
		Extractor: extractor.Model(),
		AsOf:      f.clock.Now(),
		At:        f.clock.Now(),
	})
	if err != nil {
		t.Fatalf("Facts: %v", err)
	}
	if len(facts) != 1 || facts[0].Object != "hazelnuts" {
		t.Fatalf("facts = %+v, want the one the extractor should have found", facts)
	}
	// The fact was read out of an episode recorded long before the extractor
	// existed, which is the point.
	if len(facts[0].Episodes) == 0 {
		t.Error("the fact carries no provenance")
	}
}

// TestBackfillIsIdempotent is what makes it safe to run from a cron job.
func TestBackfillIsIdempotent(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	for i := range 3 {
		f.append(t, "user:1", fmt.Sprintf("episode %d", i))
	}
	// Deliberately not drained: the episodes stay pending, so the second run
	// sees exactly what the first did.
	before := f.countJobs(t)

	first, err := f.operator(t, nil).Backfill(ctx, ops.BackfillRequest{})
	if err != nil {
		t.Fatalf("Backfill: %v", err)
	}
	if first.Encode.Jobs != 1 {
		t.Fatalf("Encode.Jobs = %d, want 1", first.Encode.Jobs)
	}
	afterFirst := f.countJobs(t)
	if afterFirst != before+1 {
		t.Fatalf("the queue went from %d to %d jobs, want one more", before, afterFirst)
	}

	// A later call, so that "already existed" is distinguishable at all.
	f.clock.Advance(time.Minute)
	second, err := f.operator(t, nil).Backfill(ctx, ops.BackfillRequest{})
	if err != nil {
		t.Fatalf("Backfill: %v", err)
	}
	if got := f.countJobs(t); got != afterFirst {
		t.Errorf("the queue went from %d to %d jobs, want no new work", afterFirst, got)
	}
	if second.Encode.Collapsed != second.Encode.Jobs {
		t.Errorf("Collapsed = %d of %d jobs, want every one of them recognised as already queued",
			second.Encode.Collapsed, second.Encode.Jobs)
	}
}

// TestBackfillPagesThroughALongScope checks the cursor: the walk has to be
// resumable, because a scope can hold more episodes than one call should touch.
func TestBackfillPagesThroughALongScope(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	for i := range 7 {
		f.append(t, "user:1", fmt.Sprintf("episode %d", i))
	}

	req := ops.BackfillRequest{Limit: 3, BatchSize: 3}
	seen := 0
	for pages := 0; ; pages++ {
		if pages > 5 {
			t.Fatal("the walk did not finish in five pages")
		}
		result, err := f.operator(t, nil).Backfill(ctx, req)
		if err != nil {
			t.Fatalf("Backfill: %v", err)
		}
		seen += result.Encode.Episodes
		if result.Done {
			break
		}
		if result.NextScope != "user:1" {
			t.Fatalf("NextScope = %q, want the scope it stopped in", result.NextScope)
		}
		if result.NextSeq == 0 {
			t.Fatal("NextSeq = 0, which would restart the scope for ever")
		}
		req.Scope, req.AfterSeq = result.NextScope, result.NextSeq
	}
	if seen != 7 {
		t.Errorf("the walk covered %d episodes, want all 7", seen)
	}

	f.drain(t)
	if got := f.countEncodings(t); got != 7 {
		t.Errorf("%d encodings after draining, want 7", got)
	}
}

func TestBackfillWalksEveryScope(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	for _, scope := range []mempher.ScopeID{"user:1", "user:2", "team:9"} {
		f.append(t, scope, "an episode")
		f.append(t, scope, "another episode")
	}
	f.drain(t)
	if _, err := f.pool.Exec(ctx, "TRUNCATE mempher.episode_encodings"); err != nil {
		t.Fatalf("truncate the encodings: %v", err)
	}

	result, err := f.operator(t, nil).Backfill(ctx, ops.BackfillRequest{})
	if err != nil {
		t.Fatalf("Backfill: %v", err)
	}
	if result.Encode.Episodes != 6 {
		t.Errorf("Encode.Episodes = %d, want every scope's two episodes", result.Encode.Episodes)
	}
	if result.Encode.Jobs != 3 {
		t.Errorf("Encode.Jobs = %d, want one per scope: a job never crosses one",
			result.Encode.Jobs)
	}
	if !result.Done {
		t.Error("Done = false after walking off the end of the catalogue")
	}

	f.drain(t)
	if got := f.countEncodings(t); got != 6 {
		t.Errorf("%d encodings after draining, want 6", got)
	}
}

// TestBackfillLimitsItsScope covers the narrow request: one scope, and nobody
// else's work touched.
func TestBackfillLimitsItsScope(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "an episode")
	f.append(t, "user:2", "an episode")

	result, err := f.operator(t, nil).Backfill(ctx, ops.BackfillRequest{Scope: "user:1"})
	if err != nil {
		t.Fatalf("Backfill: %v", err)
	}
	if result.Encode.Episodes != 1 {
		t.Errorf("Encode.Episodes = %d, want only the one scope's episode", result.Encode.Episodes)
	}
	if !result.Done {
		t.Error("Done = false after exhausting the one scope asked for")
	}
}

func TestBackfillFindsNothingWhenTheQueueIsCaughtUp(t *testing.T) {
	t.Parallel()
	f := newFixture(t)

	f.append(t, "user:1", "an episode")
	f.drain(t)

	result, err := f.operator(t, nil).Backfill(t.Context(), ops.BackfillRequest{})
	if err != nil {
		t.Fatalf("Backfill: %v", err)
	}
	if result.Encode.Episodes != 0 || result.Encode.Jobs != 0 {
		t.Errorf("a caught-up queue backfilled %+v, want nothing", result.Encode)
	}
	if !result.Done {
		t.Error("Done = false with nothing left to do")
	}
}

func TestBackfillRejectsWorkThisWorkerCannotDo(t *testing.T) {
	t.Parallel()
	f := newFixture(t)

	_, err := f.operator(t, nil).Backfill(t.Context(), ops.BackfillRequest{
		Kinds: []mempher.JobKind{mempher.JobKindExtract},
	})
	if !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("backfilling extractions with no Extractor err = %v, want mempher.ErrInvalidConfig", err)
	}

	if _, err := f.operator(t, nil).Backfill(t.Context(), ops.BackfillRequest{
		AfterSeq: 4,
	}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("AfterSeq with no Scope err = %v, want mempher.ErrInvalidConfig", err)
	}
}
