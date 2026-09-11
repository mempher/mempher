package mempher_test

import (
	"errors"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/internal/pgtest"
	"github.com/mempher/mempher/memphertest"
	"github.com/mempher/mempher/ops"
	"github.com/mempher/mempher/postgres"
)

// L1 through the public API, against a real PostgreSQL 18. The temporal
// constraints, the text ranking and the anti-joins are all the database's, so
// there is nothing here a fake store could stand in for.

// The event-time instants these tests reason about. They sit far from the
// fixture's system-time clock, so a test that confused the two axes fails rather
// than passing by coincidence.
var (
	moved  = time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC)
	before = time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
)

// factFixture is the fixture wired for L1: everything newFixture has, plus an
// extractor on both the memory and the worker.
type factFixture struct {
	*fixture
	extractor *memphertest.Extractor
}

func newFactFixture(t *testing.T, rules ...memphertest.Rule) *factFixture {
	t.Helper()

	pool := pgtest.Pool(t)
	embedder := memphertest.NewEmbedder(dims)
	extractor := memphertest.NewExtractor().On(rules...)

	if _, err := postgres.Migrate(t.Context(), pool, postgres.MigrateOptions{
		VectorDimensions: embedder.Dimensions(),
	}); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	store, err := postgres.New(t.Context(), pool)
	if err != nil {
		t.Fatalf("postgres.New: %v", err)
	}

	clock := memphertest.NewClock(start)
	memory, err := mempher.New(mempher.Config{
		Store: store, Embedder: embedder, Extractor: extractor, Clock: clock,
	})
	if err != nil {
		t.Fatalf("mempher.New: %v", err)
	}
	worker, err := mempher.NewWorker(mempher.WorkerConfig{
		Store: store, Embedder: embedder, Extractor: extractor,
		Clock: clock, ID: "test-worker",
	})
	if err != nil {
		t.Fatalf("mempher.NewWorker: %v", err)
	}
	return &factFixture{
		fixture:   &fixture{memory, worker, store, pool, clock, embedder},
		extractor: extractor,
	}
}

// livesIn is the rule the supersession tests move through time: a new home
// replaces the old one, and the old one stays answerable.
func livesIn(match, city string, from time.Time) memphertest.Rule {
	return memphertest.Rule{
		Match:    match,
		Replaces: true,
		Assert: mempher.Assertion{
			Subject: "user", Predicate: "lives_in", Object: city,
			Statement:  "the user lives in " + city,
			Valid:      mempher.Validity{From: from},
			Confidence: 0.9,
		},
	}
}

// allergicTo is a rule for a claim that can be true alongside others of its own
// predicate.
func allergicTo(food string) memphertest.Rule {
	return memphertest.Rule{
		Match: food,
		Assert: mempher.Assertion{
			Subject: "user", Predicate: "allergic_to", Object: food,
			Statement:  "the user is allergic to " + food,
			Valid:      mempher.Validity{From: before},
			Confidence: 1,
		},
	}
}

// statements renders a recollection's facts for an assertion message.
func statements(r mempher.Recollection) []string {
	out := make([]string, len(r.Facts))
	for i, f := range r.Facts {
		out[i] = f.Fact.Statement
	}
	return out
}

// report finds one channel's report, so a test can assert on degradation.
func report(t *testing.T, r mempher.Recollection, channel mempher.Channel) mempher.ChannelReport {
	t.Helper()
	for _, c := range r.Channels {
		if c.Channel == channel {
			return c
		}
	}
	t.Fatalf("no report for channel %q in %v", channel, r.Channels)
	return mempher.ChannelReport{}
}

// TestWriteLoopMakesNoExtractorCall is the promise the whole design rests on,
// now that a second and far more expensive model port exists.
func TestWriteLoopMakesNoExtractorCall(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergicTo("hazelnuts"))

	f.append(t, "user:1", "I am allergic to hazelnuts")
	if got := f.extractor.Calls(); got != 0 {
		t.Errorf("Append made %d extraction calls, want 0", got)
	}

	f.drain(t)
	if got := f.extractor.Calls(); got == 0 {
		t.Error("draining the queue made no extraction call")
	}
}

// TestFactsReachRecall is the loop closed: append, consolidate, recall.
func TestFactsReachRecall(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergicTo("hazelnuts"))

	episode := f.append(t, "user:1", "I am allergic to hazelnuts")
	f.drain(t)

	rec, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
		Scope: "user:1", Query: "what should I avoid eating?",
	})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(rec.Facts) != 1 {
		t.Fatalf("recalled facts = %v, want the hazelnut allergy", statements(rec))
	}

	fact := rec.Facts[0].Fact
	switch {
	case fact.Object != "hazelnuts":
		t.Errorf("Object = %q, want hazelnuts", fact.Object)
	case len(fact.Episodes) != 1 || fact.Episodes[0] != episode.ID:
		t.Errorf("Episodes = %v, want the episode it came from (%s)", fact.Episodes, episode.ID)
	case rec.Facts[0].Tokens == 0:
		t.Error("the fact was not costed")
	}
	// A fact is evidence of a different kind, so it arrives beside the
	// episodes rather than ranked against them.
	if len(rec.Episodes) == 0 {
		t.Error("the episode itself should still be recalled")
	}
	if got := report(t, rec, mempher.ChannelFact); got.Err != nil || got.Candidates != 1 {
		t.Errorf("fact channel reported %+v, want one candidate and no error", got)
	}
}

// TestSupersessionKeepsThePastAnswerable is what the temporal design is for,
// exercised end to end: the world changed, and last year still has an answer.
func TestSupersessionKeepsThePastAnswerable(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t,
		livesIn("live in paris", "Paris", before),
		livesIn("moved to berlin", "Berlin", moved),
	)

	f.append(t, "user:1", "I live in Paris")
	f.drain(t)

	f.clock.Advance(time.Hour)
	f.append(t, "user:1", "I moved to Berlin")
	f.drain(t)

	now, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
		Scope: "user:1", Query: "where does the user live?",
	})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(now.Facts) != 1 || now.Facts[0].Fact.Object != "Berlin" {
		t.Fatalf("currently believed = %v, want only Berlin", statements(now))
	}

	// The same question about a moment before the move. Nothing was deleted,
	// so it still has an answer.
	then, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
		Scope:      "user:1",
		Query:      "where does the user live?",
		OccurredTo: moved.Add(-24 * time.Hour),
	})
	if err != nil {
		t.Fatalf("Recall in the past: %v", err)
	}
	if len(then.Facts) != 1 || then.Facts[0].Fact.Object != "Paris" {
		t.Fatalf("believed before the move = %v, want only Paris", statements(then))
	}
	if got := then.Facts[0].Fact.Valid.To; !got.Equal(moved) {
		t.Errorf("the Paris window ends at %s, want %s", got, moved)
	}
}

// TestExtractionIsIdempotentAcrossDrains: a job that runs twice must leave the
// same scope behind, because leasing is at-least-once.
func TestExtractionIsIdempotentAcrossDrains(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergicTo("hazelnuts"))

	f.append(t, "user:1", "I am allergic to hazelnuts")
	f.drain(t)

	// Put the work back as a lost job would be recovered, and run it again.
	pending, err := f.store.PendingExtractions(t.Context(), ops.PendingExtractions{
		Extractor: f.extractor.Model(), Scope: "user:1",
	})
	if err != nil {
		t.Fatalf("PendingExtractions: %v", err)
	}
	if len(pending) != 0 {
		t.Fatalf("%d episodes are still pending extraction, want none", len(pending))
	}

	episodes, err := f.store.Replay(t.Context(), "user:1", 0, 10)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if _, err := f.store.Enqueue(t.Context(), mempher.NewJob{
		Kind: mempher.JobKindExtract, Scope: "user:1",
		Episodes: []mempher.EpisodeID{episodes[0].ID},
	}, f.clock.Now()); err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	f.drain(t)

	rec, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
		Scope: "user:1", Query: "allergies",
	})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(rec.Facts) != 1 {
		t.Fatalf("recalled %v after a replayed job, want exactly one fact", statements(rec))
	}
}

// TestFactsRebuildFromL0 is the invariant, exercised on L1: throw the whole
// projection away and the worker reconstructs it from the episodes alone.
func TestFactsRebuildFromL0(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergicTo("hazelnuts"), allergicTo("shellfish"))

	f.append(t, "user:1", "I am allergic to hazelnuts and to shellfish")
	f.drain(t)

	before, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
		Scope: "user:1", Query: "allergies",
	})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(before.Facts) != 2 {
		t.Fatalf("recalled %v, want both allergies", statements(before))
	}

	// Both L1 tables, gone. Nothing else is touched: L0 is the only source of
	// truth, and this is the claim.
	if _, err := f.pool.Exec(t.Context(),
		"TRUNCATE mempher.facts, mempher.fact_extractions"); err != nil {
		t.Fatalf("truncate the projection: %v", err)
	}

	empty, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
		Scope: "user:1", Query: "allergies",
	})
	if err != nil {
		t.Fatalf("Recall after truncation: %v", err)
	}
	if len(empty.Facts) != 0 {
		t.Fatalf("recalled %v from a truncated projection", statements(empty))
	}

	// Rebuild it the way an operator would: ask what is missing, enqueue that.
	pending, err := f.store.PendingExtractions(t.Context(), ops.PendingExtractions{
		Extractor: f.extractor.Model(),
	})
	if err != nil {
		t.Fatalf("PendingExtractions: %v", err)
	}
	if len(pending) != 1 {
		t.Fatalf("%d episodes pending, want the one that was appended", len(pending))
	}
	if _, err := f.store.Enqueue(t.Context(), mempher.NewJob{
		Kind: mempher.JobKindExtract, Scope: "user:1", Episodes: pending,
	}, f.clock.Now()); err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	f.drain(t)

	after, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
		Scope: "user:1", Query: "allergies",
	})
	if err != nil {
		t.Fatalf("Recall after the rebuild: %v", err)
	}
	if len(after.Facts) != len(before.Facts) {
		t.Fatalf("rebuilt %v, want %v", statements(after), statements(before))
	}
	for i := range after.Facts {
		if after.Facts[i].Fact.Statement != before.Facts[i].Fact.Statement {
			t.Errorf("rebuilt %v, want %v", statements(after), statements(before))
			break
		}
	}
}

// TestFactsTakeTheTokenBudgetFirst: the compact, current summary of a scope is
// not what should be dropped to fit another line of transcript.
func TestFactsTakeTheTokenBudgetFirst(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergicTo("hazelnuts"))

	f.append(t, "user:1", "I am allergic to hazelnuts")
	for range 5 {
		f.append(t, "user:1", "some other thing entirely that is quite long indeed")
	}
	f.drain(t)

	rec, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
		Scope: "user:1", Query: "hazelnuts", MaxTokens: 12,
	})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(rec.Facts) != 1 {
		t.Fatalf("recalled %v under a tight budget, want the fact kept", statements(rec))
	}
	if !rec.Truncated {
		t.Error("Truncated = false, want true under a budget that cut results")
	}
	if rec.Tokens > 12 {
		t.Errorf("Tokens = %d, over the budget of 12", rec.Tokens)
	}
}

// TestFactChannelIsOptional covers both ways L1 stays out of the way: a Memory
// with no Extractor, and a request that switches the channel off.
func TestFactChannelIsOptional(t *testing.T) {
	t.Parallel()

	t.Run("no extractor configured", func(t *testing.T) {
		t.Parallel()
		f := newFixture(t)
		f.append(t, "user:1", "I am allergic to hazelnuts")
		f.drain(t)

		rec, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
			Scope: "user:1", Query: "allergies",
		})
		if err != nil {
			t.Fatalf("Recall: %v", err)
		}
		if len(rec.Facts) != 0 {
			t.Errorf("got %d facts with no Extractor configured, want none", len(rec.Facts))
		}
		// Not merely empty: the channel is not run at all, so nothing reports
		// a failure a caller would have to learn to ignore.
		for _, c := range rec.Channels {
			if c.Channel == mempher.ChannelFact {
				t.Errorf("the fact channel ran without an Extractor: %+v", c)
			}
		}
	})

	t.Run("switched off by the request", func(t *testing.T) {
		t.Parallel()
		f := newFactFixture(t, allergicTo("hazelnuts"))
		f.append(t, "user:1", "I am allergic to hazelnuts")
		f.drain(t)

		rec, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
			Scope:    "user:1",
			Query:    "allergies",
			Channels: []mempher.Channel{mempher.ChannelLexical},
		})
		if err != nil {
			t.Fatalf("Recall: %v", err)
		}
		if len(rec.Facts) != 0 {
			t.Errorf("got %v with the fact channel switched off", statements(rec))
		}
	})
}

// TestWorkerWithoutExtractorLeavesExtractJobsAlone: a worker must not claim work
// it cannot do, or it would burn the job's attempts and leave it dead.
func TestWorkerWithoutExtractorLeavesExtractJobsAlone(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergicTo("hazelnuts"))

	plain, err := mempher.NewWorker(mempher.WorkerConfig{
		Store: f.store, Embedder: f.embedder, Clock: f.clock, ID: "encode-only",
	})
	if err != nil {
		t.Fatalf("NewWorker: %v", err)
	}

	f.append(t, "user:1", "I am allergic to hazelnuts")
	for range 5 {
		if _, err := plain.DrainOnce(t.Context()); err != nil {
			t.Fatalf("DrainOnce: %v", err)
		}
	}

	if got := f.extractor.Calls(); got != 0 {
		t.Errorf("an extractor-less worker made %d extraction calls, want 0", got)
	}
	// The extract job is untouched and still claimable, so the worker that can
	// do it still will.
	pending, err := f.store.PendingExtractions(t.Context(), ops.PendingExtractions{
		Extractor: f.extractor.Model(),
	})
	if err != nil {
		t.Fatalf("PendingExtractions: %v", err)
	}
	if len(pending) != 1 {
		t.Fatalf("%d episodes pending extraction, want 1", len(pending))
	}
	f.drain(t)
	if got := f.extractor.Calls(); got == 0 {
		t.Error("the extract job was not left claimable for a worker that can run it")
	}
}

// TestExtractorFailureDegradesRecallRatherThanBreakingIt: extraction is offline
// work, so an extractor that is down costs facts, not appends and not recall.
func TestExtractorFailureDegradesRecallRatherThanBreakingIt(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergicTo("hazelnuts"))
	f.extractor.Fail(errors.New("the model is down"))

	f.append(t, "user:1", "I am allergic to hazelnuts")

	// The encode job succeeds and the extract job fails, so draining reports
	// progress without completing everything.
	for range 5 {
		if _, err := f.worker.DrainOnce(t.Context()); err != nil {
			t.Fatalf("DrainOnce: %v", err)
		}
	}

	rec, err := f.memory.Recall(t.Context(), mempher.RecallRequest{
		Scope: "user:1", Query: "allergies",
	})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(rec.Episodes) == 0 {
		t.Error("recall returned no episodes while only the extractor was down")
	}
	if len(rec.Facts) != 0 {
		t.Errorf("got %v from a failing extractor", statements(rec))
	}
	if got := report(t, rec, mempher.ChannelFact); got.Err != nil {
		t.Errorf("the fact channel itself failed: %v", got.Err)
	}
}

// TestTwoExtractorsDoNotMix: facts are keyed by extractor, so changing one is a
// backfill rather than a silent merge of two opinions.
func TestTwoExtractorsDoNotMix(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergicTo("hazelnuts"))

	f.append(t, "user:1", "I am allergic to hazelnuts")
	f.drain(t)

	second := f.extractor.WithID("memphertest/rules@2")
	facts, err := f.store.Facts(t.Context(), mempher.FactQuery{
		Scope: "user:1", Extractor: second.Model(),
		AsOf: f.clock.Now(), At: f.clock.Now(),
	})
	if err != nil {
		t.Fatalf("Facts: %v", err)
	}
	if len(facts) != 0 {
		t.Errorf("a second extractor already sees %d facts, want none", len(facts))
	}
}

// TestConfigRejectsAFactStoreWithNoExtractor catches the wiring that would leave
// recall silently factless.
func TestConfigRejectsAFactStoreWithNoExtractor(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t)

	_, err := mempher.New(mempher.Config{
		Store: f.store, Embedder: f.embedder, FactStore: f.store,
	})
	if !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Fatalf("err = %v, want ErrInvalidConfig", err)
	}
}
