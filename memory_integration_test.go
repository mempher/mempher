package mempher_test

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mempher/mempher"
	"github.com/mempher/mempher/internal/pgtest"
	"github.com/mempher/mempher/memphertest"
	"github.com/mempher/mempher/ops"
	"github.com/mempher/mempher/postgres"
)

// This file exercises the whole library through its public API against a real
// PostgreSQL 18. It lives in package mempher_test because postgres imports
// mempher: an in-package test could not import it back.

const dims = 64

var start = time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)

// fixture is everything a test needs, wired the way a caller would wire it.
type fixture struct {
	memory   *mempher.Memory
	worker   *mempher.Worker
	store    *postgres.Store
	pool     *pgxpool.Pool
	clock    *memphertest.Clock
	embedder *memphertest.Embedder
}

func newFixture(t *testing.T) *fixture {
	t.Helper()

	pool := pgtest.Pool(t)
	embedder := memphertest.NewEmbedder(dims)
	if _, err := postgres.Migrate(t.Context(), pool, postgres.MigrateOptions{
		VectorDimensions: embedder.Dimensions(),
	}); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	store, err := postgres.New(t.Context(), pool)
	if err != nil {
		t.Fatalf("postgres.New: %v", err)
	}
	if err := store.VerifyEmbedder(embedder); err != nil {
		t.Fatalf("VerifyEmbedder: %v", err)
	}

	clock := memphertest.NewClock(start)
	memory, err := mempher.New(mempher.Config{
		Store: store, Embedder: embedder, Clock: clock,
	})
	if err != nil {
		t.Fatalf("mempher.New: %v", err)
	}
	worker, err := mempher.NewWorker(mempher.WorkerConfig{
		Store: store, Embedder: embedder, Clock: clock, ID: "test-worker",
	})
	if err != nil {
		t.Fatalf("mempher.NewWorker: %v", err)
	}
	return &fixture{memory, worker, store, pool, clock, embedder}
}

func (f *fixture) append(t *testing.T, scope mempher.ScopeID, content string) mempher.Episode {
	t.Helper()
	ep, err := f.memory.Append(t.Context(), mempher.AppendRequest{
		Scope:   scope,
		Content: content,
		Role:    mempher.RoleUser,
		Source:  "test",
	})
	if err != nil {
		t.Fatalf("Append %q: %v", content, err)
	}
	return ep
}

// drain runs the consolidation loop until the queue is empty.
func (f *fixture) drain(t *testing.T) int {
	t.Helper()
	total := 0
	for range 20 {
		n, err := f.worker.DrainOnce(t.Context())
		if err != nil {
			t.Fatalf("DrainOnce: %v", err)
		}
		if n == 0 {
			return total
		}
		total += n
	}
	t.Fatal("the queue would not drain")
	return total
}

func recalled(r mempher.Recollection) []string {
	out := make([]string, len(r.Episodes))
	for i, e := range r.Episodes {
		out[i] = e.Episode.Content
	}
	return out
}

// TestWriteLoopMakesNoModelCalls is the promise the whole design rests on.
func TestWriteLoopMakesNoModelCalls(t *testing.T) {
	t.Parallel()
	f := newFixture(t)

	for i := range 5 {
		f.append(t, "user:1", fmt.Sprintf("episode %d", i))
	}
	if got := f.embedder.Calls(); got != 0 {
		t.Errorf("Append made %d embedding calls, want 0", got)
	}

	// And the worker is where the spend happens.
	f.drain(t)
	if f.embedder.Calls() == 0 {
		t.Error("the worker made no embedding calls")
	}
}

// TestLexicalIsImmediateSemanticFollows is the visibility contract.
func TestLexicalIsImmediateSemanticFollows(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "I have a hazelnut allergy")

	before, err := f.memory.Recall(ctx, mempher.RecallRequest{
		Scope: "user:1", Query: "hazelnut allergy",
	})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(before.Episodes) != 1 {
		t.Fatalf("got %v, want the episode via the lexical channel alone", recalled(before))
	}
	if hits := before.Episodes[0].Hits; len(hits) != 1 || hits[0].Channel != mempher.ChannelLexical {
		t.Errorf("hits = %v, want only lexical before encoding", hits)
	}

	f.drain(t)

	after, err := f.memory.Recall(ctx, mempher.RecallRequest{
		Scope: "user:1", Query: "hazelnut allergy",
	})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(after.Episodes) != 1 {
		t.Fatalf("got %v, want the episode", recalled(after))
	}
	if hits := after.Episodes[0].Hits; len(hits) != 2 {
		t.Errorf("hits = %v, want both channels once encoded", hits)
	}
}

// TestRecallFusesChannels checks the whole read loop end to end: the episode both
// channels agree on should win.
func TestRecallFusesChannels(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "I have a hazelnut allergy and cannot drink hazelnut syrup")
	f.append(t, "user:1", "my allergy makes travel awkward")
	f.append(t, "user:1", "I drink dark roast coffee with no sugar")
	f.drain(t)

	got, err := f.memory.Recall(ctx, mempher.RecallRequest{
		Scope: "user:1", Query: "hazelnut allergy",
	})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(got.Episodes) == 0 {
		t.Fatal("nothing was recalled")
	}
	if !strings.Contains(got.Episodes[0].Episode.Content, "hazelnut") {
		t.Errorf("best result is %q, want the hazelnut episode", got.Episodes[0].Episode.Content)
	}

	t.Run("scores descend and every result carries its provenance", func(t *testing.T) {
		for i, r := range got.Episodes {
			if len(r.Hits) == 0 {
				t.Errorf("result %d has no hits", i)
			}
			for _, hit := range r.Hits {
				if hit.Rank < 1 {
					t.Errorf("result %d hit %s has rank %d", i, hit.Channel, hit.Rank)
				}
			}
			if i > 0 && r.Score > got.Episodes[i-1].Score {
				t.Errorf("scores are not descending at %d", i)
			}
		}
	})

	t.Run("both channels are reported", func(t *testing.T) {
		if len(got.Channels) != 2 {
			t.Fatalf("got %d channel reports, want 2", len(got.Channels))
		}
		for _, report := range got.Channels {
			if report.Err != nil {
				t.Errorf("channel %s failed: %v", report.Channel, report.Err)
			}
			if report.Candidates == 0 {
				t.Errorf("channel %s contributed nothing", report.Channel)
			}
		}
	})

	t.Run("AsOf is reported so the request can be replayed", func(t *testing.T) {
		if !got.AsOf.Equal(f.clock.Now()) {
			t.Errorf("AsOf = %s, want the clock's now %s", got.AsOf, f.clock.Now())
		}
	})
}

// TestRecallDegradesWhenTheEmbedderFails is the reason a channel failure is
// reported rather than fatal.
func TestRecallDegradesWhenTheEmbedderFails(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "I have a hazelnut allergy")
	f.drain(t)

	outage := errors.New("the embedding provider is down")
	f.embedder.Fail(outage)

	got, err := f.memory.Recall(ctx, mempher.RecallRequest{
		Scope: "user:1", Query: "hazelnut allergy",
	})
	if err != nil {
		t.Fatalf("Recall should degrade, not fail: %v", err)
	}
	if len(got.Episodes) != 1 {
		t.Errorf("got %v, want the lexical channel to still answer", recalled(got))
	}

	var semantic mempher.ChannelReport
	for _, report := range got.Channels {
		if report.Channel == mempher.ChannelSemantic {
			semantic = report
		}
	}
	if !errors.Is(semantic.Err, outage) {
		t.Errorf("the semantic channel reports %v, want the embedder outage", semantic.Err)
	}

	t.Run("but failing every channel is an error", func(t *testing.T) {
		_, err := f.memory.Recall(ctx, mempher.RecallRequest{
			Scope:    "user:1",
			Query:    "hazelnut allergy",
			Channels: []mempher.Channel{mempher.ChannelSemantic},
		})
		if !errors.Is(err, mempher.ErrAllChannelsFailed) {
			t.Errorf("err = %v, want mempher.ErrAllChannelsFailed", err)
		}
		if !errors.Is(err, outage) {
			t.Errorf("the underlying cause was lost: %v", err)
		}
	})
}

func TestRecallAsOfAndBudget(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "hazelnut one")
	cut := f.clock.Now()
	f.clock.Advance(time.Hour)
	f.append(t, "user:1", "hazelnut two")
	f.drain(t)

	t.Run("as-of hides what was not yet known", func(t *testing.T) {
		got, err := f.memory.Recall(ctx, mempher.RecallRequest{
			Scope: "user:1", Query: "hazelnut", AsOf: cut,
		})
		if err != nil {
			t.Fatalf("Recall: %v", err)
		}
		if len(got.Episodes) != 1 || got.Episodes[0].Episode.Content != "hazelnut one" {
			t.Errorf("got %v, want only the episode known at the cut", recalled(got))
		}
	})

	t.Run("limit truncates and says so", func(t *testing.T) {
		got, err := f.memory.Recall(ctx, mempher.RecallRequest{
			Scope: "user:1", Query: "hazelnut", Limit: 1,
		})
		if err != nil {
			t.Fatalf("Recall: %v", err)
		}
		if len(got.Episodes) != 1 || !got.Truncated {
			t.Errorf("got %d episodes, truncated %v; want 1 and true",
				len(got.Episodes), got.Truncated)
		}
	})

	t.Run("the token budget is never exceeded", func(t *testing.T) {
		full, err := f.memory.Recall(ctx, mempher.RecallRequest{Scope: "user:1", Query: "hazelnut"})
		if err != nil {
			t.Fatalf("Recall: %v", err)
		}
		if len(full.Episodes) != 2 || full.Tokens == 0 {
			t.Fatalf("expected two episodes with a token cost, got %d (%d tokens)",
				len(full.Episodes), full.Tokens)
		}

		budget := full.Episodes[0].Tokens
		got, err := f.memory.Recall(ctx, mempher.RecallRequest{
			Scope: "user:1", Query: "hazelnut", MaxTokens: budget,
		})
		if err != nil {
			t.Fatalf("Recall: %v", err)
		}
		if got.Tokens > budget {
			t.Errorf("returned %d tokens against a budget of %d", got.Tokens, budget)
		}
		if len(got.Episodes) != 1 || !got.Truncated {
			t.Errorf("got %d episodes, truncated %v; want 1 and true",
				len(got.Episodes), got.Truncated)
		}
	})

	t.Run("a budget smaller than the best result returns nothing rather than overflowing",
		func(t *testing.T) {
			got, err := f.memory.Recall(ctx, mempher.RecallRequest{
				Scope: "user:1", Query: "hazelnut", MaxTokens: 1,
			})
			if err != nil {
				t.Fatalf("Recall: %v", err)
			}
			if len(got.Episodes) != 0 || !got.Truncated {
				t.Errorf("got %v (truncated %v), want nothing and a truncation flag",
					recalled(got), got.Truncated)
			}
		})
}

func TestRecallChannelSelection(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "I have a hazelnut allergy")
	f.drain(t)

	for _, channel := range []mempher.Channel{mempher.ChannelSemantic, mempher.ChannelLexical} {
		t.Run(string(channel), func(t *testing.T) {
			got, err := f.memory.Recall(ctx, mempher.RecallRequest{
				Scope: "user:1", Query: "hazelnut allergy",
				Channels: []mempher.Channel{channel, channel}, // repeats collapse
			})
			if err != nil {
				t.Fatalf("Recall: %v", err)
			}
			if len(got.Channels) != 1 || got.Channels[0].Channel != channel {
				t.Fatalf("channel reports = %+v, want just %s", got.Channels, channel)
			}
			if len(got.Episodes) != 1 {
				t.Fatalf("got %v, want the episode", recalled(got))
			}
			if hits := got.Episodes[0].Hits; len(hits) != 1 || hits[0].Channel != channel {
				t.Errorf("hits = %v, want only %s", hits, channel)
			}
		})
	}
}

func TestScopeIsolationEndToEnd(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "my hazelnut allergy")
	f.append(t, "user:2", "my hazelnut allergy")
	f.drain(t)

	for _, scope := range []mempher.ScopeID{"user:1", "user:2"} {
		got, err := f.memory.Recall(ctx, mempher.RecallRequest{Scope: scope, Query: "hazelnut"})
		if err != nil {
			t.Fatalf("Recall %s: %v", scope, err)
		}
		if len(got.Episodes) != 1 {
			t.Fatalf("scope %s recalled %d episodes, want 1", scope, len(got.Episodes))
		}
		if got.Episodes[0].Episode.Scope != scope {
			t.Errorf("scope %s got an episode from %s", scope, got.Episodes[0].Episode.Scope)
		}
	}
}

// TestWorkerRetriesThenGivesUp walks the consolidation loop's failure path.
func TestWorkerRetriesThenGivesUp(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	f.append(t, "user:1", "an episode that will not encode")
	f.embedder.Fail(errors.New("the embedding provider is down"))

	done, err := f.worker.DrainOnce(ctx)
	if err != nil {
		t.Fatalf("DrainOnce: %v", err)
	}
	if done != 0 {
		t.Errorf("%d jobs reported done despite the embedder failing", done)
	}

	// The job is pending again, with its retry pushed into the future.
	pending, err := f.store.PendingEncodings(ctx, ops.PendingEncodings{
		Model: f.embedder.Model(),
	})
	if err != nil {
		t.Fatalf("PendingEncodings: %v", err)
	}
	if len(pending) != 1 {
		t.Errorf("got %d episodes still needing an encoding, want 1", len(pending))
	}

	// Nothing is claimable until the backoff elapses.
	if done, err := f.worker.DrainOnce(ctx); err != nil || done != 0 {
		t.Errorf("DrainOnce during backoff = %d, %v", done, err)
	}

	// Once it does, and the provider is back, the work completes.
	f.embedder.Fail(nil)
	f.clock.Advance(time.Hour)
	if done := f.drain(t); done != 1 {
		t.Errorf("drained %d jobs after recovery, want 1", done)
	}
	pending, err = f.store.PendingEncodings(ctx, ops.PendingEncodings{
		Model: f.embedder.Model(),
	})
	if err != nil {
		t.Fatalf("PendingEncodings: %v", err)
	}
	if len(pending) != 0 {
		t.Errorf("%d episodes still need encoding after recovery", len(pending))
	}
}

// TestWorkerRunStopsOnCancellation checks the loop honours its context.
func TestWorkerRunStopsOnCancellation(t *testing.T) {
	t.Parallel()
	f := newFixture(t)

	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan error, 1)
	go func() { done <- f.worker.Run(ctx) }()

	f.append(t, "user:1", "some work to do")

	// Wait for the loop to have done the work, rather than for a fixed
	// interval and a hope. A lease, an embedding and a write take as long as
	// they take, and on a loaded machine that is longer than any number
	// written here would be; the deadline exists only so a loop that never
	// makes progress fails instead of hanging.
	deadline := time.Now().Add(30 * time.Second)
	for {
		// Run has no reason to return before it is cancelled, so catching it
		// here is what proves the loop stayed up rather than dying quietly and
		// leaving the work to look done by someone else.
		select {
		case err := <-done:
			t.Fatalf("Run returned before it was cancelled: %v", err)
		default:
		}

		pending, err := f.store.PendingEncodings(t.Context(), ops.PendingEncodings{
			Model: f.embedder.Model(),
		})
		if err != nil {
			t.Fatalf("PendingEncodings: %v", err)
		}
		if len(pending) == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d episodes were left unencoded", len(pending))
		}
		time.Sleep(20 * time.Millisecond)
	}

	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("Run returned %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not stop when its context was cancelled")
	}
}

// TestProjectionsRebuildFromL0 is the invariant, exercised: throw the vector
// projection away and the worker reconstructs it from the episodes alone.
func TestProjectionsRebuildFromL0(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	for i := range 4 {
		f.append(t, "user:1", fmt.Sprintf("hazelnut episode %d", i))
	}
	f.drain(t)

	before, err := f.memory.Recall(ctx, mempher.RecallRequest{Scope: "user:1", Query: "hazelnut"})
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}

	// Drop every encoding, as a bad extractor or a model change would. The
	// episodes themselves cannot be dropped: the schema forbids it.
	if _, err := f.pool.Exec(ctx, "TRUNCATE mempher.episode_encodings"); err != nil {
		t.Fatalf("truncate encodings: %v", err)
	}

	// The queue is a projection of L0 too: ask which episodes lack an encoding
	// and queue them.
	pending, err := f.store.PendingEncodings(ctx, ops.PendingEncodings{Model: f.embedder.Model()})
	if err != nil {
		t.Fatalf("PendingEncodings: %v", err)
	}
	if len(pending) != 4 {
		t.Fatalf("got %d episodes needing encoding, want 4", len(pending))
	}
	for _, id := range pending {
		if _, err := f.store.Enqueue(ctx, mempher.NewJob{
			Kind: mempher.JobKindEncode, Scope: "user:1", Episodes: []mempher.EpisodeID{id},
		}, f.clock.Now()); err != nil {
			t.Fatalf("Enqueue: %v", err)
		}
	}
	f.drain(t)

	after, err := f.memory.Recall(ctx, mempher.RecallRequest{Scope: "user:1", Query: "hazelnut"})
	if err != nil {
		t.Fatalf("Recall after rebuild: %v", err)
	}
	if fmt.Sprint(recalled(after)) != fmt.Sprint(recalled(before)) {
		t.Errorf("rebuilt recall = %v, want the original %v", recalled(after), recalled(before))
	}
}
