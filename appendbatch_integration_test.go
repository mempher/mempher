package mempher_test

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/ops"
)

// AppendBatch through the public API. What it has to preserve is everything
// Append already guaranteed -- a dense total order, jobs that commit with their
// episodes, one scope -- while paying for them once instead of n times.

// turn builds a batch of requests for one scope.
func turn(scope mempher.ScopeID, contents ...string) []mempher.AppendRequest {
	reqs := make([]mempher.AppendRequest, len(contents))
	for i, content := range contents {
		reqs[i] = mempher.AppendRequest{
			Scope:   scope,
			Content: content,
			Role:    mempher.RoleUser,
			Source:  "test",
		}
	}
	return reqs
}

func TestAppendBatchSequencesInOrder(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	// A single append first, so the batch has to continue an existing log
	// rather than start at 1.
	first := f.append(t, "user:1", "zero")

	got, err := f.memory.AppendBatch(ctx, turn("user:1", "one", "two", "three"))
	if err != nil {
		t.Fatalf("AppendBatch: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("returned %d episodes, want 3", len(got))
	}

	for i, ep := range got {
		if want := first.Seq + int64(i) + 1; ep.Seq != want {
			t.Errorf("episode %d has Seq %d, want %d: the block must be contiguous",
				i, ep.Seq, want)
		}
		if ep.ID.IsZero() {
			t.Errorf("episode %d has no id", i)
		}
	}
	// The order returned is the order given, which is what lets a caller match
	// an id back to what it sent.
	for i, want := range []string{"one", "two", "three"} {
		if got[i].Content != want {
			t.Errorf("episode %d is %q, want %q", i, got[i].Content, want)
		}
	}
	// Recorded together means recorded at one instant.
	for i, ep := range got {
		if !ep.IngestedAt.Equal(got[0].IngestedAt) {
			t.Errorf("episode %d was ingested at %s, want the batch's %s",
				i, ep.IngestedAt, got[0].IngestedAt)
		}
	}

	// And the log reads back in that order.
	replayed, err := f.store.Replay(ctx, "user:1", 0, 10)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	var contents []string
	for _, ep := range replayed {
		contents = append(contents, ep.Content)
	}
	if strings.Join(contents, ",") != "zero,one,two,three" {
		t.Errorf("replay = %v, want the order they were appended in", contents)
	}
}

// TestAppendBatchEnqueuesOneJobPerKind is the reason a batch is worth having:
// the consolidation it implies is batched too, so a turn costs one model call
// rather than one per episode.
func TestAppendBatchEnqueuesOneJobPerKind(t *testing.T) {
	t.Parallel()
	f := newFactFixture(t, allergy)
	ctx := t.Context()

	got, err := f.memory.AppendBatch(ctx, turn("user:1", "one", "two", "three", "four"))
	if err != nil {
		t.Fatalf("AppendBatch: %v", err)
	}

	o, err := ops.New(ops.Config{Store: f.store, Embedder: f.embedder, Clock: f.clock})
	if err != nil {
		t.Fatalf("ops.New: %v", err)
	}
	jobs, err := o.Jobs(ctx, ops.JobQuery{Scope: "user:1"})
	if err != nil {
		t.Fatalf("Jobs: %v", err)
	}
	if len(jobs) != 2 {
		t.Fatalf("%d jobs for a batch of 4, want one encode and one extract", len(jobs))
	}
	for _, job := range jobs {
		if len(job.Episodes) != 4 {
			t.Errorf("%s job names %d episodes, want all 4", job.Kind, len(job.Episodes))
		}
		// In Seq order, because an extractor reading a turn out of order
		// resolves "that" against the wrong episode.
		for i, id := range job.Episodes {
			if id != got[i].ID {
				t.Errorf("%s job episode %d is %s, want %s", job.Kind, i, id, got[i].ID)
			}
		}
	}

	// One encode job means one call to EmbedDocuments, which is where the
	// money goes.
	before := f.embedder.Calls()
	f.drain(t)
	if calls := f.embedder.Calls() - before; calls != 1 {
		t.Errorf("draining a batch of 4 made %d embedder calls, want 1", calls)
	}
}

// TestAppendBatchIsAtomic: one bad episode appends none of them.
func TestAppendBatchIsAtomic(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	reqs := turn("user:1", "fine", "also fine", "")
	if _, err := f.memory.AppendBatch(ctx, reqs); !errors.Is(err, mempher.ErrInvalidContent) {
		t.Fatalf("err = %v, want mempher.ErrInvalidContent", err)
	}

	replayed, err := f.store.Replay(ctx, "user:1", 0, 10)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if len(replayed) != 0 {
		t.Errorf("%d episodes were appended by a rejected batch, want none", len(replayed))
	}
}

func TestAppendBatchRejectsBadBatches(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	if _, err := f.memory.AppendBatch(ctx, nil); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("empty batch err = %v, want mempher.ErrInvalidConfig", err)
	}

	// One scope per batch, because sequencing locks that scope's row.
	mixed := turn("user:1", "mine")
	mixed = append(mixed, turn("user:2", "theirs")...)
	if _, err := f.memory.AppendBatch(ctx, mixed); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("mixed-scope batch err = %v, want mempher.ErrInvalidConfig", err)
	}

	contents := make([]string, mempher.MaxAppendBatch+1)
	for i := range contents {
		contents[i] = fmt.Sprintf("episode %d", i)
	}
	if _, err := f.memory.AppendBatch(ctx, turn("user:1", contents...)); !errors.Is(
		err, mempher.ErrInvalidConfig) {
		t.Errorf("over-cap batch err = %v, want mempher.ErrInvalidConfig", err)
	}
}

// TestAppendBatchKeepsTheLogTotallyOrdered runs batches and singles at one scope
// at once. Sequence allocation is a lock on the scope row, so the result has to
// be dense and gap-free however the calls interleave.
func TestAppendBatchKeepsTheLogTotallyOrdered(t *testing.T) {
	t.Parallel()
	f := newFixture(t)
	ctx := t.Context()

	const batches, perBatch, singles = 4, 5, 6
	var wg sync.WaitGroup
	for b := range batches {
		wg.Add(1)
		go func() {
			defer wg.Done()
			contents := make([]string, perBatch)
			for i := range contents {
				contents[i] = fmt.Sprintf("batch %d episode %d", b, i)
			}
			if _, err := f.memory.AppendBatch(ctx, turn("user:1", contents...)); err != nil {
				t.Errorf("AppendBatch: %v", err)
			}
		}()
	}
	for i := range singles {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := f.memory.Append(ctx, mempher.AppendRequest{
				Scope: "user:1", Content: fmt.Sprintf("single %d", i),
				Role: mempher.RoleUser, Source: "test",
			}); err != nil {
				t.Errorf("Append: %v", err)
			}
		}()
	}
	wg.Wait()

	replayed, err := f.store.Replay(ctx, "user:1", 0, 100)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	want := batches*perBatch + singles
	if len(replayed) != want {
		t.Fatalf("%d episodes, want %d", len(replayed), want)
	}
	for i, ep := range replayed {
		if ep.Seq != int64(i+1) {
			t.Fatalf("episode %d has Seq %d: the sequence is not dense", i, ep.Seq)
		}
	}

	// A batch's episodes stay contiguous: another caller cannot land inside
	// one, because the whole block is allocated under one lock.
	seqOf := map[string]int64{}
	for _, ep := range replayed {
		seqOf[ep.Content] = ep.Seq
	}
	for b := range batches {
		base := seqOf[fmt.Sprintf("batch %d episode 0", b)]
		for i := range perBatch {
			got := seqOf[fmt.Sprintf("batch %d episode %d", b, i)]
			if got != base+int64(i) {
				t.Errorf("batch %d episode %d has Seq %d, want %d: the block was split",
					b, i, got, base+int64(i))
			}
		}
	}
}
