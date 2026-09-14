package eval_test

import (
	"fmt"
	"os"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/embed/openai"
	"github.com/mempher/mempher/eval"
	"github.com/mempher/mempher/internal/pgtest"
	"github.com/mempher/mempher/memphertest"
	"github.com/mempher/mempher/postgres"
)

// dims is the width of the stand-in embedder. Nothing measured here reads a
// vector, but mempher.New requires an Embedder and the schema is migrated for
// whatever it declares.
const dims = 128

// TestLongMemEval runs the retrieval benchmark and prints what it measured.
//
// It is a test rather than a benchmark because what it produces is a score, not
// a duration, and because Go's benchmark harness would run it repeatedly for a
// number that does not vary. Run it with:
//
//	LONGMEMEVAL=/path/to/longmemeval_s go test ./eval/ -run LongMemEval -v -timeout 30m
//
// Each scope is erased once it has been scored, which keeps the run tractable;
// LONGMEMEVAL_KEEP=1 accumulates instead, so that latency is measured against
// the whole corpus rather than one conversation.
func TestLongMemEval(t *testing.T) {
	path := os.Getenv("LONGMEMEVAL")
	if path == "" {
		t.Skip("set LONGMEMEVAL to a longmemeval_s, _m or _oracle file to run the benchmark")
	}

	pool := pgtest.Pool(t)
	embedder := embedderFromEnv(t)
	if _, err := postgres.Migrate(t.Context(), pool, postgres.MigrateOptions{
		VectorDimensions: embedder.Dimensions(),
	}); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	store, err := postgres.New(t.Context(), pool)
	if err != nil {
		t.Fatalf("postgres.New: %v", err)
	}
	memory, err := mempher.New(mempher.Config{Store: store, Embedder: embedder})
	if err != nil {
		t.Fatalf("mempher.New: %v", err)
	}

	// A worker only when the semantic channel is being measured: encoding a
	// quarter of a million episodes is the expensive half of the run, and a
	// lexical measurement does not read a single vector.
	variants := variantsFor(t, store, embedder, memory)
	var worker *mempher.Worker
	if usesSemantic(variants) {
		worker, err = mempher.NewWorker(mempher.WorkerConfig{
			Store: store, Embedder: embedder, ID: "eval",
		})
		if err != nil {
			t.Fatalf("mempher.NewWorker: %v", err)
		}
	}

	// Erasing each scope once it is scored keeps the run tractable, and the
	// cost is only to the latency figures.
	//
	// HNSW insert cost grows with index size, so a run that accumulates every
	// haystack is paying more to ingest question 400 than question 4: measured,
	// it went from eight seconds a question to nearly fifty. Erasing holds the
	// index at one scope. It cannot affect a quality metric -- a recall never
	// crosses a scope, so a question can only ever have searched its own
	// haystack either way -- but it does mean latency is measured against one
	// conversation rather than against the whole corpus. LONGMEMEVAL_KEEP=1
	// accumulates instead, which is how to measure latency at full size.
	var forgetful mempher.Store = store
	if os.Getenv("LONGMEMEVAL_KEEP") != "" {
		forgetful = nil
	}

	cfg := eval.Config{
		Memory:   memory,
		Store:    forgetful,
		Worker:   worker,
		Variants: variants,
		Limit:    intFromEnv("LONGMEMEVAL_K", eval.DefaultLimit),
		Progress: func(done int, last eval.Result) {
			if done%25 == 0 {
				t.Logf("  %d questions, last %s hit=%v", done, last.ID, last.Hit)
			}
		},
	}

	dataset := capped(eval.LongMemEval(path), intFromEnv("LONGMEMEVAL_QUESTIONS", 0))
	started := time.Now()
	summaries, err := eval.Run(t.Context(), cfg, dataset)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	t.Logf("embedder %s, wall clock %s", embedder.Model(), time.Since(started).Round(time.Second))
	for _, summary := range summaries {
		t.Logf("\n%s", summary)
	}

	if summaries[0].Questions == 0 {
		t.Fatal("the benchmark scored no questions at all")
	}
	// Not a quality bar -- the numbers are reported, not asserted. This catches
	// the run that retrieved nothing because a channel was misnamed or the
	// haystack never landed, which otherwise looks like a bad score.
	for _, summary := range summaries {
		if summary.HitRate == 0 {
			t.Errorf("every question missed on %s, so the harness is broken, not the ranking",
				summary.Variant)
		}
	}
}

// usesSemantic reports whether any variant needs episodes to have been encoded,
// which is what decides if the run pays for a worker.
func usesSemantic(variants []eval.Variant) bool {
	for _, variant := range variants {
		if slices.Contains(variant.Channels, mempher.ChannelSemantic) {
			return true
		}
	}
	return false
}

// weighted returns a Memory identical to the one under test except for how
// fusion scores the lexical channel, so that a run can compare fusion settings
// without ingesting the corpus once per setting.
func weighted(t *testing.T, store mempher.Store, embedder mempher.Embedder, lexical float64) *mempher.Memory {
	t.Helper()
	memory, err := mempher.New(mempher.Config{
		Store:    store,
		Embedder: embedder,
		Fusion: mempher.FusionOptions{
			Weights: map[mempher.Channel]float64{mempher.ChannelLexical: lexical},
		},
	})
	if err != nil {
		t.Fatalf("mempher.New weighted %v: %v", lexical, err)
	}
	return memory
}

// capped shortens a dataset, for a smoke run over a handful of questions.
func capped(ds eval.Dataset, limit int) eval.Dataset {
	if limit <= 0 {
		return ds
	}
	inner := ds.Questions
	ds.Questions = func(yield func(eval.Question, error) bool) {
		seen := 0
		for question, err := range inner {
			if !yield(question, err) {
				return
			}
			if seen++; seen >= limit {
				return
			}
		}
	}
	return ds
}

// embedderFromEnv builds the embedder under test.
//
// With LONGMEMEVAL_EMBED_URL set it is a real model behind an OpenAI-compatible
// endpoint -- a provider, or anything local that speaks the same shape. Without
// it the stand-in hashed bag of words stands in, which is fine for a lexical run
// because nothing reads a vector, and useless for a semantic one: a number from
// it would describe a hash function rather than a retrieval system.
func embedderFromEnv(t *testing.T) mempher.Embedder {
	t.Helper()
	url := os.Getenv("LONGMEMEVAL_EMBED_URL")
	if url == "" {
		return memphertest.NewEmbedder(dims)
	}
	embedder, err := openai.New(openai.Config{
		BaseURL:     url,
		Model:       os.Getenv("LONGMEMEVAL_EMBED_MODEL"),
		Dimensions:  intFromEnv("LONGMEMEVAL_EMBED_DIMS", 0),
		QueryPrefix: os.Getenv("LONGMEMEVAL_EMBED_PREFIX"),
		MaxBatch:    intFromEnv("LONGMEMEVAL_EMBED_BATCH", 0),
	})
	if err != nil {
		t.Fatalf("openai.New: %v", err)
	}
	return embedder
}

// variantsFor is what the run compares.
//
// With a real embedder that is each channel alone and then fusion at several
// lexical weights, because the equal-weight default turned out to score below
// the semantic channel on its own and the question is whether weighting
// recovers it. Without an embedder it is lexical alone, since nothing else can
// answer. All of them share one ingestion.
func variantsFor(
	t *testing.T,
	store mempher.Store,
	embedder mempher.Embedder,
	memory *mempher.Memory,
) []eval.Variant {
	t.Helper()
	lexicalOnly := eval.Variant{
		Name:     "lexical",
		Channels: []mempher.Channel{mempher.ChannelLexical},
		Memory:   memory,
	}
	if _, stand := embedder.(*memphertest.Embedder); stand {
		return []eval.Variant{lexicalOnly}
	}

	both := []mempher.Channel{mempher.ChannelSemantic, mempher.ChannelLexical}
	variants := []eval.Variant{
		lexicalOnly,
		{
			Name:     "semantic",
			Channels: []mempher.Channel{mempher.ChannelSemantic},
			Memory:   memory,
		},
	}
	for _, weight := range []float64{1, 0.5, 0.25, 0.1} {
		variants = append(variants, eval.Variant{
			Name:     fmt.Sprintf("fused, lexical weight %.2g", weight),
			Channels: both,
			Memory:   weighted(t, store, embedder, weight),
		})
	}
	return variants
}

func intFromEnv(key string, fallback int) int {
	if raw := os.Getenv(key); raw != "" {
		if value, err := strconv.Atoi(raw); err == nil {
			return value
		}
	}
	return fallback
}
