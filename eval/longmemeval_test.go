package eval_test

import (
	"os"
	"slices"
	"strconv"
	"strings"
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
// Scopes are deliberately not erased between questions. By the end the episode
// table holds every haystack at once, so the recall latency reported is latency
// against a quarter of a million episodes rather than against one conversation.
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
	sets := channelSetsFromEnv(embedder)
	var worker *mempher.Worker
	if usesSemantic(sets) {
		worker, err = mempher.NewWorker(mempher.WorkerConfig{
			Store: store, Embedder: embedder, ID: "eval",
		})
		if err != nil {
			t.Fatalf("mempher.NewWorker: %v", err)
		}
	}

	cfg := eval.Config{
		Memory:   memory,
		Worker:   worker,
		Channels: sets,
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
			t.Errorf("every question missed on %v, so the harness is broken, not the ranking",
				summary.Channels)
		}
	}
}

// usesSemantic reports whether any set needs episodes to have been encoded,
// which is what decides if the run pays for a worker.
func usesSemantic(sets [][]mempher.Channel) bool {
	for _, set := range sets {
		if slices.Contains(set, mempher.ChannelSemantic) {
			return true
		}
	}
	return false
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

// channelSetsFromEnv selects what to measure.
//
// With a real embedder the default is all three sets -- each channel alone and
// then fused -- because that is the comparison worth having and the haystack is
// ingested once for all of them. Without one it is lexical alone, since nothing
// else can answer.
//
// LONGMEMEVAL_CHANNELS overrides it: sets separated by spaces, channels within a
// set by commas, as in "lexical semantic semantic,lexical".
func channelSetsFromEnv(embedder mempher.Embedder) [][]mempher.Channel {
	raw := os.Getenv("LONGMEMEVAL_CHANNELS")
	if raw == "" {
		if _, stand := embedder.(*memphertest.Embedder); stand {
			return [][]mempher.Channel{{mempher.ChannelLexical}}
		}
		return [][]mempher.Channel{
			{mempher.ChannelLexical},
			{mempher.ChannelSemantic},
			{mempher.ChannelSemantic, mempher.ChannelLexical},
		}
	}
	var sets [][]mempher.Channel
	for _, group := range strings.Fields(raw) {
		var set []mempher.Channel
		for _, name := range strings.Split(group, ",") {
			set = append(set, mempher.Channel(strings.TrimSpace(name)))
		}
		sets = append(sets, set)
	}
	return sets
}

func intFromEnv(key string, fallback int) int {
	if raw := os.Getenv(key); raw != "" {
		if value, err := strconv.Atoi(raw); err == nil {
			return value
		}
	}
	return fallback
}
