package eval_test

import (
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/mempher/mempher"
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
	memory, err := mempher.New(mempher.Config{Store: store, Embedder: embedder})
	if err != nil {
		t.Fatalf("mempher.New: %v", err)
	}

	channels := channelsFromEnv()
	cfg := eval.Config{
		Memory:   memory,
		Channels: channels,
		Limit:    intFromEnv("LONGMEMEVAL_K", eval.DefaultLimit),
		Progress: func(done int, last eval.Result) {
			if done%25 == 0 {
				t.Logf("  %d questions, last %s hit=%v", done, last.ID, last.Hit)
			}
		},
	}

	dataset := capped(eval.LongMemEval(path), intFromEnv("LONGMEMEVAL_QUESTIONS", 0))
	started := time.Now()
	summary, err := eval.Run(t.Context(), cfg, dataset)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	t.Logf("channels: %v, wall clock %s\n\n%s",
		channels, time.Since(started).Round(time.Second), summary)

	if summary.Questions == 0 {
		t.Fatal("the benchmark scored no questions at all")
	}
	// Not a quality bar -- the numbers are reported, not asserted. This catches
	// the run that silently retrieved nothing because a channel was misnamed or
	// the haystack never landed.
	if summary.HitRate == 0 {
		t.Error("every question missed, which means the harness is broken, not the ranking")
	}
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

// channelsFromEnv selects the channels to measure. The default is lexical
// alone, because it is the only channel that answers without a model: the
// semantic channel needs a real Embedder and a Worker to have encoded anything,
// and the stand-in embedder here would produce a number that describes a hash
// function rather than a retrieval system.
func channelsFromEnv() []mempher.Channel {
	raw := os.Getenv("LONGMEMEVAL_CHANNELS")
	if raw == "" {
		return []mempher.Channel{mempher.ChannelLexical}
	}
	var channels []mempher.Channel
	for _, name := range strings.Split(raw, ",") {
		channels = append(channels, mempher.Channel(strings.TrimSpace(name)))
	}
	return channels
}

func intFromEnv(key string, fallback int) int {
	if raw := os.Getenv(key); raw != "" {
		if value, err := strconv.Atoi(raw); err == nil {
			return value
		}
	}
	return fallback
}
