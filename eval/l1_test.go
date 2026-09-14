package eval_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/eval"
	"github.com/mempher/mempher/extract"
	extractopenai "github.com/mempher/mempher/extract/openai"
	"github.com/mempher/mempher/internal/pgtest"
	"github.com/mempher/mempher/memphertest"
	"github.com/mempher/mempher/postgres"
)

// TestLongMemEvalL1 measures fact extraction: whether the claims derived from a
// conversation are the right ones, and whether a later episode closes an earlier
// claim's window instead of leaving both standing.
//
//	LONGMEMEVAL=/path/to/longmemeval_oracle \
//	LONGMEMEVAL_EXTRACT_URL=http://127.0.0.1:8479/v1 \
//	LONGMEMEVAL_EXTRACT_MODEL=phi-3-mini-4k-instruct \
//	go test ./eval/ -run LongMemEvalL1 -v -timeout 240m
//
// The oracle file rather than the haystack one, deliberately: this measures
// extraction, and the evidence sessions are what there is to extract from.
// Running it over fifty sessions of distractors would cost fifty times as much
// to answer a question that belongs to L0 and is already answered there.
func TestLongMemEvalL1(t *testing.T) {
	path := os.Getenv("LONGMEMEVAL")
	if path == "" {
		t.Skip("set LONGMEMEVAL to a longmemeval_oracle file to run the L1 benchmark")
	}
	url := os.Getenv("LONGMEMEVAL_EXTRACT_URL")
	if url == "" {
		t.Skip("set LONGMEMEVAL_EXTRACT_URL to an OpenAI-compatible chat endpoint")
	}

	pool := pgtest.Pool(t)
	// The stand-in embedder: L1 reads no vector, and a real one would spend the
	// run's budget encoding episodes nothing here searches.
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

	completer, err := extractopenai.New(extractopenai.Config{
		BaseURL: url,
		Model:   os.Getenv("LONGMEMEVAL_EXTRACT_MODEL"),
		// A local stand-in does not implement strict json_schema, so the schema
		// is a hint and the extract package's validation is what holds the line.
		// That is the configuration worth measuring anyway: it is what a caller
		// pointing at a gateway without strict mode gets.
		Lenient: true,
	})
	if err != nil {
		t.Fatalf("openai.New: %v", err)
	}
	extractor, err := extract.New(completer, extract.Options{
		// No predicate vocabulary. LongMemEval is open-domain, so a closed one
		// would be invented for the benchmark rather than derived from a
		// deployment -- and how much an open vocabulary costs is one of the
		// things worth finding out.
		MaxAssertions: intFromEnv("LONGMEMEVAL_MAX_ASSERTIONS", 6),
	})
	if err != nil {
		t.Fatalf("extract.New: %v", err)
	}

	memory, err := mempher.New(mempher.Config{
		Store: store, Embedder: embedder, Extractor: extractor,
	})
	if err != nil {
		t.Fatalf("mempher.New: %v", err)
	}
	worker, err := mempher.NewWorker(mempher.WorkerConfig{
		Store: store, Embedder: embedder, Extractor: extractor, ID: "l1-eval",
	})
	if err != nil {
		t.Fatalf("mempher.NewWorker: %v", err)
	}

	dataset := capped(
		onlyType(eval.LongMemEval(path), os.Getenv("LONGMEMEVAL_TYPE")),
		intFromEnv("LONGMEMEVAL_QUESTIONS", 0))

	started := time.Now()
	summary, err := eval.RunL1(t.Context(), eval.L1Config{
		Memory:    memory,
		Worker:    worker,
		Facts:     store,
		Extractor: extractor,
		Store:     store,
		Replay:    os.Getenv("LONGMEMEVAL_REPLAY") != "",
		Progress: func(done int, last eval.L1Result) {
			t.Logf("  %d: %s  %d facts (%d closed, %d contradicting) from %d episodes in %s",
				done, last.ID, last.Facts, last.Closed, last.Contradictions,
				last.Episodes, last.Duration.Round(time.Second))
		},
	}, dataset)
	if err != nil {
		t.Fatalf("RunL1: %v", err)
	}

	t.Logf("extractor %s, wall clock %s\n\n%s",
		extractor.Model(), time.Since(started).Round(time.Second), summary)

	if summary.Questions == 0 {
		t.Fatal("the benchmark scored no questions at all")
	}
	// Not a quality bar. This catches the run where every extraction failed and
	// the zeros mean "nothing happened" rather than "nothing was found".
	if summary.FactsPerEpisode == 0 {
		t.Error("no question produced a single fact, so the extractor is broken, not strict")
	}
}

// onlyType narrows a dataset to one question category, which is how the
// supersession numbers get measured on the questions that are about the world
// having changed.
func onlyType(ds eval.Dataset, want string) eval.Dataset {
	if want == "" {
		return ds
	}
	inner := ds.Questions
	ds.Name += " [" + want + "]"
	ds.Questions = func(yield func(eval.Question, error) bool) {
		for question, err := range inner {
			if err == nil && !strings.EqualFold(question.Type, want) {
				continue
			}
			if !yield(question, err) {
				return
			}
		}
	}
	return ds
}
