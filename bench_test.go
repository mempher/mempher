package mempher_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/internal/pgtest"
	"github.com/mempher/mempher/memphertest"
	"github.com/mempher/mempher/postgres"
)

// Benchmarks for the two paths a request waits on. Everything runs against a
// real PostgreSQL 18 in a container, because the numbers worth having are the
// database's: a fake store would benchmark this package's own bookkeeping.
//
// The Embedder is memphertest's hashed bag of words, which costs microseconds.
// That is deliberate -- it measures what mempher spends rather than what a
// model provider spends, and a real embedder adds its own network round trip to
// Recall and nothing at all to Append.

// benchSeed is how many episodes a recall benchmark searches over.
const benchSeed = 1000

// benchFixture is the fixture without the *testing.T.
type benchFixture struct {
	memory   *mempher.Memory
	worker   *mempher.Worker
	store    *postgres.Store
	embedder *memphertest.Embedder
}

func newBenchFixture(b *testing.B, extractor mempher.Extractor) *benchFixture {
	b.Helper()

	pool := pgtest.Pool(b)
	embedder := memphertest.NewEmbedder(dims)
	if _, err := postgres.Migrate(b.Context(), pool, postgres.MigrateOptions{
		VectorDimensions: embedder.Dimensions(),
	}); err != nil {
		b.Fatalf("Migrate: %v", err)
	}
	store, err := postgres.New(b.Context(), pool)
	if err != nil {
		b.Fatalf("postgres.New: %v", err)
	}
	memory, err := mempher.New(mempher.Config{
		Store: store, Embedder: embedder, Extractor: extractor,
	})
	if err != nil {
		b.Fatalf("mempher.New: %v", err)
	}
	worker, err := mempher.NewWorker(mempher.WorkerConfig{
		Store: store, Embedder: embedder, Extractor: extractor, Batch: 64,
	})
	if err != nil {
		b.Fatalf("mempher.NewWorker: %v", err)
	}
	return &benchFixture{memory, worker, store, embedder}
}

// seed appends n episodes and consolidates every one of them, so a recall
// benchmark searches a scope that is fully encoded rather than half of one.
func (f *benchFixture) seed(ctx context.Context, b *testing.B, n int) {
	b.Helper()
	for i := range n {
		if _, err := f.memory.Append(ctx, mempher.AppendRequest{
			Scope:   "bench",
			Content: benchContent(i),
			Role:    mempher.RoleUser,
			Source:  "bench",
		}); err != nil {
			b.Fatalf("Append: %v", err)
		}
	}
	for {
		drained, err := f.worker.DrainOnce(ctx)
		if err != nil {
			b.Fatalf("DrainOnce: %v", err)
		}
		if drained == 0 {
			return
		}
	}
}

// benchContent varies enough that neither channel is answering one repeated
// string: distinct vocabulary per episode, with recurring terms to match on.
func benchContent(i int) string {
	subjects := []string{"the billing service", "the search index", "the payment webhook",
		"the export job", "the session cache"}
	verbs := []string{"was deployed", "failed twice", "was rolled back",
		"needed a migration", "timed out"}
	return fmt.Sprintf("%s %s on day %d, and someone opened ticket %d about it",
		subjects[i%len(subjects)], verbs[(i/5)%len(verbs)], i%90, 1000+i)
}

// BenchmarkAppend measures the write path: validate, one round trip, return. No
// model call happens here, which is the property being measured.
func BenchmarkAppend(b *testing.B) {
	f := newBenchFixture(b, nil)
	ctx := b.Context()

	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		if _, err := f.memory.Append(ctx, mempher.AppendRequest{
			Scope:   "bench",
			Content: benchContent(i),
			Role:    mempher.RoleUser,
			Source:  "bench",
			Binding: mempher.Binding{"session": "s-1"},
		}); err != nil {
			b.Fatalf("Append: %v", err)
		}
	}
}

// BenchmarkAppendBatch measures what a round trip and a commit cost when they
// are shared. The per-episode number is what to compare against BenchmarkAppend.
func BenchmarkAppendBatch(b *testing.B) {
	for _, size := range []int{8, 32, 64} {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			f := newBenchFixture(b, nil)
			ctx := b.Context()

			reqs := make([]mempher.AppendRequest, size)
			b.ReportAllocs()
			for i := 0; b.Loop(); i++ {
				for j := range reqs {
					reqs[j] = mempher.AppendRequest{
						Scope:   "bench",
						Content: benchContent(i*size + j),
						Role:    mempher.RoleUser,
						Source:  "bench",
						Binding: mempher.Binding{"session": "s-1"},
					}
				}
				if _, err := f.memory.AppendBatch(ctx, reqs); err != nil {
					b.Fatalf("AppendBatch: %v", err)
				}
			}
			// Per episode, which is the number worth comparing.
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*size), "ns/episode")
		})
	}
}

// BenchmarkRecall measures the read loop over a fully encoded scope: embed the
// query, run every channel at once, fuse, budget.
func BenchmarkRecall(b *testing.B) {
	f := newBenchFixture(b, nil)
	ctx := b.Context()
	f.seed(ctx, b, benchSeed)

	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		got, err := f.memory.Recall(ctx, mempher.RecallRequest{
			Scope: "bench",
			Query: benchQuery(i),
			Limit: 10,
		})
		if err != nil {
			b.Fatalf("Recall: %v", err)
		}
		if len(got.Episodes) == 0 {
			b.Fatal("Recall returned nothing; the benchmark is measuring an empty scope")
		}
	}
}

// BenchmarkRecallSemantic and BenchmarkRecallLexical show where the time in a
// fused recall goes, since the channels run concurrently and the slowest one
// sets the floor.
func BenchmarkRecallSemantic(b *testing.B) {
	benchmarkChannel(b, mempher.ChannelSemantic)
}

func BenchmarkRecallLexical(b *testing.B) {
	benchmarkChannel(b, mempher.ChannelLexical)
}

func benchmarkChannel(b *testing.B, channel mempher.Channel) {
	f := newBenchFixture(b, nil)
	ctx := b.Context()
	f.seed(ctx, b, benchSeed)

	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		if _, err := f.memory.Recall(ctx, mempher.RecallRequest{
			Scope:    "bench",
			Query:    benchQuery(i),
			Limit:    10,
			Channels: []mempher.Channel{channel},
		}); err != nil {
			b.Fatalf("Recall: %v", err)
		}
	}
}

// BenchmarkRecallWithFacts adds the L1 channel, which is a lookup rather than a
// search and should barely register.
func BenchmarkRecallWithFacts(b *testing.B) {
	extractor := memphertest.NewExtractor().On(memphertest.Rule{
		Match: "billing",
		Assert: mempher.Assertion{
			Subject: "billing", Predicate: "owned_by", Object: "payments team",
			Statement:  "the billing service is owned by the payments team",
			Confidence: 0.9,
		},
	})
	f := newBenchFixture(b, extractor)
	ctx := b.Context()
	f.seed(ctx, b, benchSeed)

	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		got, err := f.memory.Recall(ctx, mempher.RecallRequest{
			Scope: "bench",
			Query: benchQuery(i),
			Limit: 10,
		})
		if err != nil {
			b.Fatalf("Recall: %v", err)
		}
		if len(got.Facts) == 0 {
			b.Fatal("no facts; the benchmark is not measuring L1")
		}
	}
}

// benchQuery varies the query so neither the database's plan cache nor its
// buffer cache is answering the same question every iteration.
func benchQuery(i int) string {
	queries := []string{
		"billing service deployed",
		"search index rolled back",
		"payment webhook timed out",
		"export job migration",
		"session cache failed",
	}
	return queries[i%len(queries)]
}
