// Command quickstart is mempher end to end in one file: append, consolidate,
// recall, watch a fact get bounded in time rather than overwritten, and erase.
//
// It needs one thing, a PostgreSQL 18 with pgvector:
//
//	docker run --rm -p 5432:5432 -e POSTGRES_PASSWORD=mempher pgvector/pgvector:pg18
//	go run ./examples/quickstart
//
// The Embedder and Extractor here come from memphertest: a hashed bag of words
// and a handful of substring rules. They are deterministic and need no API key,
// which is what makes this runnable -- and they are the two interfaces you
// replace with a real model client. Nothing else in this file changes when you
// do.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mempher/mempher"
	"github.com/mempher/mempher/memphertest"
	"github.com/mempher/mempher/postgres"
)

const scope = "user:8123"

func main() {
	ctx := context.Background()

	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		dsn = "postgres://postgres:mempher@localhost:5432/postgres?sslmode=disable"
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	// A toy embedder and a toy extractor, so this runs with no API key. Swap
	// them for your own; they are the only two interfaces that call a model.
	embedder := memphertest.NewEmbedder(256)
	extractor := memphertest.NewExtractor().On(
		livesIn("paris", "Paris", date(2019, 4, 1)),
		livesIn("berlin", "Berlin", date(2024, 6, 1)),
		memphertest.Rule{
			Match: "hazelnut",
			Assert: mempher.Assertion{
				Subject: "user", Predicate: "allergic_to", Object: "hazelnuts",
				Statement:  "the user is allergic to hazelnuts",
				Confidence: 0.95,
			},
		},
	)

	// Embedded, forward-only, and safe to call from every instance at startup.
	migrated, err := postgres.Migrate(ctx, pool, postgres.MigrateOptions{
		VectorDimensions: embedder.Dimensions(),
	})
	if err != nil {
		log.Fatalf("migrate: %v", err)
	}
	fmt.Printf("schema at version %d (%d migrations applied by this run)\n\n",
		migrated.Version, len(migrated.Applied))

	store, err := postgres.New(ctx, pool)
	if err != nil {
		log.Fatalf("store: %v", err)
	}

	memory, err := mempher.New(mempher.Config{
		Store: store, Embedder: embedder, Extractor: extractor,
	})
	if err != nil {
		log.Fatalf("memory: %v", err)
	}
	worker, err := mempher.NewWorker(mempher.WorkerConfig{
		Store: store, Embedder: embedder, Extractor: extractor,
	})
	if err != nil {
		log.Fatalf("worker: %v", err)
	}

	// ---------------------------------------------------------------- write
	// No model call, one round trip. OccurredAt is when it happened in the
	// world, which is not when we heard it.
	say(ctx, memory, "I moved to Paris last spring.", date(2019, 4, 2))
	say(ctx, memory, "I'm allergic to hazelnuts, so no Nutella.", date(2019, 5, 10))
	say(ctx, memory, "Deployed the billing service this morning.", date(2024, 1, 9))
	fmt.Print("appended 3 episodes -- already searchable lexically\n\n")

	// ---------------------------------------------------------- consolidate
	// Every model call in the library lives here, off the write path.
	drained, err := worker.DrainOnce(ctx)
	if err != nil {
		log.Fatalf("drain: %v", err)
	}
	fmt.Printf("worker drained %d jobs -- now searchable semantically, and facts exist\n\n", drained)

	// ----------------------------------------------------------------- read
	show(ctx, memory, "allergic to hazelnuts")

	// ------------------------------------------------- the world moves on
	say(ctx, memory, "Actually I've moved to Berlin now.", date(2024, 6, 2))
	if _, err := worker.DrainOnce(ctx); err != nil {
		log.Fatalf("drain: %v", err)
	}
	fmt.Println("---- after moving to Berlin ----")
	show(ctx, memory, "moved to Berlin")

	// The old claim was not overwritten. Ask about last year and it answers.
	fmt.Println("---- and where did they live in 2020? ----")
	recalled, err := memory.Recall(ctx, mempher.RecallRequest{
		Scope: scope, Query: "where do they live?",
		// An event-time cut: facts are answered as of this instant, not today.
		OccurredTo: date(2020, 1, 1),
	})
	if err != nil {
		log.Fatalf("recall: %v", err)
	}
	for _, f := range recalled.Facts {
		fmt.Printf("  fact  %s  %s\n", f.Fact.Statement, window(f.Fact.Valid))
	}
	fmt.Println()

	// --------------------------------------------------------------- erase
	// The one operation here that destroys anything.
	erased, err := memory.Forget(ctx, mempher.ForgetRequest{Scope: scope})
	if err != nil {
		log.Fatalf("forget: %v", err)
	}
	fmt.Printf("erased: %d episodes, %d encodings, %d facts, %d jobs (scope removed: %t)\n",
		erased.Episodes, erased.Encodings, erased.Facts, erased.Jobs, erased.ScopeRemoved)
}

// say appends one episode at a given event time.
func say(ctx context.Context, memory *mempher.Memory, content string, at time.Time) {
	if _, err := memory.Append(ctx, mempher.AppendRequest{
		Scope:      scope,
		Content:    content,
		Role:       mempher.RoleUser,
		Source:     "quickstart",
		OccurredAt: at,
		Binding:    mempher.Binding{"session": "s-42"},
	}); err != nil {
		log.Fatalf("append: %v", err)
	}
}

// show recalls and prints both layers with their provenance.
func show(ctx context.Context, memory *mempher.Memory, query string) {
	recalled, err := memory.Recall(ctx, mempher.RecallRequest{
		Scope: scope, Query: query, Limit: 3, MaxTokens: 500,
	})
	if err != nil {
		log.Fatalf("recall: %v", err)
	}

	fmt.Printf("query: %q\n", query)
	for _, f := range recalled.Facts {
		fmt.Printf("  fact     %s  %s\n", f.Fact.Statement, window(f.Fact.Valid))
	}
	// Scores are reciprocal rank fusion, so they are small by construction:
	// what carries meaning is the order and which channels agreed.
	for _, r := range recalled.Episodes {
		fmt.Printf("  episode  %.4f  %q  via %s\n", r.Score, r.Episode.Content, channels(r.Hits))
	}
	for _, c := range recalled.Channels {
		if c.Err != nil {
			fmt.Printf("  channel %s failed: %v\n", c.Channel, c.Err)
		}
	}
	fmt.Printf("  (%d tokens)\n\n", recalled.Tokens)
}

// window renders a validity window the way a reader thinks about one.
func window(v mempher.Validity) string {
	if v.IsOpen() {
		return "[since " + v.From.Format("2006-01-02") + ", still true]"
	}
	return "[" + v.From.Format("2006-01-02") + " to " + v.To.Format("2006-01-02") + "]"
}

// channels names which retrieval paths found an episode: the provenance that
// keeps a thin result from looking like an empty memory.
func channels(hits []mempher.ChannelHit) string {
	out := ""
	for i, hit := range hits {
		if i > 0 {
			out += "+"
		}
		out += hit.Channel.String()
	}
	return out
}

// livesIn is a rule where a new home supersedes the old one, which is the case
// the whole temporal design exists for.
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

func date(year int, month time.Month, day int) time.Time {
	return time.Date(year, month, day, 12, 0, 0, 0, time.UTC)
}
