package postgres_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mempher/mempher"
	"github.com/mempher/mempher/internal/pgtest"
	"github.com/mempher/mempher/postgres"
)

// testDimensions is deliberately tiny: HNSW is happy with it, and every test
// that builds the schema pays for the index.
const testDimensions = 8

func testOptions() postgres.MigrateOptions {
	return postgres.MigrateOptions{
		VectorDimensions: testDimensions,
		TextSearchConfig: "english",
		LockTimeout:      10 * time.Second,
	}
}

func mustMigrate(t *testing.T, pool *pgxpool.Pool) postgres.MigrateResult {
	t.Helper()
	result, err := postgres.Migrate(t.Context(), pool, testOptions())
	if err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	return result
}

func TestMigrateFreshDatabase(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	result := mustMigrate(t, pool)

	// Asserted as a property rather than against a literal version, so that
	// adding a migration does not mean editing this test: a fresh database
	// gets the whole embedded set, contiguously, from 1.
	if len(result.Applied) == 0 {
		t.Fatalf("Applied = %v, want every embedded migration", result.Applied)
	}
	for i, version := range result.Applied {
		if want := i + 1; version != want {
			t.Errorf("Applied[%d] = %d, want %d (got %v)", i, version, want, result.Applied)
		}
	}
	if want := result.Applied[len(result.Applied)-1]; result.Version != want {
		t.Errorf("Version = %d, want %d, the last migration applied", result.Version, want)
	}
	if result.ExtensionSchema != postgres.DefaultExtensionSchema {
		t.Errorf("ExtensionSchema = %q, want %q",
			result.ExtensionSchema, postgres.DefaultExtensionSchema)
	}
}

// TestMigrateIsIdempotent is the property every startup path depends on.
func TestMigrateIsIdempotent(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	first := mustMigrate(t, pool)

	second, err := postgres.Migrate(t.Context(), pool, testOptions())
	if err != nil {
		t.Fatalf("second Migrate: %v", err)
	}
	if len(second.Applied) != 0 {
		t.Errorf("second run applied %v, want nothing", second.Applied)
	}
	if second.Version != first.Version {
		t.Errorf("Version = %d after the second run, want %d", second.Version, first.Version)
	}

	// One ledger row per migration, however many there are: a second run that
	// re-recorded history would show up here as a doubled count.
	var ledger int
	if err := pool.QueryRow(t.Context(),
		"SELECT count(*) FROM mempher.schema_migrations").Scan(&ledger); err != nil {
		t.Fatalf("count ledger: %v", err)
	}
	if ledger != first.Version {
		t.Errorf("ledger has %d rows after two runs, want %d", ledger, first.Version)
	}
}

// TestMigrateBuildsAWorkingSchema checks the migration produced the objects the
// rest of the adapter is about to rely on, including the trigger that makes the
// append-only invariant the database's rule rather than a convention.
func TestMigrateBuildsAWorkingSchema(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	mustMigrate(t, pool)
	ctx := t.Context()

	t.Run("tables", func(t *testing.T) {
		want := []string{
			"episodes", "episode_encodings", "facts", "fact_extractions",
			"jobs", "schema_config", "schema_migrations", "scopes",
		}
		for _, table := range want {
			var exists bool
			if err := pool.QueryRow(ctx, `
				SELECT EXISTS (
					SELECT 1 FROM pg_tables WHERE schemaname = 'mempher' AND tablename = $1
				)`, table).Scan(&exists); err != nil {
				t.Fatalf("look up %s: %v", table, err)
			}
			if !exists {
				t.Errorf("mempher.%s was not created", table)
			}
		}
	})

	t.Run("indexes", func(t *testing.T) {
		var hnsw, gin, gist int
		if err := pool.QueryRow(ctx, `
			SELECT
				count(*) FILTER (WHERE indexdef LIKE '%USING hnsw%'),
				count(*) FILTER (WHERE indexdef LIKE '%USING gin%'),
				count(*) FILTER (WHERE indexdef LIKE '%USING gist%')
			FROM pg_indexes WHERE schemaname = 'mempher'
		`).Scan(&hnsw, &gin, &gist); err != nil {
			t.Fatalf("read indexes: %v", err)
		}
		if hnsw != 1 {
			t.Errorf("found %d HNSW indexes, want 1", hnsw)
		}
		if gin < 2 {
			t.Errorf("found %d GIN indexes, want at least 2 (tsvector and binding)", gin)
		}
		// The facts temporal primary key. Without btree_gist it could not be
		// built, so its absence means the extension went missing.
		if gist < 1 {
			t.Errorf("found %d GiST indexes, want the facts temporal key", gist)
		}
	})

	t.Run("schema config records the deployment decisions", func(t *testing.T) {
		var dims int
		var config string
		if err := pool.QueryRow(ctx,
			"SELECT vector_dimensions, text_search_config FROM mempher.schema_config").
			Scan(&dims, &config); err != nil {
			t.Fatalf("read schema_config: %v", err)
		}
		if dims != testDimensions || config != "english" {
			t.Errorf("schema_config = (%d, %q), want (%d, %q)",
				dims, config, testDimensions, "english")
		}
	})

	t.Run("L0 is append-only", func(t *testing.T) {
		if _, err := pool.Exec(ctx,
			"INSERT INTO mempher.scopes (id, last_seq, created_at) VALUES ('s', 1, now())"); err != nil {
			t.Fatalf("seed scope: %v", err)
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO mempher.episodes
				(scope_id, seq, content, role, source, occurred_at, ingested_at)
			VALUES ('s', 1, 'happened', 'user', 'test', now(), now())
		`); err != nil {
			t.Fatalf("seed episode: %v", err)
		}

		for _, stmt := range []string{
			"UPDATE mempher.episodes SET content = 'rewritten'",
			"DELETE FROM mempher.episodes",
			"TRUNCATE mempher.episodes CASCADE",
			// A statement matching no rows must fail too, which is why the
			// trigger is statement-level.
			"UPDATE mempher.episodes SET content = 'x' WHERE scope_id = 'nope'",
		} {
			if _, err := pool.Exec(ctx, stmt); err == nil {
				t.Errorf("%q was permitted; L0 must be append-only", stmt)
			}
		}

		var remaining int
		if err := pool.QueryRow(ctx, "SELECT count(*) FROM mempher.episodes").Scan(&remaining); err != nil {
			t.Fatalf("count episodes: %v", err)
		}
		if remaining != 1 {
			t.Errorf("%d episodes remain, want the original 1", remaining)
		}
	})

	// The invariant L1 adds: one scope cannot believe the same claim over two
	// overlapping windows. It is the temporal primary key that enforces it,
	// which is the whole reason this library requires PostgreSQL 18.
	t.Run("L1 rejects overlapping validity windows", func(t *testing.T) {
		if _, err := pool.Exec(ctx,
			"INSERT INTO mempher.scopes (id, last_seq, created_at) VALUES ('l1', 0, now())"); err != nil {
			t.Fatalf("seed scope: %v", err)
		}

		insert := func(object string, window string) error {
			_, err := pool.Exec(ctx, `
				INSERT INTO mempher.facts
					(scope_id, extractor, subject, predicate, object,
					 statement, valid, episode_ids, asserted_at, updated_at)
				VALUES ('l1', 'test-extractor', 'user', 'lives_in', $1,
					'the user lives somewhere', `+window+`,
					ARRAY[uuidv7()], now(), now())
			`, object)
			if err != nil {
				return fmt.Errorf("insert %s over %s: %w", object, window, err)
			}
			return nil
		}

		const (
			until2024 = "tstzrange('2019-01-01Z', '2024-01-01Z', '[)')"
			from2023  = "tstzrange('2023-01-01Z', NULL, '[)')"
			from2024  = "tstzrange('2024-01-01Z', NULL, '[)')"
		)

		if err := insert("Paris", until2024); err != nil {
			t.Fatalf("first fact: %v", err)
		}
		// Same claim, overlapping window: the contradiction the key exists for.
		if err := insert("Paris", from2023); err == nil {
			t.Error("an overlapping window for the same claim was permitted")
		}
		// Adjacent, not overlapping. Half-open bounds are what make a change of
		// state need no gap between the old window and the new one.
		if err := insert("Paris", from2024); err != nil {
			t.Errorf("an adjacent window was rejected: %v", err)
		}
		// A different object of the same predicate is a different fact, because
		// two of them can be true at once. Only the extractor knows whether one
		// replaces the other, so it says so with a retraction.
		if err := insert("Berlin", from2023); err != nil {
			t.Errorf("a second object for one predicate was rejected: %v", err)
		}

		// The window shape the key depends on. An inclusive upper bound would
		// let two "adjacent" windows share an instant and slip past the
		// overlap check.
		for _, bad := range []struct {
			why    string
			window string
		}{
			{"an inclusive upper bound", "tstzrange('2030-01-01Z', '2031-01-01Z', '[]')"},
			{"an unbounded start", "tstzrange(NULL, '2031-01-01Z', '[)')"},
			{"an empty window", "tstzrange('2030-01-01Z', '2030-01-01Z', '[)')"},
		} {
			if err := insert("Lisbon", bad.window); err == nil {
				t.Errorf("%s was permitted", bad.why)
			}
		}
	})

	// Adding a job kind is a forward migration, so the CHECK has to have moved.
	t.Run("the queue accepts the extract kind", func(t *testing.T) {
		if _, err := pool.Exec(ctx,
			"INSERT INTO mempher.scopes (id, last_seq, created_at) VALUES ('q2', 0, now())"); err != nil {
			t.Fatalf("seed scope: %v", err)
		}
		for _, kind := range []string{"encode", "extract"} {
			if _, err := pool.Exec(ctx, `
				INSERT INTO mempher.jobs
					(kind, scope_id, episode_ids, run_after, created_at, updated_at)
				VALUES ($1, 'q2', ARRAY[uuidv7()], now(), now(), now())
			`, kind); err != nil {
				t.Errorf("enqueue %q: %v", kind, err)
			}
		}
		if _, err := pool.Exec(ctx, `
			INSERT INTO mempher.jobs
				(kind, scope_id, episode_ids, run_after, created_at, updated_at)
			VALUES ('nonesuch', 'q2', ARRAY[uuidv7()], now(), now(), now())
		`); err == nil {
			t.Error("an undefined job kind was permitted; the set must stay closed")
		}
	})
}

// TestMigrateConcurrent is the reason the advisory lock exists: every instance of
// an application calls Migrate at startup, at the same time, against one
// database.
func TestMigrateConcurrent(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	const racers = 4

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		results []postgres.MigrateResult
		errs    []error
	)
	start := make(chan struct{})
	for range racers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			result, err := postgres.Migrate(t.Context(), pool, testOptions())
			mu.Lock()
			defer mu.Unlock()
			results = append(results, result)
			errs = append(errs, err)
		}()
	}
	close(start)
	wg.Wait()

	appliedBy, version := 0, 0
	for i, err := range errs {
		if err != nil {
			t.Errorf("racer %d failed: %v", i, err)
			continue
		}
		if len(results[i].Applied) > 0 {
			appliedBy++
		}
		// Every racer must come away with the same version, whichever one of
		// them did the work: a racer that returned before the migration it
		// waited on had committed is the bug this catches.
		switch {
		case version == 0:
			version = results[i].Version
		case results[i].Version != version:
			t.Errorf("racer %d saw version %d, but another saw %d",
				i, results[i].Version, version)
		}
	}
	if appliedBy != 1 {
		t.Errorf("%d racers applied migrations, want exactly 1", appliedBy)
	}

	var ledger int
	if err := pool.QueryRow(t.Context(),
		"SELECT count(*) FROM mempher.schema_migrations").Scan(&ledger); err != nil {
		t.Fatalf("count ledger: %v", err)
	}
	if ledger != version {
		t.Errorf("ledger has %d rows, want %d", ledger, version)
	}
}

// TestMigrateReleasesItsLockAndConnection checks the two resources that would
// poison a long-lived process if they leaked.
func TestMigrateReleasesItsLockAndConnection(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	mustMigrate(t, pool)

	if inUse := pool.Stat().AcquiredConns(); inUse != 0 {
		t.Errorf("%d connections still acquired after Migrate, want 0", inUse)
	}

	// If the advisory lock were still held, this would return false.
	var got bool
	if err := pool.QueryRow(t.Context(),
		"SELECT pg_try_advisory_lock(8348320501231727730)").Scan(&got); err != nil {
		t.Fatalf("try advisory lock: %v", err)
	}
	if !got {
		t.Error("the migration advisory lock is still held")
	}
	if _, err := pool.Exec(t.Context(),
		"SELECT pg_advisory_unlock(8348320501231727730)"); err != nil {
		t.Fatalf("unlock: %v", err)
	}
}

func TestMigrateLockTimeout(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	ctx := t.Context()

	// Hold the lock on a connection of our own, as a competing migration
	// would, and keep holding it for the duration of the test.
	holder, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire holder: %v", err)
	}
	defer holder.Release()
	var held bool
	if err := holder.QueryRow(ctx,
		"SELECT pg_try_advisory_lock(8348320501231727730)").Scan(&held); err != nil {
		t.Fatalf("hold lock: %v", err)
	}
	if !held {
		t.Fatal("could not take the lock to hold it")
	}
	defer func() {
		_, _ = holder.Exec(context.WithoutCancel(ctx),
			"SELECT pg_advisory_unlock(8348320501231727730)")
	}()

	opts := testOptions()
	opts.LockTimeout = 150 * time.Millisecond

	started := time.Now()
	_, err = postgres.Migrate(ctx, pool, opts)
	elapsed := time.Since(started)

	if !errors.Is(err, postgres.ErrLockTimeout) {
		t.Fatalf("err = %v, want ErrLockTimeout", err)
	}
	if elapsed > 5*time.Second {
		t.Errorf("waited %s for a %s timeout", elapsed, opts.LockTimeout)
	}
	// Even on the failure path the connection must go back to the pool.
	if inUse := pool.Stat().AcquiredConns(); inUse != 1 {
		t.Errorf("%d connections acquired, want only the test's holder", inUse)
	}
}

// TestMigrateRejectsEditedHistory is the forward-only guarantee.
func TestMigrateRejectsEditedHistory(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	mustMigrate(t, pool)

	if _, err := pool.Exec(t.Context(),
		"UPDATE mempher.schema_migrations SET checksum = 'tampered' WHERE version = 1"); err != nil {
		t.Fatalf("tamper with the ledger: %v", err)
	}

	_, err := postgres.Migrate(t.Context(), pool, testOptions())
	if !errors.Is(err, postgres.ErrChecksumMismatch) {
		t.Fatalf("err = %v, want ErrChecksumMismatch", err)
	}
}

func TestMigrateRejectsANewerDatabase(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	mustMigrate(t, pool)

	if _, err := pool.Exec(t.Context(), `
		INSERT INTO mempher.schema_migrations (version, name, checksum, duration_ms)
		VALUES (99, 'from_the_future', 'whatever', 0)
	`); err != nil {
		t.Fatalf("insert future migration: %v", err)
	}

	_, err := postgres.Migrate(t.Context(), pool, testOptions())
	if !errors.Is(err, postgres.ErrDatabaseAhead) {
		t.Fatalf("err = %v, want ErrDatabaseAhead", err)
	}
}

// TestMigrateRejectsChangedSchemaOptions covers the mistake this check exists
// for: swapping the embedding model for one of a different width.
func TestMigrateRejectsChangedSchemaOptions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*postgres.MigrateOptions)
	}{
		{
			name:   "a wider embedding",
			mutate: func(o *postgres.MigrateOptions) { o.VectorDimensions = testDimensions * 2 },
		},
		{
			name:   "a different language",
			mutate: func(o *postgres.MigrateOptions) { o.TextSearchConfig = "french" },
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			pool := pgtest.Pool(t)
			mustMigrate(t, pool)

			changed := testOptions()
			tc.mutate(&changed)
			_, err := postgres.Migrate(t.Context(), pool, changed)
			if !errors.Is(err, postgres.ErrSchemaConfigMismatch) {
				t.Fatalf("err = %v, want ErrSchemaConfigMismatch", err)
			}
		})
	}
}

func TestMigrateRejectsUnknownTextSearchConfig(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	opts := testOptions()
	opts.TextSearchConfig = "klingon"

	_, err := postgres.Migrate(t.Context(), pool, opts)
	if !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Fatalf("err = %v, want mempher.ErrInvalidConfig", err)
	}
	// It must fail before creating anything.
	var schemas int
	if err := pool.QueryRow(t.Context(),
		"SELECT count(*) FROM pg_namespace WHERE nspname = 'mempher'").Scan(&schemas); err != nil {
		t.Fatalf("count schemas: %v", err)
	}
	if schemas != 0 {
		t.Error("the mempher schema was created despite invalid options")
	}
}

// TestMigrateDiscoversExtensionsInAnotherSchema covers a real deployment: a DBA
// has already installed pgvector somewhere other than public, and mempher must
// use it rather than assume.
func TestMigrateDiscoversExtensionsInAnotherSchema(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	ctx := t.Context()

	for _, stmt := range []string{
		"CREATE SCHEMA extras",
		"CREATE EXTENSION vector SCHEMA extras",
		"CREATE EXTENSION btree_gin SCHEMA extras",
	} {
		if _, err := pool.Exec(ctx, stmt); err != nil {
			t.Fatalf("%s: %v", stmt, err)
		}
	}

	result, err := postgres.Migrate(ctx, pool, testOptions())
	if err != nil {
		t.Fatalf("Migrate with extensions in extras: %v", err)
	}
	if result.ExtensionSchema != "extras" {
		t.Errorf("ExtensionSchema = %q, want %q", result.ExtensionSchema, "extras")
	}

	// Asking for the wrong place must be reported, not quietly corrected.
	wrong := testOptions()
	wrong.ExtensionSchema = "public"
	if _, err := postgres.Migrate(ctx, pool, wrong); !errors.Is(err, postgres.ErrSchemaConfigMismatch) {
		t.Errorf("err = %v, want ErrSchemaConfigMismatch", err)
	}
}

func TestMigrateRejectsBadArguments(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)

	if _, err := postgres.Migrate(t.Context(), nil, testOptions()); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("nil pool err = %v, want mempher.ErrInvalidConfig", err)
	}
	if _, err := postgres.Migrate(t.Context(), pool, postgres.MigrateOptions{}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("zero options err = %v, want mempher.ErrInvalidConfig", err)
	}
}

// TestMigrateHonoursCancellation checks a cancelled startup does not leave the
// pool or the lock in a bad state.
func TestMigrateHonoursCancellation(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	if _, err := postgres.Migrate(ctx, pool, testOptions()); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if inUse := pool.Stat().AcquiredConns(); inUse != 0 {
		t.Errorf("%d connections still acquired, want 0", inUse)
	}

	// The database must be untouched and still migratable.
	if _, err := postgres.Migrate(t.Context(), pool, testOptions()); err != nil {
		t.Fatalf("Migrate after a cancelled attempt: %v", err)
	}
}

func ExampleMigrate() {
	// Shown rather than run: it needs a database.
	fmt.Println("postgres.Migrate(ctx, pool, postgres.MigrateOptions{VectorDimensions: emb.Dimensions()})")
	// Output:
	// postgres.Migrate(ctx, pool, postgres.MigrateOptions{VectorDimensions: emb.Dimensions()})
}
