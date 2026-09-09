package pgtest_test

import (
	"testing"

	"github.com/mempher/mempher/internal/pgtest"
)

// TestPoolProvidesTheDatabaseTheSchemaNeeds checks the three things the
// migration depends on, so a broken harness fails here with a clear message
// instead of somewhere inside a migration.
func TestPoolProvidesTheDatabaseTheSchemaNeeds(t *testing.T) {
	t.Parallel()

	pool := pgtest.Pool(t)
	ctx := t.Context()

	var version int
	if err := pool.QueryRow(ctx, "SELECT current_setting('server_version_num')::int").
		Scan(&version); err != nil {
		t.Fatalf("read server version: %v", err)
	}
	if version < 180000 {
		t.Fatalf("server_version_num = %d, want at least 180000", version)
	}

	for _, ext := range []string{"vector", "btree_gin"} {
		if _, err := pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS "+ext); err != nil {
			t.Fatalf("create extension %s: %v", ext, err)
		}
	}

	// uuidv7 is why the schema requires PostgreSQL 18 at all, and
	// uuid_extract_timestamp is how tests assert episode ordering.
	var ordered bool
	if err := pool.QueryRow(ctx, `
		SELECT uuid_extract_timestamp(uuidv7()) <= uuid_extract_timestamp(uuidv7())
	`).Scan(&ordered); err != nil {
		t.Fatalf("uuidv7 round trip: %v", err)
	}
	if !ordered {
		t.Error("uuidv7 timestamps went backwards")
	}

	// A 3-dimensional cosine distance between orthogonal vectors is 1.
	var distance float64
	if err := pool.QueryRow(ctx, `SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector`).
		Scan(&distance); err != nil {
		t.Fatalf("pgvector cosine distance: %v", err)
	}
	if distance != 1 {
		t.Errorf("cosine distance = %v, want 1", distance)
	}
}

// TestPoolIsolatesEachTest proves the per-database isolation the harness
// promises: both subtests create the same table and neither sees the other.
func TestPoolIsolatesEachTest(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"first", "second"} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			pool := pgtest.Pool(t)
			ctx := t.Context()

			if _, err := pool.Exec(ctx, "CREATE TABLE only_mine (who text)"); err != nil {
				t.Fatalf("create table: %v", err)
			}
			if _, err := pool.Exec(ctx, "INSERT INTO only_mine VALUES ($1)", name); err != nil {
				t.Fatalf("insert: %v", err)
			}

			var rows int
			if err := pool.QueryRow(ctx, "SELECT count(*) FROM only_mine").Scan(&rows); err != nil {
				t.Fatalf("count: %v", err)
			}
			if rows != 1 {
				t.Errorf("saw %d rows, want only this test's own", rows)
			}

			var database string
			if err := pool.QueryRow(ctx, "SELECT current_database()").Scan(&database); err != nil {
				t.Fatalf("current_database: %v", err)
			}
			t.Logf("isolated database: %s", database)
		})
	}
}
