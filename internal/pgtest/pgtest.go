// Package pgtest starts a real PostgreSQL 18 with pgvector for integration tests
// and hands each test its own empty database on it.
//
// Retrieval depends on pgvector's distance ordering, PostgreSQL's text search
// ranking, uuidv7 monotonicity and the append-only trigger. A fake store would
// have to reimplement all of it, and would reimplement it wrongly.
//
// Each test gets a fresh database rather than a fresh schema, because the
// migration hardcodes the schema name. That buys complete isolation, including
// per-database extensions, and lets tests run in parallel.
//
// This package is internal on purpose: nobody depending on mempher should
// inherit a testcontainers dependency.
package pgtest

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/testcontainers/testcontainers-go"
	tcpostgres "github.com/testcontainers/testcontainers-go/modules/postgres"
)

// Image is the database every integration test runs against: PostgreSQL 18 for
// uuidv7(), the pgvector image for the extension.
const Image = "pgvector/pgvector:pg18"

const (
	adminDatabase = "postgres"
	startTimeout  = 3 * time.Minute
	adminTimeout  = 30 * time.Second
)

// One container serves the whole test binary; the databases on it isolate tests.
// Testcontainers' Ryuk sidecar reaps it when the process exits, so there is no
// TestMain to forget.
var (
	startOnce sync.Once
	adminDSN  string
	startErr  error
	dbSeq     atomic.Uint64
)

// Pool returns a pool onto a fresh, empty database and registers its teardown
// with t. The database has no mempher schema, so a test can exercise migration
// behaviour too.
//
// It takes a [testing.TB] rather than a *testing.T so that benchmarks get the
// same isolated database a test does: measuring append latency against a
// database another benchmark is writing to would measure the other benchmark.
//
// It skips under -short, which is the single place that guard lives.
func Pool(t testing.TB) *pgxpool.Pool {
	t.Helper()
	if testing.Short() {
		t.Skipf("pgtest: skipped under -short (needs Docker and %s)", Image)
	}

	startOnce.Do(func() {
		// Not t.Context(): the container outlives whichever test started it.
		ctx, cancel := context.WithTimeout(context.Background(), startTimeout)
		defer cancel()
		adminDSN, startErr = start(ctx)
	})
	if startErr != nil {
		t.Fatalf("pgtest: start %s: %v", Image, startErr)
	}

	name := databaseName(t)
	if err := adminExec(t.Context(), fmt.Sprintf("CREATE DATABASE %s", quoteIdent(name))); err != nil {
		t.Fatalf("pgtest: create database %s: %v", name, err)
	}

	pool, err := pgxpool.New(t.Context(), replaceDatabase(adminDSN, name))
	if err != nil {
		t.Fatalf("pgtest: connect to %s: %v", name, err)
	}
	if err := pool.Ping(t.Context()); err != nil {
		t.Fatalf("pgtest: ping %s: %v", name, err)
	}

	t.Cleanup(func() {
		pool.Close()
		// t.Context() is already cancelled by the time cleanup runs.
		ctx, cancel := context.WithTimeout(context.Background(), adminTimeout)
		defer cancel()
		// FORCE terminates anything the test leaked, which would otherwise
		// turn teardown into a hang.
		stmt := fmt.Sprintf("DROP DATABASE IF EXISTS %s WITH (FORCE)", quoteIdent(name))
		if err := adminExec(ctx, stmt); err != nil {
			t.Errorf("pgtest: drop database %s: %v", name, err)
		}
	})
	return pool
}

// start brings up the container and returns a DSN for its maintenance database.
func start(ctx context.Context) (string, error) {
	container, err := tcpostgres.Run(ctx, Image,
		tcpostgres.WithDatabase(adminDatabase),
		tcpostgres.WithUsername("mempher"),
		tcpostgres.WithPassword("mempher"),
		// This database is thrown away, so durability is wasted time.
		testcontainers.WithCmd("postgres", "-c", "fsync=off", "-c", "full_page_writes=off"),
		tcpostgres.BasicWaitStrategies(),
	)
	if err != nil {
		return "", fmt.Errorf("run container: %w", err)
	}
	dsn, err := container.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		return "", fmt.Errorf("connection string: %w", err)
	}
	return dsn, nil
}

// adminExec runs one statement on its own connection: CREATE and DROP DATABASE
// cannot run in a transaction, nor on a connection to the database being dropped.
func adminExec(ctx context.Context, stmt string) error {
	conn, err := pgx.Connect(ctx, adminDSN)
	if err != nil {
		return fmt.Errorf("connect to %s: %w", adminDatabase, err)
	}
	defer func() { _ = conn.Close(ctx) }()
	if _, err := conn.Exec(ctx, stmt); err != nil {
		return fmt.Errorf("exec %q: %w", stmt, err)
	}
	return nil
}

// databaseName derives a unique, legal, recognisable identifier.
func databaseName(t testing.TB) string {
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '_':
			return r
		case r >= 'A' && r <= 'Z':
			return r + ('a' - 'A')
		default:
			return '_'
		}
	}, t.Name())
	if len(safe) > 40 {
		safe = safe[:40]
	}
	return fmt.Sprintf("mempher_%s_%d", safe, dbSeq.Add(1))
}

// quoteIdent quotes an identifier, since CREATE DATABASE takes no parameters.
func quoteIdent(name string) string {
	return `"` + strings.ReplaceAll(name, `"`, `""`) + `"`
}

// replaceDatabase rewrites the database in a postgres:// DSN.
func replaceDatabase(dsn, database string) string {
	slash := strings.LastIndex(dsn, "/")
	if slash < 0 {
		return dsn
	}
	tail := ""
	if q := strings.Index(dsn[slash:], "?"); q >= 0 {
		tail = dsn[slash+q:]
	}
	return dsn[:slash+1] + database + tail
}
