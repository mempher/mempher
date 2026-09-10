package postgres_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mempher/mempher"
)

// The guard 0004 installed, tested at the level it lives at. What the library
// does with erasure is covered through the public API; what matters here is that
// the exception is exactly as narrow as it claims to be.

// erasing runs one statement inside a transaction that has declared an erasure,
// which is what the trigger looks for. The error comes back wrapped but intact,
// because what the caller inspects is the refusal the guard wrote.
func erasing(t *testing.T, pool *pgxpool.Pool, sql string) error {
	t.Helper()
	ctx := t.Context()
	err := pgx.BeginFunc(ctx, pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, "SET LOCAL mempher.erasing = 'on'"); err != nil {
			return fmt.Errorf("declare the erasure: %w", err)
		}
		if _, err := tx.Exec(ctx, sql); err != nil {
			return fmt.Errorf("run %q: %w", sql, err)
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("erasing transaction: %w", err)
	}
	return nil
}

func TestErasureIsTheOnlyExceptionToAppendOnly(t *testing.T) {
	t.Parallel()
	store, pool := migrated(t)
	seedScope(t, store, "user:1")

	// Declaring an erasure buys a DELETE and nothing else. A row that exists
	// is still exactly what was recorded.
	err := erasing(t, pool, "UPDATE mempher.episodes SET content = 'redacted'")
	assertAppendOnlyRefusal(t, err, "UPDATE")

	err = erasing(t, pool, "TRUNCATE mempher.episodes CASCADE")
	assertAppendOnlyRefusal(t, err, "TRUNCATE")

	// And a DELETE that declares nothing is refused as it always was.
	_, err = pool.Exec(t.Context(), "DELETE FROM mempher.episodes")
	assertAppendOnlyRefusal(t, err, "DELETE")
}

// TestErasureDeclarationDoesNotOutliveItsTransaction is the whole safety of the
// mechanism: the flag is LOCAL, so a pooled connection cannot carry it to the
// next caller.
func TestErasureDeclarationDoesNotOutliveItsTransaction(t *testing.T) {
	t.Parallel()
	store, pool := migrated(t)
	ctx := t.Context()

	seedScope(t, store, "user:1")
	seedScope(t, store, "user:2")

	// A real erasure, which sets the flag and commits.
	if _, err := store.Forget(ctx, mempher.ForgetRequest{Scope: "user:1"}); err != nil {
		t.Fatalf("Forget: %v", err)
	}

	// The very next statement on the pool must be refused again.
	_, err := pool.Exec(ctx, "DELETE FROM mempher.episodes WHERE scope_id = 'user:2'")
	assertAppendOnlyRefusal(t, err, "DELETE")

	var flag *string
	if err := pool.QueryRow(ctx,
		"SELECT current_setting('mempher.erasing', true)").Scan(&flag); err != nil {
		t.Fatalf("read the flag: %v", err)
	}
	if flag != nil && *flag == "on" {
		t.Errorf("mempher.erasing = %q outside its transaction", *flag)
	}

	var left int
	if err := pool.QueryRow(ctx,
		"SELECT count(*) FROM mempher.episodes WHERE scope_id = 'user:2'").Scan(&left); err != nil {
		t.Fatalf("count: %v", err)
	}
	if left != 1 {
		t.Errorf("scope user:2 has %d episodes, want the one it started with", left)
	}
}

// TestForgetCountsWhatItRemoved checks the report against the tables, because
// the report is the only evidence an erasure leaves.
func TestForgetCountsWhatItRemoved(t *testing.T) {
	t.Parallel()
	store, pool := migrated(t)
	ctx := t.Context()

	seedScope(t, store, "user:1")
	seedScope(t, store, "user:1")
	seedScope(t, store, "user:2")

	result, err := store.Forget(ctx, mempher.ForgetRequest{Scope: "user:1"})
	if err != nil {
		t.Fatalf("Forget: %v", err)
	}
	if result.Episodes != 2 {
		t.Errorf("Episodes = %d, want 2", result.Episodes)
	}
	// Append enqueues one encode job per episode, and they go with them.
	if result.Jobs != 2 {
		t.Errorf("Jobs = %d, want the two encode jobs those episodes implied", result.Jobs)
	}
	if !result.ScopeRemoved {
		t.Error("ScopeRemoved = false after erasing a whole scope")
	}

	var episodes, jobs, scopes int
	if err := pool.QueryRow(ctx, `
		SELECT
		    (SELECT count(*) FROM mempher.episodes WHERE scope_id = 'user:1'),
		    (SELECT count(*) FROM mempher.jobs     WHERE scope_id = 'user:1'),
		    (SELECT count(*) FROM mempher.scopes   WHERE id       = 'user:1')`,
	).Scan(&episodes, &jobs, &scopes); err != nil {
		t.Fatalf("count: %v", err)
	}
	if episodes+jobs+scopes != 0 {
		t.Errorf("%d episodes, %d jobs and %d catalogue rows left, want none",
			episodes, jobs, scopes)
	}
}

func TestForgetRejectsBadRequestsAtTheAdapter(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)

	_, err := store.Forget(t.Context(), mempher.ForgetRequest{})
	if !errors.Is(err, mempher.ErrInvalidScope) {
		t.Errorf("no scope err = %v, want mempher.ErrInvalidScope", err)
	}
}

// assertAppendOnlyRefusal insists the refusal came from the guard rather than
// from a syntax error or a missing table, which would pass a test for the wrong
// reason.
func assertAppendOnlyRefusal(t *testing.T, err error, op string) {
	t.Helper()
	if err == nil {
		t.Fatalf("%s on mempher.episodes succeeded, want the append-only guard to refuse it", op)
	}
	if got := err.Error(); !strings.Contains(got, "append-only") || !strings.Contains(got, op) {
		t.Errorf("err = %v, want the append-only guard refusing %s", err, op)
	}
}
