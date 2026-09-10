package postgres_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/postgres"
)

// appendN appends n episodes to a scope, so that its last_seq is known.
func appendN(t *testing.T, store *postgres.Store, scope mempher.ScopeID, n int) {
	t.Helper()
	for range n {
		if _, err := store.Append(t.Context(), episode(scope, "an episode", epoch)); err != nil {
			t.Fatalf("Append to %q: %v", scope, err)
		}
	}
}

func TestScopesListsThePartitionsOfL0(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	appendN(t, store, "user:2", 3)
	appendN(t, store, "user:1", 1)
	appendN(t, store, "team:9", 2)

	got, err := store.Scopes(ctx, mempher.ScopeQuery{})
	if err != nil {
		t.Fatalf("Scopes: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("Scopes returned %d scopes, want 3", len(got))
	}

	// Id order, not insertion order: the cursor depends on it.
	want := []mempher.ScopeID{"team:9", "user:1", "user:2"}
	for i, scope := range got {
		if scope.ID != want[i] {
			t.Errorf("scope %d = %q, want %q", i, scope.ID, want[i])
		}
	}

	// last_seq is read from the catalogue rather than counted, and is the
	// number of episodes because Seq is dense from 1.
	seqs := map[mempher.ScopeID]int64{"team:9": 2, "user:1": 1, "user:2": 3}
	for _, scope := range got {
		if scope.LastSeq != seqs[scope.ID] {
			t.Errorf("%q LastSeq = %d, want %d", scope.ID, scope.LastSeq, seqs[scope.ID])
		}
		if scope.CreatedAt.IsZero() {
			t.Errorf("%q has no CreatedAt", scope.ID)
		}
	}
}

func TestScopesPagesAndFilters(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	for _, scope := range []mempher.ScopeID{"user:1", "user:2", "user:3", "team:1"} {
		appendN(t, store, scope, 1)
	}

	t.Run("pages by After", func(t *testing.T) {
		first, err := store.Scopes(ctx, mempher.ScopeQuery{Limit: 2})
		if err != nil {
			t.Fatalf("Scopes: %v", err)
		}
		if len(first) != 2 || first[0].ID != "team:1" || first[1].ID != "user:1" {
			t.Fatalf("first page = %v, want [team:1 user:1]", ids(first))
		}
		second, err := store.Scopes(ctx, mempher.ScopeQuery{After: first[1].ID, Limit: 2})
		if err != nil {
			t.Fatalf("Scopes: %v", err)
		}
		if len(second) != 2 || second[0].ID != "user:2" || second[1].ID != "user:3" {
			t.Fatalf("second page = %v, want [user:2 user:3]", ids(second))
		}
		last, err := store.Scopes(ctx, mempher.ScopeQuery{After: second[1].ID})
		if err != nil {
			t.Fatalf("Scopes: %v", err)
		}
		if len(last) != 0 {
			t.Errorf("page past the end = %v, want none", ids(last))
		}
	})

	t.Run("filters by prefix", func(t *testing.T) {
		got, err := store.Scopes(ctx, mempher.ScopeQuery{Prefix: "user:"})
		if err != nil {
			t.Fatalf("Scopes: %v", err)
		}
		if len(got) != 3 {
			t.Errorf("prefix user: returned %v, want three scopes", ids(got))
		}
	})
}

// TestScopesTreatsAPrefixAsTextNotAPattern is why starts_with is used rather
// than LIKE: a scope id is caller-supplied text, and % is legal in one.
func TestScopesTreatsAPrefixAsTextNotAPattern(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)

	appendN(t, store, "sale:50%off", 1)
	appendN(t, store, "sale:nothing", 1)

	got, err := store.Scopes(t.Context(), mempher.ScopeQuery{Prefix: "sale:50%"})
	if err != nil {
		t.Fatalf("Scopes: %v", err)
	}
	if len(got) != 1 || got[0].ID != "sale:50%off" {
		t.Errorf("prefix %q matched %v, want only the literal match", "sale:50%", ids(got))
	}
}

func TestScopesRejectsBadQueries(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	if _, err := store.Scopes(ctx, mempher.ScopeQuery{Limit: -1}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("negative limit err = %v, want mempher.ErrInvalidConfig", err)
	}
	long := mempher.ScopeID(strings.Repeat("x", 300))
	if _, err := store.Scopes(ctx, mempher.ScopeQuery{Prefix: long}); !errors.Is(err, mempher.ErrInvalidScope) {
		t.Errorf("over-long prefix err = %v, want mempher.ErrInvalidScope", err)
	}
	if _, err := store.Scopes(ctx, mempher.ScopeQuery{After: long}); !errors.Is(err, mempher.ErrInvalidScope) {
		t.Errorf("over-long cursor err = %v, want mempher.ErrInvalidScope", err)
	}
}

func ids(scopes []mempher.Scope) []mempher.ScopeID {
	out := make([]mempher.ScopeID, len(scopes))
	for i, scope := range scopes {
		out[i] = scope.ID
	}
	return out
}
