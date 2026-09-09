package postgres_test

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/internal/pgtest"
	"github.com/mempher/mempher/postgres"
)

const testModel = mempher.ModelID("fake@8")

// vec builds a distinct, valid vector of the schema's width.
func vec(seed float32) mempher.Vector {
	v := make(mempher.Vector, testDimensions)
	for i := range v {
		v[i] = seed + float32(i)/10
	}
	return v
}

func encoding(id mempher.EpisodeID, v mempher.Vector) mempher.Encoding {
	return mempher.Encoding{
		Episode:   id,
		Model:     testModel,
		Vector:    v,
		EncodedAt: epoch,
	}
}

func TestPutEncoding(t *testing.T) {
	t.Parallel()
	store, pool := migrated(t)
	ctx := t.Context()

	appended, err := store.Append(ctx, episode("user:1", "I like dark roast", epoch))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	id := appended.Episode.ID

	if err := store.PutEncoding(ctx, encoding(id, vec(0.5))); err != nil {
		t.Fatalf("PutEncoding: %v", err)
	}

	t.Run("scope and ingest time are copied from L0, not from the caller", func(t *testing.T) {
		var scope string
		var ingested time.Time
		if err := pool.QueryRow(ctx, `
			SELECT scope_id, ingested_at FROM mempher.episode_encodings
			WHERE episode_id = $1 AND model = $2
		`, id.String(), string(testModel)).Scan(&scope, &ingested); err != nil {
			t.Fatalf("read encoding: %v", err)
		}
		if scope != "user:1" {
			t.Errorf("scope_id = %q, want user:1", scope)
		}
		if !ingested.Equal(epoch) {
			t.Errorf("ingested_at = %s, want %s", ingested, epoch)
		}
	})

	t.Run("the vector round-trips exactly", func(t *testing.T) {
		// L2 distance of zero means every float32 came back unchanged, which
		// is what the shortest round-trip formatting is for.
		var distance float64
		if err := pool.QueryRow(ctx, `
			SELECT content_vec <-> $2::vector FROM mempher.episode_encodings
			WHERE episode_id = $1
		`, id.String(), vectorLiteral(vec(0.5))).Scan(&distance); err != nil {
			t.Fatalf("measure distance: %v", err)
		}
		if distance != 0 {
			t.Errorf("distance to the written vector = %v, want 0", distance)
		}
	})

	t.Run("writing again replaces, which is what makes retries safe", func(t *testing.T) {
		if err := store.PutEncoding(ctx, encoding(id, vec(9))); err != nil {
			t.Fatalf("second PutEncoding: %v", err)
		}
		var rows int
		if err := pool.QueryRow(ctx, `
			SELECT count(*) FROM mempher.episode_encodings WHERE episode_id = $1
		`, id.String()).Scan(&rows); err != nil {
			t.Fatalf("count: %v", err)
		}
		if rows != 1 {
			t.Errorf("%d encoding rows after two writes, want 1", rows)
		}
		var distance float64
		if err := pool.QueryRow(ctx, `
			SELECT content_vec <-> $2::vector FROM mempher.episode_encodings
			WHERE episode_id = $1
		`, id.String(), vectorLiteral(vec(9))).Scan(&distance); err != nil {
			t.Fatalf("measure distance: %v", err)
		}
		if distance != 0 {
			t.Error("the second write did not replace the first")
		}
	})

	t.Run("several models coexist over one episode", func(t *testing.T) {
		other := encoding(id, vec(1))
		other.Model = "other@8"
		if err := store.PutEncoding(ctx, other); err != nil {
			t.Fatalf("PutEncoding for a second model: %v", err)
		}
		var models int
		if err := pool.QueryRow(ctx, `
			SELECT count(*) FROM mempher.episode_encodings WHERE episode_id = $1
		`, id.String()).Scan(&models); err != nil {
			t.Fatalf("count: %v", err)
		}
		if models != 2 {
			t.Errorf("%d encodings, want 2", models)
		}
	})
}

func TestPutEncodingRejectsBadInput(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	appended, err := store.Append(ctx, episode("user:1", "content", epoch))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	id := appended.Episode.ID

	nan := vec(0)
	nan[2] = float32(math.NaN())
	inf := vec(0)
	inf[1] = float32(math.Inf(1))

	tests := []struct {
		name    string
		enc     mempher.Encoding
		wantErr error
	}{
		{
			name:    "unknown episode",
			enc:     encoding(mempher.EpisodeID{1, 2, 3}, vec(1)),
			wantErr: mempher.ErrNotFound,
		},
		{
			name:    "zero episode id",
			enc:     encoding(mempher.EpisodeID{}, vec(1)),
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "too few dimensions",
			enc:     encoding(id, vec(1)[:testDimensions-1]),
			wantErr: mempher.ErrDimensionMismatch,
		},
		{
			name:    "too many dimensions",
			enc:     encoding(id, append(vec(1), 1)),
			wantErr: mempher.ErrDimensionMismatch,
		},
		{
			name:    "a NaN from a broken embedder",
			enc:     encoding(id, nan),
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "an infinity from a broken embedder",
			enc:     encoding(id, inf),
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name: "surprise above one",
			enc: func() mempher.Encoding {
				e := encoding(id, vec(1))
				e.Surprise = 1.5
				return e
			}(),
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name: "no model",
			enc: func() mempher.Encoding {
				e := encoding(id, vec(1))
				e.Model = ""
				return e
			}(),
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name: "unresolved encode time",
			enc: func() mempher.Encoding {
				e := encoding(id, vec(1))
				e.EncodedAt = time.Time{}
				return e
			}(),
			wantErr: mempher.ErrInvalidConfig,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if err := store.PutEncoding(t.Context(), tc.enc); !errors.Is(err, tc.wantErr) {
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

// TestPutEncodingWithExtensionsElsewhere is why the vector type is qualified.
// The default search_path is "$user", public, so it does not include extras: a
// bare ::vector cast would fail here.
func TestPutEncodingWithExtensionsElsewhere(t *testing.T) {
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
	if _, err := postgres.Migrate(ctx, pool, testOptions()); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	store, err := postgres.New(ctx, pool)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := store.ExtensionSchema(); got != "extras" {
		t.Fatalf("ExtensionSchema() = %q, want extras", got)
	}

	appended, err := store.Append(ctx, episode("user:1", "content", epoch))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := store.PutEncoding(ctx, encoding(appended.Episode.ID, vec(1))); err != nil {
		t.Fatalf("PutEncoding with pgvector outside the search_path: %v", err)
	}
}

func TestPendingEncodings(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	var ids []mempher.EpisodeID
	for i := range 4 {
		got, err := store.Append(ctx, episode("user:1", fmt.Sprintf("episode %d", i), epoch))
		if err != nil {
			t.Fatalf("Append: %v", err)
		}
		ids = append(ids, got.Episode.ID)
	}
	otherScope, err := store.Append(ctx, episode("user:2", "elsewhere", epoch))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}

	t.Run("everything is pending before any worker runs", func(t *testing.T) {
		pending, err := store.PendingEncodings(ctx, mempher.PendingEncodings{Model: testModel})
		if err != nil {
			t.Fatalf("PendingEncodings: %v", err)
		}
		if len(pending) != 5 {
			t.Errorf("got %d pending, want 5", len(pending))
		}
	})

	t.Run("encoding one removes it", func(t *testing.T) {
		if err := store.PutEncoding(ctx, encoding(ids[0], vec(1))); err != nil {
			t.Fatalf("PutEncoding: %v", err)
		}
		pending, err := store.PendingEncodings(ctx, mempher.PendingEncodings{Model: testModel})
		if err != nil {
			t.Fatalf("PendingEncodings: %v", err)
		}
		if len(pending) != 4 {
			t.Errorf("got %d pending, want 4", len(pending))
		}
		for _, id := range pending {
			if id == ids[0] {
				t.Error("the encoded episode is still reported as pending")
			}
		}
	})

	t.Run("a different model sees everything again, which is how a backfill works",
		func(t *testing.T) {
			pending, err := store.PendingEncodings(ctx,
				mempher.PendingEncodings{Model: "a-new-model@8"})
			if err != nil {
				t.Fatalf("PendingEncodings: %v", err)
			}
			if len(pending) != 5 {
				t.Errorf("got %d pending for a new model, want all 5", len(pending))
			}
		})

	t.Run("one scope at a time", func(t *testing.T) {
		pending, err := store.PendingEncodings(ctx,
			mempher.PendingEncodings{Scope: "user:2", Model: testModel})
		if err != nil {
			t.Fatalf("PendingEncodings: %v", err)
		}
		if len(pending) != 1 || pending[0] != otherScope.Episode.ID {
			t.Errorf("scope user:2 pending = %v, want [%s]", pending, otherScope.Episode.ID)
		}
	})

	t.Run("paging within a scope", func(t *testing.T) {
		first, err := store.PendingEncodings(ctx, mempher.PendingEncodings{
			Scope: "user:1", Model: testModel, Limit: 2,
		})
		if err != nil {
			t.Fatalf("PendingEncodings: %v", err)
		}
		if len(first) != 2 {
			t.Fatalf("got %d, want 2", len(first))
		}
		// ids[0] is encoded, so the pending ones start at ids[1] (seq 2).
		if first[0] != ids[1] || first[1] != ids[2] {
			t.Errorf("first page = %v, want %v", first, ids[1:3])
		}
		second, err := store.PendingEncodings(ctx, mempher.PendingEncodings{
			Scope: "user:1", Model: testModel, AfterSeq: 3, Limit: 2,
		})
		if err != nil {
			t.Fatalf("PendingEncodings: %v", err)
		}
		if len(second) != 1 || second[0] != ids[3] {
			t.Errorf("second page = %v, want [%s]", second, ids[3])
		}
	})
}

func TestPendingEncodingsRejectsBadQueries(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)

	tests := []struct {
		name string
		q    mempher.PendingEncodings
	}{
		{name: "no model", q: mempher.PendingEncodings{}},
		{name: "negative afterSeq", q: mempher.PendingEncodings{Model: testModel, AfterSeq: -1}},
		{name: "afterSeq without a scope", q: mempher.PendingEncodings{Model: testModel, AfterSeq: 5}},
		{name: "negative limit", q: mempher.PendingEncodings{Model: testModel, Limit: -1}},
		{name: "invalid scope", q: mempher.PendingEncodings{Model: testModel, Scope: "bad\nscope"}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := store.PendingEncodings(t.Context(), tc.q)
			if err == nil {
				t.Fatal("expected an error")
			}
			if !errors.Is(err, mempher.ErrInvalidConfig) && !errors.Is(err, mempher.ErrInvalidScope) {
				t.Fatalf("err = %v, want an invalid-config or invalid-scope error", err)
			}
		})
	}
}

// vectorLiteral renders a vector the way the adapter does, for assertions that
// hand one back to PostgreSQL.
func vectorLiteral(v mempher.Vector) string {
	parts := make([]string, len(v))
	for i, f := range v {
		parts[i] = strconv.FormatFloat(float64(f), 'g', -1, 32)
	}
	return "[" + strings.Join(parts, ",") + "]"
}
