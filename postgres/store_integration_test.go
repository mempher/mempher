package postgres_test

import (
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mempher/mempher"
	"github.com/mempher/mempher/internal/pgtest"
	"github.com/mempher/mempher/memphertest"
	"github.com/mempher/mempher/postgres"
)

// A fixed instant, so every episode's timestamps are predictable.
var epoch = time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)

// migrated returns a Store over a freshly migrated database.
func migrated(t *testing.T) (*postgres.Store, *pgxpool.Pool) {
	t.Helper()
	pool := pgtest.Pool(t)
	if _, err := postgres.Migrate(t.Context(), pool, testOptions()); err != nil {
		t.Fatalf("Migrate: %v", err)
	}
	store, err := postgres.New(t.Context(), pool)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return store, pool
}

// episode builds a valid command with one encode job, as Memory.Append will.
func episode(scope mempher.ScopeID, content string, at time.Time) mempher.AppendCommand {
	return mempher.AppendCommand{
		Episodes: []mempher.NewEpisode{{
			Scope:      scope,
			Content:    content,
			Role:       mempher.RoleUser,
			Actor:      "user:1",
			Source:     "test",
			OccurredAt: at,
			IngestedAt: at,
			Binding:    mempher.Binding{"session": "s-42"},
		}},
		Jobs: []mempher.NewJob{{Kind: mempher.JobKindEncode, Scope: scope}},
	}
}

func TestNewStore(t *testing.T) {
	t.Parallel()

	t.Run("reads what the schema was migrated for", func(t *testing.T) {
		t.Parallel()
		store, _ := migrated(t)

		if got := store.Dimensions(); got != testDimensions {
			t.Errorf("Dimensions() = %d, want %d", got, testDimensions)
		}
		if got := store.TextSearchConfig(); got != "english" {
			t.Errorf("TextSearchConfig() = %q, want english", got)
		}
		if got := store.ExtensionSchema(); got != "public" {
			t.Errorf("ExtensionSchema() = %q, want public", got)
		}
	})

	t.Run("refuses an unmigrated database", func(t *testing.T) {
		t.Parallel()
		pool := pgtest.Pool(t)
		// The extensions exist but nothing else does.
		for _, ext := range []string{"vector", "btree_gin", "btree_gist"} {
			if _, err := pool.Exec(t.Context(), "CREATE EXTENSION "+ext); err != nil {
				t.Fatalf("create %s: %v", ext, err)
			}
		}
		if _, err := postgres.New(t.Context(), pool); !errors.Is(err, postgres.ErrSchemaNotReady) {
			t.Fatalf("err = %v, want ErrSchemaNotReady", err)
		}
	})

	t.Run("refuses a database without the extensions", func(t *testing.T) {
		t.Parallel()
		pool := pgtest.Pool(t)
		if _, err := postgres.New(t.Context(), pool); !errors.Is(err, postgres.ErrMissingExtension) {
			t.Fatalf("err = %v, want ErrMissingExtension", err)
		}
	})

	t.Run("refuses a nil pool", func(t *testing.T) {
		t.Parallel()
		if _, err := postgres.New(t.Context(), nil); !errors.Is(err, mempher.ErrInvalidConfig) {
			t.Fatalf("err = %v, want mempher.ErrInvalidConfig", err)
		}
	})
}

func TestVerifyEmbedder(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)

	if err := store.VerifyEmbedder(memphertest.NewEmbedder(testDimensions)); err != nil {
		t.Errorf("a matching embedder should verify: %v", err)
	}
	err := store.VerifyEmbedder(memphertest.NewEmbedder(testDimensions * 2))
	if !errors.Is(err, mempher.ErrEmbedderMismatch) {
		t.Errorf("err = %v, want mempher.ErrEmbedderMismatch", err)
	}
	if !errors.Is(store.VerifyEmbedder(nil), mempher.ErrInvalidConfig) {
		t.Error("a nil embedder should be rejected")
	}
}

func TestAppend(t *testing.T) {
	t.Parallel()
	store, pool := migrated(t)
	ctx := t.Context()

	got, err := store.Append(ctx, episode("user:1", "I am allergic to hazelnuts.", epoch))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}

	t.Run("the episode comes back with what the database assigned", func(t *testing.T) {
		switch {
		case got.Episodes[0].ID.IsZero():
			t.Error("no episode id was returned")
		case got.Episodes[0].Seq != 1:
			t.Errorf("Seq = %d, want 1", got.Episodes[0].Seq)
		case got.Episodes[0].Content != "I am allergic to hazelnuts.":
			t.Errorf("Content = %q", got.Episodes[0].Content)
		case got.Episodes[0].Role != mempher.RoleUser:
			t.Errorf("Role = %q", got.Episodes[0].Role)
		case !got.Episodes[0].IngestedAt.Equal(epoch):
			t.Errorf("IngestedAt = %s, want %s", got.Episodes[0].IngestedAt, epoch)
		}
	})

	t.Run("the encode job was enqueued in the same statement", func(t *testing.T) {
		if len(got.Jobs) != 1 {
			t.Fatalf("got %d jobs, want 1", len(got.Jobs))
		}
		job := got.Jobs[0]
		switch {
		case job.ID.IsZero():
			t.Error("no job id was returned")
		case job.Kind != mempher.JobKindEncode:
			t.Errorf("Kind = %q, want %q", job.Kind, mempher.JobKindEncode)
		case job.State != mempher.JobStatePending:
			t.Errorf("State = %q, want %q", job.State, mempher.JobStatePending)
		case job.Attempts != 0:
			t.Errorf("Attempts = %d, want 0", job.Attempts)
		case job.MaxAttempts != mempher.DefaultMaxAttempts:
			t.Errorf("MaxAttempts = %d, want %d", job.MaxAttempts, mempher.DefaultMaxAttempts)
		case len(job.Episodes) != 1 || job.Episodes[0] != got.Episodes[0].ID:
			t.Errorf("job names %v, want the appended episode", job.Episodes)
		case !job.RunAfter.Equal(epoch):
			t.Errorf("RunAfter = %s, want %s", job.RunAfter, epoch)
		}
	})

	t.Run("reading it back matches what Append reported", func(t *testing.T) {
		read, err := store.Episode(ctx, "user:1", got.Episodes[0].ID)
		if err != nil {
			t.Fatalf("Episode: %v", err)
		}
		if read.ID != got.Episodes[0].ID || read.Seq != got.Episodes[0].Seq ||
			read.Content != got.Episodes[0].Content || read.Role != got.Episodes[0].Role ||
			read.Actor != got.Episodes[0].Actor || read.Source != got.Episodes[0].Source {
			t.Errorf("read back %+v, want %+v", read, got.Episodes[0])
		}
		if !read.OccurredAt.Equal(epoch) || !read.IngestedAt.Equal(epoch) {
			t.Errorf("timestamps = %s / %s, want %s", read.OccurredAt, read.IngestedAt, epoch)
		}
		if read.Binding["session"] != "s-42" || len(read.Binding) != 1 {
			t.Errorf("Binding = %v, want {session: s-42}", read.Binding)
		}
	})

	t.Run("the id is a uuidv7 carrying its arrival time", func(t *testing.T) {
		// The id's instant is physical arrival time from the server, which is
		// deliberately not the injected clock's epoch.
		var extracted time.Time
		if err := pool.QueryRow(ctx,
			"SELECT uuid_extract_timestamp($1::uuid)", got.Episodes[0].ID.String()).
			Scan(&extracted); err != nil {
			t.Fatalf("uuid_extract_timestamp: %v", err)
		}
		if extracted.IsZero() {
			t.Error("the id carries no timestamp, so it is not a uuidv7")
		}
		if extracted.Equal(epoch) {
			t.Error("the id's instant equals the test clock, so it is not server time")
		}
	})
}

// TestAppendAllocatesDenseSequences is the property consolidation replays on.
func TestAppendAllocatesDenseSequences(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	const n = 5
	ids := make([]mempher.EpisodeID, 0, n)
	for i := range n {
		at := epoch.Add(time.Duration(i) * time.Minute)
		got, err := store.Append(ctx, episode("user:1", fmt.Sprintf("episode %d", i), at))
		if err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
		if want := int64(i + 1); got.Episodes[0].Seq != want {
			t.Errorf("episode %d has Seq %d, want %d", i, got.Episodes[0].Seq, want)
		}
		ids = append(ids, got.Episodes[0].ID)
	}

	// A second scope starts again at 1: sequences are per scope.
	other, err := store.Append(ctx, episode("user:2", "different tenant", epoch))
	if err != nil {
		t.Fatalf("Append to second scope: %v", err)
	}
	if other.Episodes[0].Seq != 1 {
		t.Errorf("second scope starts at Seq %d, want 1", other.Episodes[0].Seq)
	}

	replayed, err := store.Replay(ctx, "user:1", 0, 100)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if len(replayed) != n {
		t.Fatalf("replayed %d episodes, want %d", len(replayed), n)
	}
	for i, ep := range replayed {
		if ep.ID != ids[i] {
			t.Errorf("replay position %d is %s, want %s", i, ep.ID, ids[i])
		}
		if ep.Seq != int64(i+1) {
			t.Errorf("replay position %d has Seq %d, want %d", i, ep.Seq, i+1)
		}
	}
}

func TestAppendIsScopeIsolated(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	mine, err := store.Append(ctx, episode("user:1", "my secret", epoch))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	if _, err := store.Append(ctx, episode("user:2", "their secret", epoch)); err != nil {
		t.Fatalf("Append: %v", err)
	}

	// Asking for a real episode from the wrong scope is absence, not a
	// permission error: scope isolation must not be probeable.
	_, err = store.Episode(ctx, "user:2", mine.Episodes[0].ID)
	if !errors.Is(err, mempher.ErrNotFound) {
		t.Errorf("cross-scope read err = %v, want mempher.ErrNotFound", err)
	}

	theirs, err := store.Replay(ctx, "user:2", 0, 100)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if len(theirs) != 1 || theirs[0].Content != "their secret" {
		t.Errorf("scope user:2 sees %d episodes: %+v", len(theirs), theirs)
	}
}

func TestAppendRejectsBadCommands(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)

	tests := []struct {
		name    string
		mutate  func(*mempher.AppendCommand)
		wantErr error
	}{
		{
			name:    "no scope",
			mutate:  func(c *mempher.AppendCommand) { c.Episodes[0].Scope = "" },
			wantErr: mempher.ErrInvalidScope,
		},
		{
			name:    "no content",
			mutate:  func(c *mempher.AppendCommand) { c.Episodes[0].Content = "" },
			wantErr: mempher.ErrInvalidContent,
		},
		{
			name:    "no role",
			mutate:  func(c *mempher.AppendCommand) { c.Episodes[0].Role = "" },
			wantErr: mempher.ErrInvalidRole,
		},
		{
			name:    "no source",
			mutate:  func(c *mempher.AppendCommand) { c.Episodes[0].Source = "" },
			wantErr: mempher.ErrInvalidSource,
		},
		{
			name:    "unresolved ingest time",
			mutate:  func(c *mempher.AppendCommand) { c.Episodes[0].IngestedAt = time.Time{} },
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "unresolved event time",
			mutate:  func(c *mempher.AppendCommand) { c.Episodes[0].OccurredAt = time.Time{} },
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name: "unknown job kind",
			mutate: func(c *mempher.AppendCommand) {
				c.Jobs = []mempher.NewJob{{Kind: mempher.JobKind("nonesuch")}}
			},
			wantErr: mempher.ErrInvalidJobKind,
		},
		{
			name: "a job that already names episodes",
			mutate: func(c *mempher.AppendCommand) {
				c.Jobs[0].Episodes = []mempher.EpisodeID{{}}
			},
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name: "two jobs of one kind",
			mutate: func(c *mempher.AppendCommand) {
				c.Jobs = append(c.Jobs, mempher.NewJob{Kind: mempher.JobKindEncode})
			},
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name: "over-long content",
			mutate: func(c *mempher.AppendCommand) {
				c.Episodes[0].Content = strings.Repeat("x", mempher.MaxContentLen+1)
			},
			wantErr: mempher.ErrInvalidContent,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			cmd := episode("user:1", "content", epoch)
			tc.mutate(&cmd)
			if _, err := store.Append(t.Context(), cmd); !errors.Is(err, tc.wantErr) {
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

// TestAppendWithoutJobs checks the LEFT JOIN still yields the episode row.
func TestAppendWithoutJobs(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)

	cmd := episode("user:1", "no work implied", epoch)
	cmd.Jobs = nil

	got, err := store.Append(t.Context(), cmd)
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	if got.Episodes[0].ID.IsZero() || got.Episodes[0].Seq != 1 {
		t.Errorf("episode = %+v, want an id and Seq 1", got.Episodes[0])
	}
	if len(got.Jobs) != 0 {
		t.Errorf("got %d jobs, want none", len(got.Jobs))
	}
}

// TestAppendWithNilBinding covers the NOT NULL jsonb column: a nil map marshals
// to null, which the CHECK rejects.
func TestAppendWithNilBinding(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)

	cmd := episode("user:1", "no binding", epoch)
	cmd.Episodes[0].Binding = nil

	got, err := store.Append(t.Context(), cmd)
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	read, err := store.Episode(t.Context(), "user:1", got.Episodes[0].ID)
	if err != nil {
		t.Fatalf("Episode: %v", err)
	}
	if len(read.Binding) != 0 {
		t.Errorf("Binding = %v, want empty", read.Binding)
	}
}

func TestEpisodeAndReplayRejectBadArguments(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	if _, err := store.Episode(ctx, "", mempher.EpisodeID{}); !errors.Is(err, mempher.ErrInvalidScope) {
		t.Errorf("empty scope err = %v, want mempher.ErrInvalidScope", err)
	}
	if _, err := store.Episode(ctx, "user:1", mempher.EpisodeID{}); !errors.Is(err, mempher.ErrNotFound) {
		t.Errorf("zero id err = %v, want mempher.ErrNotFound", err)
	}
	if _, err := store.Replay(ctx, "user:1", -1, 10); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("negative afterSeq err = %v, want mempher.ErrInvalidConfig", err)
	}
	if _, err := store.Replay(ctx, "user:1", 0, 0); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("zero limit err = %v, want mempher.ErrInvalidConfig", err)
	}
}

func TestReplayPagesInOrder(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	for i := range 5 {
		if _, err := store.Append(ctx,
			episode("user:1", fmt.Sprintf("episode %d", i), epoch)); err != nil {
			t.Fatalf("Append %d: %v", i, err)
		}
	}

	var seen []int64
	after := int64(0)
	for range 10 {
		page, err := store.Replay(ctx, "user:1", after, 2)
		if err != nil {
			t.Fatalf("Replay: %v", err)
		}
		if len(page) == 0 {
			break
		}
		for _, ep := range page {
			seen = append(seen, ep.Seq)
		}
		after = page[len(page)-1].Seq
	}
	if want := []int64{1, 2, 3, 4, 5}; fmt.Sprint(seen) != fmt.Sprint(want) {
		t.Errorf("paged sequences = %v, want %v", seen, want)
	}
}
