package mempher

import (
	"context"
	"errors"
	"testing"
	"time"
)

// storeOnly satisfies [Store] and nothing else, so a test can prove that New
// insists on a Queue when the store cannot serve as one.
type storeOnly struct{}

func (storeOnly) Append(context.Context, AppendCommand) (AppendResult, error) {
	return AppendResult{}, errNotImplemented
}
func (storeOnly) Episode(context.Context, ScopeID, EpisodeID) (Episode, error) {
	return Episode{}, errNotImplemented
}
func (storeOnly) Replay(context.Context, ScopeID, int64, int) ([]Episode, error) {
	return nil, errNotImplemented
}
func (storeOnly) SearchSemantic(context.Context, SemanticQuery) ([]Candidate, error) {
	return nil, errNotImplemented
}
func (storeOnly) SearchLexical(context.Context, LexicalQuery) ([]Candidate, error) {
	return nil, errNotImplemented
}
func (storeOnly) PutEncoding(context.Context, Encoding) error { return errNotImplemented }
func (storeOnly) PendingEncodings(context.Context, PendingEncodings) ([]EpisodeID, error) {
	return nil, errNotImplemented
}

// storeAndQueue is shaped like the real postgres adapter: one type serving both
// ports off one pool.
type storeAndQueue struct{ storeOnly }

func (storeAndQueue) Enqueue(context.Context, NewJob) (Job, error) { return Job{}, errNotImplemented }
func (storeAndQueue) Lease(context.Context, LeaseRequest) ([]Job, error) {
	return nil, errNotImplemented
}
func (storeAndQueue) Succeed(context.Context, JobID, WorkerID, time.Time) error {
	return errNotImplemented
}
func (storeAndQueue) Fail(context.Context, JobID, WorkerID, error, time.Time) error {
	return errNotImplemented
}
func (storeAndQueue) Reclaim(context.Context, time.Time) (int, error) { return 0, errNotImplemented }
func (storeAndQueue) Job(context.Context, JobID) (Job, error)         { return Job{}, errNotImplemented }

var (
	_ Store = storeOnly{}
	_ Store = storeAndQueue{}
	_ Queue = storeAndQueue{}
)

// stubEmbedder is enough to satisfy New; behaviour is tested in memphertest.
type stubEmbedder struct {
	dims  int
	model ModelID
}

func (e stubEmbedder) EmbedDocuments(context.Context, []string) ([]Vector, error) {
	return nil, errNotImplemented
}
func (e stubEmbedder) EmbedQuery(context.Context, string) (Vector, error) {
	return nil, errNotImplemented
}
func (e stubEmbedder) Dimensions() int { return e.dims }
func (e stubEmbedder) Model() ModelID  { return e.model }

func goodEmbedder() Embedder { return stubEmbedder{dims: 8, model: "stub@8"} }

func TestNew(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		cfg     Config
		wantErr error
	}{
		{
			name: "a store that is also a queue needs no Queue",
			cfg:  Config{Store: storeAndQueue{}, Embedder: goodEmbedder()},
		},
		{
			name: "an explicit queue is accepted",
			cfg:  Config{Store: storeOnly{}, Queue: storeAndQueue{}, Embedder: goodEmbedder()},
		},
		{
			name:    "a store that is not a queue needs one",
			cfg:     Config{Store: storeOnly{}, Embedder: goodEmbedder()},
			wantErr: ErrInvalidConfig,
		},
		{
			name:    "no store",
			cfg:     Config{Embedder: goodEmbedder()},
			wantErr: ErrInvalidConfig,
		},
		{
			name:    "no embedder",
			cfg:     Config{Store: storeAndQueue{}},
			wantErr: ErrInvalidConfig,
		},
		{
			name:    "embedder with no dimensions",
			cfg:     Config{Store: storeAndQueue{}, Embedder: stubEmbedder{dims: 0, model: "stub"}},
			wantErr: ErrInvalidConfig,
		},
		{
			name:    "embedder with no model id",
			cfg:     Config{Store: storeAndQueue{}, Embedder: stubEmbedder{dims: 8}},
			wantErr: ErrInvalidConfig,
		},
		{
			name: "negative default limit",
			cfg: Config{
				Store: storeAndQueue{}, Embedder: goodEmbedder(), DefaultLimit: -1,
			},
			wantErr: ErrInvalidConfig,
		},
		{
			name: "invalid fusion options surface from New",
			cfg: Config{
				Store: storeAndQueue{}, Embedder: goodEmbedder(),
				Fusion: FusionOptions{K: -1},
			},
			wantErr: ErrInvalidConfig,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m, err := New(tc.cfg)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				if m != nil {
					t.Error("a failed New should return a nil Memory")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if m.queue == nil {
				t.Error("queue was not resolved")
			}
			if m.clock == nil || m.tokens == nil {
				t.Error("clock and token counter should have defaults")
			}
			if m.defaultLimit != DefaultRecallLimit {
				t.Errorf("defaultLimit = %d, want %d", m.defaultLimit, DefaultRecallLimit)
			}
			if m.fusion.K != DefaultFusionK {
				t.Errorf("fusion K = %v, want %v", m.fusion.K, DefaultFusionK)
			}
		})
	}
}

// TestNewQueueFallbackPrefersExplicit pins which queue wins when both are
// available, because silently ignoring an explicit port would be the worse bug.
func TestNewQueueFallbackPrefersExplicit(t *testing.T) {
	t.Parallel()

	explicit := storeAndQueue{}
	m, err := New(Config{Store: storeAndQueue{}, Queue: explicit, Embedder: goodEmbedder()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, ok := m.queue.(storeAndQueue); !ok {
		t.Fatalf("queue is %T, want the explicitly configured one", m.queue)
	}
}

func TestNewWorker(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		cfg     WorkerConfig
		wantErr error
	}{
		{
			name: "a store that is also a queue needs no Queue",
			cfg:  WorkerConfig{Store: storeAndQueue{}, Embedder: goodEmbedder()},
		},
		{
			name:    "a store that is not a queue needs one",
			cfg:     WorkerConfig{Store: storeOnly{}, Embedder: goodEmbedder()},
			wantErr: ErrInvalidConfig,
		},
		{name: "no store", cfg: WorkerConfig{Embedder: goodEmbedder()}, wantErr: ErrInvalidConfig},
		{name: "no embedder", cfg: WorkerConfig{Store: storeAndQueue{}}, wantErr: ErrInvalidConfig},
		{
			name:    "negative batch",
			cfg:     WorkerConfig{Store: storeAndQueue{}, Embedder: goodEmbedder(), Batch: -1},
			wantErr: ErrInvalidConfig,
		},
		{
			name: "negative lease",
			cfg: WorkerConfig{
				Store: storeAndQueue{}, Embedder: goodEmbedder(), LeaseDuration: -time.Second,
			},
			wantErr: ErrInvalidConfig,
		},
		{
			name: "negative poll interval",
			cfg: WorkerConfig{
				Store: storeAndQueue{}, Embedder: goodEmbedder(), PollInterval: -time.Second,
			},
			wantErr: ErrInvalidConfig,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			w, err := NewWorker(tc.cfg)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if w.ID() == "" {
				t.Error("a worker should derive an id when none is configured")
			}
			if w.batch != DefaultWorkerBatch || w.lease != DefaultLeaseDuration ||
				w.poll != DefaultPollInterval {
				t.Errorf("defaults not applied: batch=%d lease=%s poll=%s",
					w.batch, w.lease, w.poll)
			}
		})
	}
}

// TestNewWorkerKeepsConfiguredID checks the identity a stuck lease will show.
func TestNewWorkerKeepsConfiguredID(t *testing.T) {
	t.Parallel()

	w, err := NewWorker(WorkerConfig{
		Store: storeAndQueue{}, Embedder: goodEmbedder(), ID: "encoder-3",
	})
	if err != nil {
		t.Fatalf("NewWorker: %v", err)
	}
	if got := w.ID(); got != "encoder-3" {
		t.Errorf("ID() = %q, want %q", got, "encoder-3")
	}
}
