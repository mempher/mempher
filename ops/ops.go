// Package ops is the operational half of mempher: recovering the work L0
// implies, reading and pruning the job queue, and walking the catalogue of
// scopes.
//
// None of it is on the write or the read path. It is a separate package so that
// a reader meeting mempher for the first time meets [mempher.Memory.Append],
// [mempher.Memory.Recall] and [mempher.Memory.Forget] without also meeting the
// machinery for running them in production -- and so that the machinery is
// exactly one import away when a queue stalls at three in the morning.
//
// One backend normally serves all of it. The postgres subpackage satisfies both
// of the interfaces here and every port in the root package, off one pool.
package ops

import (
	"context"
	"fmt"
	"time"

	"github.com/mempher/mempher"
)

// Store is what operating a mempher deployment needs of its backend: the
// catalogue, the anti-join that says what work is outstanding, and the queue as
// something to read rather than only to drain.
//
// It is one interface rather than four because a deployment that splits these
// across different backends is not a deployment this package can help with
// anyway: a backfill has to enqueue against the same queue a worker leases from.
type Store interface {
	// Scopes lists the partitions of L0 in id order.
	Scopes(ctx context.Context, q ScopeQuery) ([]Scope, error)

	// Episode reads one episode, which is how a backfill places its cursor.
	Episode(ctx context.Context, scope mempher.ScopeID, id mempher.EpisodeID) (mempher.Episode, error)

	// PendingEncodings returns ids of episodes with no encoding for a model.
	PendingEncodings(ctx context.Context, q PendingEncodings) ([]mempher.EpisodeID, error)

	// Enqueue adds one job, or returns the outstanding one it would have
	// duplicated.
	Enqueue(ctx context.Context, job mempher.NewJob, now time.Time) (mempher.Job, error)

	// Jobs lists jobs matching q, oldest first.
	Jobs(ctx context.Context, q JobQuery) ([]mempher.Job, error)

	// Stats counts the queue by kind and state.
	Stats(ctx context.Context, scope mempher.ScopeID) ([]JobCount, error)

	// Retry returns one dead job to pending with its attempts reset.
	Retry(ctx context.Context, id mempher.JobID, now time.Time) (mempher.Job, error)

	// Purge deletes finished jobs and reports how many rows went.
	Purge(ctx context.Context, req PurgeRequest) (int, error)
}

// FactStore is the L1 half, needed only to backfill extractions.
type FactStore interface {
	// PendingExtractions returns ids of episodes no extraction marker covers.
	PendingExtractions(ctx context.Context, q PendingExtractions) ([]mempher.EpisodeID, error)
}

// Config assembles an [Ops]. Only Store is required.
//
// The model-calling ports are optional because most of what this package does
// needs neither. Listing a stalled queue at three in the morning should not
// require constructing an embedder, so it does not: they are required by
// [Ops.Backfill] alone, and only for the kind of work each one names.
type Config struct {
	// Store is the backend. Required.
	Store Store
	// FactStore persists L1. Optional: when nil, and Store also implements
	// [FactStore], the store is used for both. Required to backfill
	// extractions.
	FactStore FactStore
	// Embedder names the vector space to backfill encodings for. Optional,
	// and required for exactly that: without it [Ops.Backfill] cannot say
	// which encodings are missing, because "missing" is per model.
	Embedder mempher.Embedder
	// Extractor names whose extractions to backfill. Optional, on the same
	// terms as Embedder.
	Extractor mempher.Extractor
	// Clock is the source of time. Nil means [mempher.SystemClock].
	Clock mempher.Clock
}

// Ops operates one mempher deployment. It is safe for concurrent use and holds
// no mutable state.
type Ops struct {
	store     Store
	facts     FactStore
	embedder  mempher.Embedder
	extractor mempher.Extractor
	clock     mempher.Clock

	// kinds is what this Ops can enqueue, fixed at construction by which
	// model-calling ports it was given.
	kinds []mempher.JobKind
}

// New assembles an [Ops] from cfg. It performs no I/O.
func New(cfg Config) (*Ops, error) {
	if cfg.Store == nil {
		return nil, fmt.Errorf("mempher/ops: new: Store is required: %w", mempher.ErrInvalidConfig)
	}

	facts := cfg.FactStore
	if facts == nil {
		if asFactStore, ok := cfg.Store.(FactStore); ok {
			facts = asFactStore
		}
	}

	o := &Ops{
		store:     cfg.Store,
		facts:     facts,
		embedder:  cfg.Embedder,
		extractor: cfg.Extractor,
		clock:     cfg.Clock,
	}
	if cfg.Embedder != nil {
		if cfg.Embedder.Model() == "" {
			return nil, fmt.Errorf("mempher/ops: new: embedder reports no model id: %w",
				mempher.ErrInvalidConfig)
		}
		o.kinds = append(o.kinds, mempher.JobKindEncode)
	}
	if cfg.Extractor != nil {
		if cfg.Extractor.Model() == "" {
			return nil, fmt.Errorf("mempher/ops: new: extractor reports no model id: %w",
				mempher.ErrInvalidConfig)
		}
		if facts == nil {
			return nil, fmt.Errorf(
				"mempher/ops: new: FactStore is required because %T does not implement "+
					"ops.FactStore: %w", cfg.Store, mempher.ErrInvalidConfig)
		}
		o.kinds = append(o.kinds, mempher.JobKindExtract)
	}
	if o.clock == nil {
		o.clock = mempher.SystemClock{}
	}
	return o, nil
}

// Scopes lists the partitions of L0 in id order.
//
// It is the catalogue everything else assumes: Seq is only ordered within a
// scope, so anything spanning the whole log walks it one scope at a time.
func (o *Ops) Scopes(ctx context.Context, q ScopeQuery) ([]Scope, error) {
	return o.store.Scopes(ctx, q)
}

// Jobs lists jobs matching q, oldest first. Listing [mempher.JobStateDead] is
// what this exists for: a queue whose failures cannot be found is a queue that
// fails silently.
func (o *Ops) Jobs(ctx context.Context, q JobQuery) ([]mempher.Job, error) {
	return o.store.Jobs(ctx, q)
}

// Stats counts the queue by kind and state, and reports the age of the oldest
// job in each bucket. Depth alone does not say whether a queue is working: a
// thousand pending jobs is healthy if the oldest arrived a second ago.
func (o *Ops) Stats(ctx context.Context, scope mempher.ScopeID) ([]JobCount, error) {
	return o.store.Stats(ctx, scope)
}

// Retry returns one dead job to pending with its attempts reset, and is what to
// call once the cause in [mempher.Job.LastError] has been dealt with.
func (o *Ops) Retry(ctx context.Context, id mempher.JobID) (mempher.Job, error) {
	return o.store.Retry(ctx, id, o.clock.Now())
}

// Purge deletes finished jobs matching req and reports how many rows went. It
// never deletes work that is still owed.
func (o *Ops) Purge(ctx context.Context, req PurgeRequest) (int, error) {
	return o.store.Purge(ctx, req)
}
