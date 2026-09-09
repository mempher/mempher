// The consolidation loop.

package mempher

import (
	"context"
	"fmt"
	"os"
	"time"
)

// Worker defaults, used when the corresponding [WorkerConfig] field is zero.
const (
	// DefaultWorkerBatch is how many jobs one lease claims.
	DefaultWorkerBatch = 8
	// DefaultPollInterval is how long a worker waits after finding no work.
	DefaultPollInterval = time.Second
)

// WorkerConfig assembles a [Worker]. Store and Embedder are required;
// everything else has a default.
type WorkerConfig struct {
	// Store is where projections are written. Required.
	Store Store
	// Queue is where work is claimed. Optional: when nil, and Store also
	// implements [Queue], the store is used for both.
	Queue Queue
	// Embedder produces the vectors. Required.
	Embedder Embedder
	// Clock is the source of time. Nil means [SystemClock].
	Clock Clock
	// ID identifies this worker in job leases. Empty means a value derived
	// from the hostname and process id.
	ID WorkerID
	// Batch is how many jobs to claim at a time. Zero means
	// [DefaultWorkerBatch].
	Batch int
	// LeaseDuration is how long a claim holds, and must exceed how long a
	// batch takes: an expired lease is reclaimed and the work runs twice.
	// Zero means [DefaultLeaseDuration].
	LeaseDuration time.Duration
	// PollInterval is how long [Worker.Run] waits after finding no work.
	// Zero means [DefaultPollInterval].
	PollInterval time.Duration
}

// Worker drains the job queue: the consolidation loop, which in this stage only
// encodes episodes into vectors. All model spend lives here, off the write path.
//
// Every job body must be idempotent, because leasing is at-least-once and a
// reclaimed job runs again. Encoding is idempotent by construction: it derives
// from an immutable episode and writes through an upsert.
type Worker struct {
	store    Store
	queue    Queue
	embedder Embedder
	clock    Clock
	id       WorkerID
	batch    int
	lease    time.Duration
	poll     time.Duration
}

// NewWorker assembles a [Worker] from cfg, resolving defaults and rejecting a
// configuration that cannot work. It performs no I/O.
func NewWorker(cfg WorkerConfig) (*Worker, error) {
	if cfg.Store == nil {
		return nil, fmt.Errorf("mempher: new worker: Store is required: %w", ErrInvalidConfig)
	}
	if cfg.Embedder == nil {
		return nil, fmt.Errorf("mempher: new worker: Embedder is required: %w", ErrInvalidConfig)
	}
	if dims := cfg.Embedder.Dimensions(); dims <= 0 {
		return nil, fmt.Errorf("mempher: new worker: embedder reports %d dimensions: %w",
			dims, ErrInvalidConfig)
	}
	if cfg.Embedder.Model() == "" {
		return nil, fmt.Errorf("mempher: new worker: embedder reports no model id: %w", ErrInvalidConfig)
	}

	queue := cfg.Queue
	if queue == nil {
		asQueue, ok := cfg.Store.(Queue)
		if !ok {
			return nil, fmt.Errorf(
				"mempher: new worker: Queue is required because %T does not implement mempher.Queue: %w",
				cfg.Store, ErrInvalidConfig)
		}
		queue = asQueue
	}

	switch {
	case cfg.Batch < 0:
		return nil, fmt.Errorf("mempher: new worker: Batch is %d: %w", cfg.Batch, ErrInvalidConfig)
	case cfg.LeaseDuration < 0:
		return nil, fmt.Errorf("mempher: new worker: LeaseDuration is %s: %w",
			cfg.LeaseDuration, ErrInvalidConfig)
	case cfg.PollInterval < 0:
		return nil, fmt.Errorf("mempher: new worker: PollInterval is %s: %w",
			cfg.PollInterval, ErrInvalidConfig)
	}

	w := &Worker{
		store:    cfg.Store,
		queue:    queue,
		embedder: cfg.Embedder,
		clock:    cfg.Clock,
		id:       cfg.ID,
		batch:    cfg.Batch,
		lease:    cfg.LeaseDuration,
		poll:     cfg.PollInterval,
	}
	if w.clock == nil {
		w.clock = SystemClock{}
	}
	if w.id == "" {
		w.id = defaultWorkerID()
	}
	if w.batch == 0 {
		w.batch = DefaultWorkerBatch
	}
	if w.lease == 0 {
		w.lease = DefaultLeaseDuration
	}
	if w.poll == 0 {
		w.poll = DefaultPollInterval
	}
	return w, nil
}

// ID returns the identity this worker records in job leases, which is what to
// look for in the queue when a job is stuck.
func (w *Worker) ID() WorkerID { return w.id }

// defaultWorkerID names a worker after its machine and process, so a held lease
// points at something you can go and look at.
func defaultWorkerID() WorkerID {
	host, err := os.Hostname()
	if err != nil || host == "" {
		host = "unknown"
	}
	return WorkerID(fmt.Sprintf("%s/%d", host, os.Getpid()))
}

// Run drains the queue until ctx is cancelled, waiting PollInterval whenever it
// finds no work, and returns ctx.Err() on cancellation. A transient queue
// failure does not stop the loop.
func (w *Worker) Run(ctx context.Context) error {
	return errNotImplemented
}

// DrainOnce claims and runs at most one batch, and reports how many succeeded.
// It is what a test calls to make deferred work happen at a known point, and
// what a one-shot command calls instead of a loop.
func (w *Worker) DrainOnce(ctx context.Context) (int, error) {
	return 0, errNotImplemented
}
