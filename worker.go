// The consolidation loop.

package mempher

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"time"
)

// Worker defaults, used when the corresponding [WorkerConfig] field is zero.
const (
	// DefaultWorkerBatch is how many jobs one lease claims.
	DefaultWorkerBatch = 8
	// DefaultPollInterval is how long a worker waits after finding no work.
	DefaultPollInterval = time.Second
	// DefaultReclaimInterval is how often a worker releases the jobs whose
	// leases have expired.
	DefaultReclaimInterval = time.Minute
	// DefaultRetryBackoff is how long a job waits before its second attempt.
	// Each further attempt doubles it, up to [MaxRetryBackoff].
	DefaultRetryBackoff = 10 * time.Second
	// MaxRetryBackoff caps the wait between attempts.
	MaxRetryBackoff = 10 * time.Minute
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
	// ReclaimInterval is how often [Worker.Run] releases jobs whose leases
	// have expired. Zero means [DefaultReclaimInterval].
	ReclaimInterval time.Duration
	// Logger receives failures that [Worker.Run] absorbs to stay alive. Nil
	// means [slog.Default]: a worker that swallowed them silently would be
	// worse than one that is slightly noisy.
	Logger *slog.Logger
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
	log      *slog.Logger
	id       WorkerID
	batch    int
	lease    time.Duration
	poll     time.Duration
	reclaim  time.Duration
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
	case cfg.ReclaimInterval < 0:
		return nil, fmt.Errorf("mempher: new worker: ReclaimInterval is %s: %w",
			cfg.ReclaimInterval, ErrInvalidConfig)
	}

	w := &Worker{
		store:    cfg.Store,
		queue:    queue,
		embedder: cfg.Embedder,
		clock:    cfg.Clock,
		log:      cfg.Logger,
		id:       cfg.ID,
		batch:    cfg.Batch,
		lease:    cfg.LeaseDuration,
		poll:     cfg.PollInterval,
		reclaim:  cfg.ReclaimInterval,
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
	if w.reclaim == 0 {
		w.reclaim = DefaultReclaimInterval
	}
	if w.log == nil {
		w.log = slog.Default()
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
	timer := time.NewTimer(0)
	defer timer.Stop()

	var lastReclaim time.Time
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("mempher: worker %s: %w", w.id, ctx.Err())
		case <-timer.C:
		}

		if now := w.clock.Now(); now.Sub(lastReclaim) >= w.reclaim {
			lastReclaim = now
			if released, err := w.queue.Reclaim(ctx, now); err != nil {
				w.logFailure(ctx, "reclaiming expired leases", err)
			} else if released > 0 {
				w.log.InfoContext(ctx, "mempher: released expired job leases",
					"worker", string(w.id), "jobs", released)
			}
		}

		done, err := w.DrainOnce(ctx)
		if err != nil {
			w.logFailure(ctx, "draining the job queue", err)
		}

		// Work usually arrives in bursts, so a batch that found something is
		// followed straight away by another attempt.
		wait := w.poll
		if done > 0 {
			wait = 0
		}
		timer.Reset(wait)
	}
}

// logFailure reports what Run absorbed, unless the cause is simply that the
// caller is shutting the worker down.
func (w *Worker) logFailure(ctx context.Context, doing string, err error) {
	if ctx.Err() != nil {
		return
	}
	w.log.ErrorContext(ctx, "mempher: worker failed but is continuing",
		"worker", string(w.id), "doing", doing, "error", err.Error())
}

// DrainOnce claims and runs at most one batch, and reports how many succeeded.
// It is what a test calls to make deferred work happen at a known point, and
// what a one-shot command calls instead of a loop.
func (w *Worker) DrainOnce(ctx context.Context) (int, error) {
	jobs, err := w.queue.Lease(ctx, LeaseRequest{
		Kinds:    []JobKind{JobKindEncode},
		Worker:   w.id,
		Limit:    w.batch,
		Duration: w.lease,
		Now:      w.clock.Now(),
	})
	if err != nil {
		return 0, err
	}

	done := 0
	for _, job := range jobs {
		// Stopping mid-batch leaves the rest leased; they are reclaimed once
		// the lease expires, which is the same path a crash takes.
		if err := ctx.Err(); err != nil {
			return done, fmt.Errorf("mempher: worker %s: %w", w.id, err)
		}
		if err := w.runJob(ctx, job); err != nil {
			w.reportFailure(ctx, job, err)
			continue
		}
		if err := w.queue.Succeed(ctx, job.ID, w.id, w.clock.Now()); err != nil {
			w.logFailure(ctx, "marking a job done", err)
			continue
		}
		done++
	}
	return done, nil
}

// runJob does the work a job describes.
func (w *Worker) runJob(ctx context.Context, job Job) error {
	switch job.Kind {
	case JobKindEncode:
		return w.encode(ctx, job)
	default:
		return fmt.Errorf("job %s has kind %q: %w", job.ID, job.Kind, ErrInvalidJobKind)
	}
}

// encode embeds a job's episodes and writes the encodings.
//
// It is idempotent by construction: the episodes are immutable, and the write is
// an upsert, so a job that is retried or reclaimed produces exactly the same
// rows. The whole batch is embedded in one call, because that is where the money
// goes.
func (w *Worker) encode(ctx context.Context, job Job) error {
	episodes := make([]Episode, 0, len(job.Episodes))
	texts := make([]string, 0, len(job.Episodes))
	for _, id := range job.Episodes {
		episode, err := w.store.Episode(ctx, job.Scope, id)
		if err != nil {
			return fmt.Errorf("read episode %s: %w", id, err)
		}
		episodes = append(episodes, episode)
		texts = append(texts, episode.Content)
	}

	vectors, err := w.embedder.EmbedDocuments(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed %d episodes: %w", len(texts), err)
	}
	if len(vectors) != len(episodes) {
		return fmt.Errorf("embedder %q returned %d vectors for %d episodes: %w",
			w.embedder.Model(), len(vectors), len(episodes), ErrInvalidConfig)
	}

	encodedAt := w.clock.Now()
	for i, episode := range episodes {
		if err := w.store.PutEncoding(ctx, Encoding{
			Episode:   episode.ID,
			Model:     w.embedder.Model(),
			Vector:    vectors[i],
			EncodedAt: encodedAt,
		}); err != nil {
			return fmt.Errorf("write encoding for episode %s: %w", episode.ID, err)
		}
	}
	return nil
}

// reportFailure hands a failed job back to the queue with a retry time.
func (w *Worker) reportFailure(ctx context.Context, job Job, cause error) {
	now := w.clock.Now()
	if err := w.queue.Fail(ctx, job.ID, w.id, cause, now, now.Add(retryBackoff(job.Attempts))); err != nil {
		w.logFailure(ctx, "recording a job failure", err)
		return
	}
	w.logFailure(ctx, fmt.Sprintf("running %s job %s", job.Kind, job.ID), cause)
}

// retryBackoff is how long a job waits before its next attempt: doubling from
// [DefaultRetryBackoff] up to [MaxRetryBackoff].
//
// It has no jitter, so a test can predict it exactly. The herd it might stampede
// is a database that just told every worker the same thing, which is a problem
// worth having.
func retryBackoff(attempt int) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	backoff := DefaultRetryBackoff
	for range attempt - 1 {
		backoff *= 2
		if backoff >= MaxRetryBackoff {
			return MaxRetryBackoff
		}
	}
	return backoff
}
