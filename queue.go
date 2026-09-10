// The deferred-work port.

package mempher

import (
	"context"
	"time"
)

// Queue defaults, used when the corresponding field is left zero.
const (
	// DefaultMaxAttempts is how many times a job runs before going dead.
	DefaultMaxAttempts = 5
	// DefaultLeaseDuration is how long a leased job stays invisible to other
	// workers before it is reclaimable.
	DefaultLeaseDuration = 5 * time.Minute
)

// JobKind is the kind of deferred work a job represents. The set is closed and
// mirrored by a CHECK constraint; later stages add kinds by forward migration.
type JobKind string

// The job kinds this stage understands.
const (
	// JobKindEncode embeds an episode's content and writes the encoding.
	JobKindEncode JobKind = "encode"
	// JobKindExtract reads an episode and writes the facts it yields.
	//
	// It is separate from JobKindEncode rather than one "consolidate" kind
	// because the two projections fail, retry and get backfilled
	// independently: an embedding model change reruns every encode and no
	// extraction, and a prompt fix reruns the reverse. One kind would couple
	// a cheap backfill to an expensive one.
	JobKindExtract JobKind = "extract"
)

// Valid reports whether k is one of the defined kinds.
func (k JobKind) Valid() bool {
	switch k {
	case JobKindEncode, JobKindExtract:
		return true
	default:
		return false
	}
}

// String returns the job kind name.
func (k JobKind) String() string { return string(k) }

// JobState is where a job sits in its lifecycle. A job runs at most MaxAttempts
// times; the attempt that exhausts them leaves it dead rather than retrying.
type JobState string

// The states a job moves through.
const (
	// JobStatePending is claimable once RunAfter has passed.
	JobStatePending JobState = "pending"
	// JobStateRunning is leased to a worker until LeasedUntil.
	JobStateRunning JobState = "running"
	// JobStateDone completed successfully and is kept for audit.
	JobStateDone JobState = "done"
	// JobStateDead exhausted its attempts and needs a human.
	JobStateDead JobState = "dead"
)

// Valid reports whether s is one of the defined states.
func (s JobState) Valid() bool {
	switch s {
	case JobStatePending, JobStateRunning, JobStateDone, JobStateDead:
		return true
	default:
		return false
	}
}

// String returns the state name.
func (s JobState) String() string { return string(s) }

// NewJob describes work to enqueue.
type NewJob struct {
	// Kind is what work to do. Required.
	Kind JobKind
	// Scope is the partition the work belongs to. Required.
	Scope ScopeID
	// Episodes are the episodes to operate on, and must not be empty.
	//
	// Deferred work always refers to episodes, so the queue names them
	// directly instead of carrying an opaque payload: L0 stays the only input
	// to any projection, and no jsonb blob is waiting to become an
	// interface{}. Within an [AppendCommand] a [Store] fills this in.
	Episodes []EpisodeID
	// RunAfter is the earliest time the job may be leased. Zero means now.
	RunAfter time.Time
	// MaxAttempts is how many times to try. Zero means [DefaultMaxAttempts].
	MaxAttempts int
}

// Job is a unit of deferred work as the queue holds it.
type Job struct {
	// ID is the uuidv7 primary key, minted by the database.
	ID JobID
	// Kind is what work to do.
	Kind JobKind
	// Scope is the partition the work belongs to.
	Scope ScopeID
	// Episodes are the episodes to operate on; never empty.
	Episodes []EpisodeID
	// State is where the job sits in its lifecycle.
	State JobState
	// Attempts is how many times it has been leased.
	Attempts int
	// MaxAttempts is how many times it may be leased before going dead.
	MaxAttempts int
	// RunAfter is the earliest time it may be leased.
	RunAfter time.Time
	// LeasedUntil is when the current lease expires; zero unless running.
	LeasedUntil time.Time
	// LeasedBy is the worker holding it; empty unless running.
	LeasedBy WorkerID
	// LastError is the most recent failure message, if any.
	LastError string
	// CreatedAt and UpdatedAt come from the [Clock].
	CreatedAt time.Time
	UpdatedAt time.Time
}

// LeaseRequest claims a batch of ready jobs for one worker.
type LeaseRequest struct {
	// Kinds restricts what to claim. Empty means any kind.
	Kinds []JobKind
	// Worker is the claiming worker's identity. Required.
	Worker WorkerID
	// Limit is the most jobs to claim. Zero means one.
	Limit int
	// Duration is how long the lease holds. Zero means
	// [DefaultLeaseDuration].
	Duration time.Duration
	// Now is the current time, supplied rather than read by the [Store].
	// Required.
	Now time.Time
}

// JobQuery lists jobs, oldest first.
//
// It exists because a job that needs a human is currently unreachable: the queue
// can be asked about a job whose id you already have, and nothing hands you the
// id of the job that died. A queue whose failures cannot be found is a queue
// that fails silently.
type JobQuery struct {
	// Scope restricts the listing to one partition. Empty means every scope,
	// which is what looking for dead work wants.
	Scope ScopeID
	// Kinds, when non-empty, keeps only jobs of one of them.
	Kinds []JobKind
	// States, when non-empty, keeps only jobs in one of them. Listing
	// [JobStateDead] is the reason this type exists.
	States []JobState
	// After pages the listing: only jobs ordered after it are returned. Ids
	// are uuidv7, so that order is the order the jobs were created in.
	After JobID
	// Limit is the most jobs to return. Zero means a store-chosen default.
	Limit int
}

// JobCount is the depth and age of one bucket of the queue.
//
// Age is reported beside depth because depth alone does not say whether a queue
// is working. A thousand pending jobs is healthy if the oldest arrived a second
// ago and is an outage if it arrived yesterday.
type JobCount struct {
	// Kind and State name the bucket.
	Kind  JobKind
	State JobState
	// Jobs is how many jobs it holds.
	Jobs int
	// Oldest is the CreatedAt of the oldest job in the bucket, so that a
	// stalled queue is visible without reading a job.
	Oldest time.Time
}

// PurgeRequest deletes finished jobs.
//
// The queue is not L0. It is a work list, and a job that has been done keeps a
// row for as long as it is worth auditing and no longer: a table that only grows
// makes every listing and every count slower for the sake of history nothing
// reads. What it may never delete is outstanding work, which is why the states
// it accepts are checked rather than passed through.
type PurgeRequest struct {
	// Before keeps jobs updated at or after this instant. Required: a purge
	// with no horizon is a truncate wearing a filter.
	Before time.Time
	// States restricts what to delete, and may name only [JobStateDone] and
	// [JobStateDead]. Empty means both. Naming [JobStatePending] or
	// [JobStateRunning] is [ErrInvalidJobState]: deleting a job that is
	// still owed would lose the work with no record that it was lost.
	States []JobState
	// Limit caps how many rows one call deletes. Zero means a store-chosen
	// default. It is here so that purging a year of history is a loop of
	// short transactions rather than one that holds locks for minutes.
	Limit int
}

// Queue is the deferred-work port. Leasing is at-least-once: a worker that dies
// holding a lease has its job reclaimed, so every job body must be idempotent.
//
// The first six methods run the queue. The last four operate it: they are what
// an on-call engineer, a metrics exporter and a nightly maintenance job need,
// and nothing on the write or read path calls them.
type Queue interface {
	// Enqueue adds one job at time now. Duplicate work for the same kind,
	// scope and episodes that is still pending or running collapses onto the
	// existing job, which is returned instead.
	Enqueue(ctx context.Context, job NewJob, now time.Time) (Job, error)

	// Lease atomically claims ready jobs for one worker, skipping jobs other
	// workers hold.
	Lease(ctx context.Context, req LeaseRequest) ([]Job, error)

	// Succeed marks a leased job done, or returns [ErrJobNotLeased] if the
	// caller no longer holds the lease.
	Succeed(ctx context.Context, id JobID, worker WorkerID, at time.Time) error

	// Fail records at time at that an attempt failed. The job returns to
	// pending with RunAfter set to retryAt, unless that attempt exhausted
	// MaxAttempts, in which case it goes dead.
	Fail(ctx context.Context, id JobID, worker WorkerID, cause error, at, retryAt time.Time) error

	// Reclaim releases jobs whose leases expired before now, and reports how
	// many. It is how work survives a worker crash. A job that had attempts
	// left returns to pending; one that had none goes dead.
	Reclaim(ctx context.Context, now time.Time) (int, error)

	// Job returns one job by id, or [ErrNotFound].
	Job(ctx context.Context, id JobID) (Job, error)

	// Jobs lists jobs matching q, oldest first.
	Jobs(ctx context.Context, q JobQuery) ([]Job, error)

	// Stats counts the queue by kind and state, and reports the age of the
	// oldest job in each bucket. An empty scope counts every scope. Buckets
	// holding no jobs are omitted rather than reported as zero, so a caller
	// exporting these as metrics must decide for itself whether a missing
	// bucket is a zero or a gap.
	Stats(ctx context.Context, scope ScopeID) ([]JobCount, error)

	// Retry returns one dead job to pending at time now, with its attempts
	// reset, and returns it as it now stands. It is what to call once the
	// cause in [Job.LastError] has been dealt with.
	//
	// It fails with [ErrJobNotDead] if the job is in any other state, and
	// with [ErrJobOutstanding] if the same work has since been enqueued
	// again -- in which case the work will happen anyway and the caller can
	// ignore it. Reviving a job is deliberately the only way a job moves
	// backwards: everything else about a job's lifecycle is decided by the
	// worker holding it.
	Retry(ctx context.Context, id JobID, now time.Time) (Job, error)

	// Purge deletes finished jobs matching req and reports how many rows
	// went. It never deletes a job that is still owed.
	Purge(ctx context.Context, req PurgeRequest) (int, error)
}
