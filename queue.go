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
)

// Valid reports whether k is one of the defined kinds.
func (k JobKind) Valid() bool {
	switch k {
	case JobKindEncode:
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

// Queue is the deferred-work port. Leasing is at-least-once: a worker that dies
// holding a lease has its job reclaimed, so every job body must be idempotent.
type Queue interface {
	// Enqueue adds one job. Duplicate work for the same kind, scope and
	// episodes that is still pending or running collapses onto the existing
	// job.
	Enqueue(ctx context.Context, job NewJob) (Job, error)

	// Lease atomically claims ready jobs for one worker, skipping jobs other
	// workers hold.
	Lease(ctx context.Context, req LeaseRequest) ([]Job, error)

	// Succeed marks a leased job done, or returns [ErrJobNotLeased] if the
	// caller no longer holds the lease.
	Succeed(ctx context.Context, id JobID, worker WorkerID, at time.Time) error

	// Fail records a failed attempt. The job returns to pending with RunAfter
	// set to retryAt, unless that attempt exhausted MaxAttempts, in which
	// case it goes dead.
	Fail(ctx context.Context, id JobID, worker WorkerID, cause error, retryAt time.Time) error

	// Reclaim returns jobs whose leases expired before now to pending, and
	// reports how many. It is how work survives a worker crash.
	Reclaim(ctx context.Context, now time.Time) (int, error)

	// Job returns one job by id, or [ErrNotFound].
	Job(ctx context.Context, id JobID) (Job, error)
}
