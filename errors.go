// Expected conditions, reported as wrapped sentinels.

package mempher

import "errors"

// Test these with [errors.Is]. Every error this package returns wraps one of
// them, or the driver error underneath.
var (
	// ErrInvalidScope means a [ScopeID] was empty, over-long or not clean UTF-8.
	ErrInvalidScope = errors.New("invalid scope id")
	// ErrInvalidContent means episode content was empty or over-long.
	ErrInvalidContent = errors.New("invalid episode content")
	// ErrInvalidRole means a [Role] was outside the defined set.
	ErrInvalidRole = errors.New("invalid role")
	// ErrInvalidSource means a [Source] was empty or over-long.
	ErrInvalidSource = errors.New("invalid source")
	// ErrInvalidActor means an [Actor] was over-long.
	ErrInvalidActor = errors.New("invalid actor")
	// ErrInvalidBinding means a [Binding] broke a key, value or count limit.
	ErrInvalidBinding = errors.New("invalid binding")
	// ErrInvalidChannel means an unknown [Channel] was requested.
	ErrInvalidChannel = errors.New("invalid channel")
	// ErrInvalidJobKind means an unknown [JobKind] was enqueued.
	ErrInvalidJobKind = errors.New("invalid job kind")
	// ErrInvalidJobState means a [JobState] was outside the defined set, or
	// was one the operation refuses: a [PurgeRequest] naming pending or
	// running work is the case this exists for.
	ErrInvalidJobState = errors.New("invalid job state")
	// ErrInvalidFact means an [Assertion] broke a length limit, left a
	// required part empty, or carried a confidence outside [0,1].
	ErrInvalidFact = errors.New("invalid fact")
	// ErrInvalidValidity means a [Validity] had no start, or ended at or
	// before it began.
	ErrInvalidValidity = errors.New("invalid validity window")
	// ErrInvalidExtraction means an [ExtractCommand] named no episodes, or
	// asserted or retracted more than one extraction may.
	ErrInvalidExtraction = errors.New("invalid extraction")
	// ErrInvalidConfig means a [Config] or [WorkerConfig] was missing a
	// required port or held a nonsensical value.
	ErrInvalidConfig = errors.New("invalid config")

	// ErrNotFound means the episode, job or scope does not exist, or exists
	// outside the scope that asked. Scope isolation is reported as absence,
	// never as a permission error.
	ErrNotFound = errors.New("not found")

	// ErrDimensionMismatch means a [Vector] did not match the width the
	// schema or the [Embedder] declares.
	ErrDimensionMismatch = errors.New("vector dimension mismatch")
	// ErrEmbedderMismatch means the configured [Embedder] disagrees with what
	// the database was migrated for.
	ErrEmbedderMismatch = errors.New("embedder does not match schema")
	// ErrFactConflict means an assertion overlapped an existing fact for the
	// same scope, extractor, subject, predicate and object over a window it
	// contradicts. It is reported rather than resolved: silently widening one
	// of the two windows would lose the disagreement that caused it.
	ErrFactConflict = errors.New("fact validity windows conflict")
	// ErrAllChannelsFailed means every retrieval channel failed, so the result
	// would be silently empty rather than merely degraded. Lesser failures are
	// reported in [Recollection.Channels].
	ErrAllChannelsFailed = errors.New("all retrieval channels failed")

	// ErrJobNotLeased means a completion or failure was reported for a job
	// this worker does not hold, usually because its lease expired and the job
	// was reclaimed.
	ErrJobNotLeased = errors.New("job not leased by this worker")
	// ErrJobNotDead means a job was sent back to pending from a state other
	// than dead. Work that is done is re-done by backfilling from L0, which
	// derives what is missing, rather than by reviving the job that did it.
	ErrJobNotDead = errors.New("job is not dead")
	// ErrJobOutstanding means the work a revived job names is already queued
	// or running. It is safe to ignore: the work happens either way, and the
	// dead row stays dead so that the two cannot both run.
	ErrJobOutstanding = errors.New("the same work is already queued")
	// ErrClosed means the store was closed.
	ErrClosed = errors.New("store is closed")
)
