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
	// ErrAllChannelsFailed means every retrieval channel failed, so the result
	// would be silently empty rather than merely degraded. Lesser failures are
	// reported in [Recollection.Channels].
	ErrAllChannelsFailed = errors.New("all retrieval channels failed")

	// ErrJobNotLeased means a completion or failure was reported for a job
	// this worker does not hold, usually because its lease expired and the job
	// was reclaimed.
	ErrJobNotLeased = errors.New("job not leased by this worker")
	// ErrClosed means the store was closed.
	ErrClosed = errors.New("store is closed")
)
