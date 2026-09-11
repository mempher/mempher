// The queue and the catalogue, as an operator reads them.

package ops

import (
	"time"

	"github.com/mempher/mempher"
)

// JobQuery lists jobs, oldest first.
//
// It exists because a job that needs a human is currently unreachable: the queue
// can be asked about a job whose id you already have, and nothing hands you the
// id of the job that died. A queue whose failures cannot be found is a queue
// that fails silently.
type JobQuery struct {
	// Scope restricts the listing to one partition. Empty means every scope,
	// which is what looking for dead work wants.
	Scope mempher.ScopeID
	// Kinds, when non-empty, keeps only jobs of one of them.
	Kinds []mempher.JobKind
	// States, when non-empty, keeps only jobs in one of them. Listing
	// [JobStateDead] is the reason this type exists.
	States []mempher.JobState
	// After pages the listing: only jobs ordered after it are returned. Ids
	// are uuidv7, so that order is the order the jobs were created in.
	After mempher.JobID
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
	Kind  mempher.JobKind
	State mempher.JobState
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
	States []mempher.JobState
	// Limit caps how many rows one call deletes. Zero means a store-chosen
	// default. It is here so that purging a year of history is a loop of
	// short transactions rather than one that holds locks for minutes.
	Limit int
}

// PendingEncodings asks which episodes still lack an encoding.
//
// The answer is derived from L0 and the encodings table alone, so it is the
// safety net behind the queue: enqueue what it returns and any episode missed by
// a lost job, or left behind by a change of model, gets encoded.
type PendingEncodings struct {
	// Scope restricts the search to one partition. Empty means all scopes,
	// which is what a backfill wants.
	Scope mempher.ScopeID
	// Model is the vector space to look for. Required.
	Model mempher.ModelID
	// AfterSeq pages a large backfill within one scope. Requires Scope.
	AfterSeq int64
	// Limit is the most ids to return. Zero means a store-chosen default.
	Limit int
}

// PendingExtractions asks which episodes an extractor has not yet read.
//
// The answer is derived from L0 and the extraction markers alone, so it is the
// safety net behind the queue: enqueue what it returns and any episode missed by
// a lost job, or left behind by a change of extractor, gets read.
type PendingExtractions struct {
	// Scope restricts the search to one partition. Empty means all scopes,
	// which is what a backfill wants.
	Scope mempher.ScopeID
	// Extractor is whose markers to look for. Required.
	Extractor mempher.ExtractorID
	// AfterSeq pages a large backfill within one scope. Requires Scope.
	AfterSeq int64
	// Limit is the most ids to return. Zero means a store-chosen default.
	Limit int
}
