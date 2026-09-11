// Recovering the work L0 implies but the queue no longer holds.

package ops

import (
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/mempher/mempher"
)

// Backfill defaults, used when the corresponding [BackfillRequest] field is
// zero.
const (
	// DefaultBackfillBatch is how many episodes one backfilled job names.
	//
	// It is larger than one -- which is what an append enqueues -- because a
	// backfill is measured in round trips per episode and a live append is
	// measured in latency. Sixteen is small enough that one failure loses
	// little and large enough that a million episodes is not a million model
	// calls.
	DefaultBackfillBatch = 16
	// DefaultBackfillLimit is how many episodes one [Ops.Backfill] call
	// enqueues work for, per kind, before returning its cursor.
	DefaultBackfillLimit = 1000
)

// BackfillRequest asks for the deferred work that L0 implies but the queue does
// not hold: episodes with no encoding for the configured [Embedder], and
// episodes no extraction marker covers for the configured [Extractor].
//
// It is the answer to the three ways work goes missing. A job can die after
// exhausting its attempts. An [Extractor] can be configured on a memory that
// already holds episodes, which enqueued no extract jobs when they arrived. And
// a model or a prompt can change, which changes the key a projection is stored
// under and leaves every episode pending again under the new one.
type BackfillRequest struct {
	// Scope restricts the backfill to one partition. Empty walks every
	// scope, in id order, which is what a change of model wants.
	Scope mempher.ScopeID
	// Kinds restricts what to enqueue. Empty means every kind this Ops
	// can enqueue, which excludes [JobKindExtract] when it has no [Extractor].
	//
	// The two kinds are worth separating here for the same reason they are
	// separate kinds: a new embedding model reruns every encode and no
	// extraction, and a fixed prompt reruns the reverse.
	Kinds []mempher.JobKind
	// AfterSeq resumes a paged backfill within one scope, and requires
	// Scope, because Seq is only ordered inside one.
	AfterSeq int64
	// BatchSize is how many episodes one enqueued job names. Zero means
	// [DefaultBackfillBatch].
	BatchSize int
	// Limit is the most episodes to enqueue work for, per kind, in this
	// call. Zero means [DefaultBackfillLimit].
	Limit int
	// RunAfter defers the enqueued jobs until an instant of the caller's
	// choosing, leaving live appends to be drained first. Zero enqueues them
	// as ready.
	//
	// It is the only lever the queue offers over ordering: jobs are leased by
	// kind and then by RunAfter, so a backfill enqueued as ready competes
	// with every append that arrives while it drains.
	RunAfter time.Time
}

// BackfillCount is what one kind of work cost in a [BackfillRequest].
type BackfillCount struct {
	// Episodes is how many pending episodes were found.
	Episodes int
	// Jobs is how many jobs now cover them, whether this call created them
	// or found them already outstanding.
	Jobs int
	// Collapsed is how many of those jobs already existed, recognised by a
	// CreatedAt earlier than this call. Re-running an interrupted backfill
	// collapses onto the work it already enqueued rather than duplicating
	// it, and this is how that shows.
	//
	// It is a report and not a guarantee: a [Clock] that does not advance
	// between two calls, which is the normal case in a test, cannot tell the
	// two apart.
	Collapsed int
}

// BackfillResult reports what a [Ops.Backfill] call enqueued, and where to
// resume.
//
// Backfill enqueues work and never does it. It makes no model call, writes no
// projection, and hands everything to the ordinary consolidation loop, so there
// is one code path that embeds an episode and one that extracts from it, whether
// the episode arrived a second ago or is being replayed under a new model.
type BackfillResult struct {
	// Encode is what the encode kind found and enqueued.
	Encode BackfillCount
	// Extract is what the extract kind found and enqueued. It is zero when
	// the worker has no [Extractor], or when Kinds excluded it.
	Extract BackfillCount
	// NextScope and NextSeq resume the walk: pass them back as
	// [BackfillRequest.Scope] and [BackfillRequest.AfterSeq] to continue
	// from where this call stopped. They are zero once Done.
	NextScope mempher.ScopeID
	NextSeq   int64
	// Done reports that the walk reached the end of what the request asked
	// for. It does not mean the queue is empty: the jobs this call enqueued
	// still have to be drained.
	Done bool
}

// Backfill enqueues the deferred work that L0 implies and the queue does not
// hold, and reports what it found.
//
// It is the safety net behind the queue, and it derives its answer from L0 and
// the projection tables alone rather than from the queue's own history: an
// episode with no encoding for this worker's [Embedder] needs encoding,
// whatever happened to the job that should have done it. That is the whole
// reason a projection is allowed to be dropped -- truncate the encodings, run
// this, and the vectors come back.
//
// It is safe to run against a live system and safe to run twice. Pending
// episodes come back in Seq order, so the batches this call would enqueue are
// the batches the last one enqueued, and the queue collapses the duplicates. The
// exception is a batch that was partly drained in between, whose remaining
// episodes regroup into a batch the queue has not seen; that enqueues an
// overlapping job, which is wasteful and harmless, because every job body is
// idempotent.
//
// One call does one page. A caller that wants the whole log loops on
// [BackfillResult.Done], which keeps a backfill of ten million episodes
// cancellable and keeps its cost visible between pages instead of inside one
// call that does not return.
func (o *Ops) Backfill(ctx context.Context, req BackfillRequest) (BackfillResult, error) {
	if err := req.Validate(); err != nil {
		return BackfillResult{}, err
	}
	kinds, err := o.backfillKinds(req.Kinds)
	if err != nil {
		return BackfillResult{}, err
	}
	batch := req.BatchSize
	if batch == 0 {
		batch = DefaultBackfillBatch
	}
	limit := req.Limit
	if limit == 0 {
		limit = DefaultBackfillLimit
	}

	// One instant for the whole call, because it is what Collapsed is measured
	// against: two pages enqueued a millisecond apart would report the same job
	// differently.
	now := o.clock.Now()

	cursors := make([]*backfillCursor, len(kinds))
	for i, kind := range kinds {
		cursors[i] = &backfillCursor{kind: kind, remaining: limit}
	}

	var out BackfillResult
	after := mempher.ScopeID("")
	for {
		scopes := []Scope{{ID: req.Scope}}
		if req.Scope == "" {
			listed, err := o.store.Scopes(ctx, ScopeQuery{After: after, Limit: backfillScopePage})
			if err != nil {
				return out, fmt.Errorf("mempher: backfill: list scopes: %w", err)
			}
			if len(listed) == 0 {
				out.Done = true
				return out, nil
			}
			scopes = listed
			after = listed[len(listed)-1].ID
		}

		for _, scope := range scopes {
			// Only the scope the caller resumed into starts part-way through.
			seed := int64(0)
			if scope.ID == req.Scope {
				seed = req.AfterSeq
			}
			for _, cursor := range cursors {
				cursor.seq, cursor.exhausted = seed, false
			}

			if err := o.backfillScope(ctx, scope.ID, cursors, req, batch, now, &out); err != nil {
				return out, err
			}
			// A spent budget stops the whole walk rather than only the kind
			// that spent it: there is one cursor, and it cannot say that
			// encode reached one scope while extract reached another.
			if spentBudget(cursors) {
				out.NextScope, out.NextSeq = scope.ID, resumeSeq(cursors)
				return out, nil
			}
		}

		if req.Scope != "" {
			out.Done = true
			return out, nil
		}
	}
}

// backfillScopePage is how many scopes are listed at a time while walking them
// all. It bounds the catalogue read, nothing else: the work itself is bounded by
// [BackfillRequest.Limit].
const backfillScopePage = 100

// backfillCursor is one kind's progress through one scope.
type backfillCursor struct {
	kind mempher.JobKind
	// seq is the last episode this kind has enqueued work for, within the
	// scope being walked.
	seq int64
	// remaining is what is left of this kind's budget for the whole call.
	remaining int
	// exhausted means this scope holds no more pending episodes for this
	// kind, which is different from having run out of budget.
	exhausted bool
}

// backfillScope drains one scope until every kind is exhausted or out of budget.
func (o *Ops) backfillScope(
	ctx context.Context,
	scope mempher.ScopeID,
	cursors []*backfillCursor,
	req BackfillRequest,
	batch int,
	now time.Time,
	out *BackfillResult,
) error {
	for {
		progressed := false
		for _, cursor := range cursors {
			if cursor.exhausted || cursor.remaining == 0 {
				continue
			}

			ids, err := o.pending(ctx, cursor.kind, scope, cursor.seq, cursor.remaining)
			if err != nil {
				return fmt.Errorf("mempher: backfill: pending %s work in scope %q: %w",
					cursor.kind, scope, err)
			}
			if len(ids) == 0 {
				cursor.exhausted = true
				continue
			}

			jobs, collapsed, err := o.enqueueBatches(ctx, cursor.kind, scope, ids, batch, now, req.RunAfter)
			// Counted before the error is returned, so a run that fails
			// half way still reports the half it did.
			count := out.countFor(cursor.kind)
			count.Episodes += len(ids)
			count.Jobs += jobs
			count.Collapsed += collapsed
			if err != nil {
				return fmt.Errorf("mempher: backfill: scope %q: %w", scope, err)
			}

			// The cursor is a Seq and the pending queries answer in ids, so
			// the last episode of the page is read to place it. One extra
			// round trip per page, against a page that has just become as
			// many model calls.
			last, err := o.store.Episode(ctx, scope, ids[len(ids)-1])
			if err != nil {
				return fmt.Errorf("mempher: backfill: read episode %s: %w", ids[len(ids)-1], err)
			}
			cursor.seq = last.Seq
			cursor.remaining -= len(ids)
			progressed = true
		}
		if !progressed || spentBudget(cursors) {
			return nil
		}
	}
}

// pending asks what one kind has left to do in one scope.
func (o *Ops) pending(
	ctx context.Context,
	kind mempher.JobKind,
	scope mempher.ScopeID,
	afterSeq int64,
	limit int,
) ([]mempher.EpisodeID, error) {
	switch kind {
	case mempher.JobKindEncode:
		return o.store.PendingEncodings(ctx, PendingEncodings{
			Scope:    scope,
			Model:    o.embedder.Model(),
			AfterSeq: afterSeq,
			Limit:    limit,
		})
	case mempher.JobKindExtract:
		return o.facts.PendingExtractions(ctx, PendingExtractions{
			Scope:     scope,
			Extractor: o.extractor.Model(),
			AfterSeq:  afterSeq,
			Limit:     limit,
		})
	default:
		return nil, fmt.Errorf("kind %q: %w", kind, mempher.ErrInvalidJobKind)
	}
}

// enqueueBatches turns a page of pending episodes into jobs, and reports how
// many of those jobs the queue already held.
func (o *Ops) enqueueBatches(
	ctx context.Context,
	kind mempher.JobKind,
	scope mempher.ScopeID,
	ids []mempher.EpisodeID,
	batch int,
	now, runAfter time.Time,
) (jobs, collapsed int, err error) {
	for start := 0; start < len(ids); start += batch {
		job, err := o.store.Enqueue(ctx, mempher.NewJob{
			Kind:     kind,
			Scope:    scope,
			Episodes: slices.Clone(ids[start:min(start+batch, len(ids))]),
			RunAfter: runAfter,
		}, now)
		if err != nil {
			return jobs, collapsed, fmt.Errorf("enqueue a %s job: %w", kind, err)
		}
		jobs++
		// Enqueue returns the outstanding job when it collapses one, and an
		// outstanding job is older than this call.
		if job.CreatedAt.Before(now) {
			collapsed++
		}
	}
	return jobs, collapsed, nil
}

// backfillKinds resolves what to enqueue, rejecting work this worker could not
// drain. A worker with no [Extractor] cannot backfill extractions for the same
// reason it does not claim them: nothing would name the facts.
func (o *Ops) backfillKinds(requested []mempher.JobKind) ([]mempher.JobKind, error) {
	if len(requested) == 0 {
		return o.kinds, nil
	}
	out := make([]mempher.JobKind, 0, len(requested))
	for _, kind := range requested {
		if !slices.Contains(o.kinds, kind) {
			return nil, fmt.Errorf(
				"mempher: backfill: this worker handles %v, not %s jobs: %w",
				o.kinds, kind, mempher.ErrInvalidConfig)
		}
		if !slices.Contains(out, kind) {
			out = append(out, kind)
		}
	}
	return out, nil
}

// countFor is where one kind's tally lives.
func (r *BackfillResult) countFor(kind mempher.JobKind) *BackfillCount {
	if kind == mempher.JobKindExtract {
		return &r.Extract
	}
	return &r.Encode
}

// spentBudget reports whether any kind has used its whole allowance, which is
// what ends a walk.
func spentBudget(cursors []*backfillCursor) bool {
	for _, cursor := range cursors {
		if cursor.remaining == 0 {
			return true
		}
	}
	return false
}

// resumeSeq is where to start the scope again: the least progress any kind made
// that still has work there. Resuming a kind earlier than it reached re-enqueues
// work the queue already holds, which collapses; resuming it later would skip
// episodes for good.
func resumeSeq(cursors []*backfillCursor) int64 {
	next := int64(-1)
	for _, cursor := range cursors {
		if cursor.exhausted {
			continue
		}
		if next < 0 || cursor.seq < next {
			next = cursor.seq
		}
	}
	if next < 0 {
		return 0
	}
	return next
}
