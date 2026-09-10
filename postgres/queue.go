// The deferred-work port: claiming jobs, finishing them, and recovering the ones
// a dead worker was holding.

package postgres

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/mempher/mempher"
)

// maxLastErrorLen bounds what a failure message costs. A stack trace or a
// megabyte of driver output tells you nothing the first two thousand characters
// did not.
const maxLastErrorLen = 2000

// Defaults for the operational queries, used when the corresponding field is
// zero.
const (
	// DefaultJobsLimit is how many jobs [Store.Jobs] returns when a query
	// does not say.
	DefaultJobsLimit = 100
	// DefaultPurgeLimit is how many rows one [Store.Purge] call deletes when
	// a request does not say. It is deliberately modest: a purge is a
	// maintenance loop of short transactions, not one long one.
	DefaultPurgeLimit = 1000
)

// jobColumns is the read shape of a job, in the order scanJob expects.
const jobColumns = `id, kind, scope_id, episode_ids, state, attempts, max_attempts,
	run_after, leased_until, leased_by, last_error, created_at, updated_at`

// enqueueSQL inserts a job, or returns the outstanding one it would have
// duplicated, in a single statement.
//
// The partial unique index only covers pending and running rows, so the same
// episode can be encoded again later, after a change of model, while work that
// is still outstanding is never queued twice.
const enqueueSQL = `
WITH inserted AS (
    INSERT INTO mempher.jobs
        (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
    VALUES ($1, $2, $3, $4, $5, $6, $6)
    ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
        DO NOTHING
    RETURNING ` + jobColumns + `
)
SELECT ` + jobColumns + ` FROM inserted
UNION ALL
SELECT ` + jobColumns + ` FROM mempher.jobs
WHERE NOT EXISTS (SELECT 1 FROM inserted)
  AND kind = $1 AND scope_id = $2 AND episode_ids = $3
  AND state IN ('pending', 'running')`

// Enqueue adds one job, or returns the outstanding job it would have duplicated.
func (s *Store) Enqueue(
	ctx context.Context,
	job mempher.NewJob,
	now time.Time,
) (mempher.Job, error) {
	if err := validateNewJob(job, now); err != nil {
		return mempher.Job{}, fmt.Errorf("mempher/postgres: enqueue: %w", err)
	}

	runAfter := job.RunAfter
	if runAfter.IsZero() {
		runAfter = now
	}
	maxAttempts := job.MaxAttempts
	if maxAttempts == 0 {
		maxAttempts = mempher.DefaultMaxAttempts
	}

	row := s.pool.QueryRow(ctx, enqueueSQL,
		string(job.Kind), string(job.Scope), episodeUUIDs(job.Episodes),
		runAfter, int32(maxAttempts), now)

	enqueued, err := scanJob(row)
	switch {
	case isCode(err, pgerrcodeForeignKeyViolat):
		return mempher.Job{}, fmt.Errorf(
			"mempher/postgres: enqueue: scope %q has no episodes yet: %w",
			job.Scope, mempher.ErrNotFound)
	case isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema):
		return mempher.Job{}, fmt.Errorf("mempher/postgres: enqueue: run Migrate first: %w",
			ErrSchemaNotReady)
	case err != nil:
		return mempher.Job{}, fmt.Errorf("mempher/postgres: enqueue %s job: %w", job.Kind, err)
	}
	return enqueued, nil
}

// Lease atomically claims ready jobs for one worker.
//
// SKIP LOCKED is what lets many workers share a queue: a row another worker is
// claiming is passed over rather than waited on, so throughput does not collapse
// into a single file. The attempt is counted here, at claim time, so a worker
// that dies mid-job has still used one of its attempts and cannot retry for ever.
func (s *Store) Lease(ctx context.Context, req mempher.LeaseRequest) ([]mempher.Job, error) {
	if err := validateLease(req); err != nil {
		return nil, fmt.Errorf("mempher/postgres: lease: %w", err)
	}

	limit := req.Limit
	if limit == 0 {
		limit = 1
	}
	duration := req.Duration
	if duration == 0 {
		duration = mempher.DefaultLeaseDuration
	}

	b := &queryBuilder{}
	now := b.arg(req.Now)
	until := b.arg(req.Now.Add(duration))
	worker := b.arg(string(req.Worker))

	var candidates strings.Builder
	fmt.Fprintf(&candidates, `SELECT id FROM mempher.jobs
        WHERE state = 'pending' AND run_after <= %s AND attempts < max_attempts`, now)
	if len(req.Kinds) > 0 {
		kinds := make([]string, len(req.Kinds))
		for i, kind := range req.Kinds {
			kinds[i] = string(kind)
		}
		fmt.Fprintf(&candidates, "\n          AND kind = ANY(%s)", b.arg(kinds))
	}
	fmt.Fprintf(&candidates, "\n        ORDER BY run_after, id\n        LIMIT %s\n        FOR UPDATE SKIP LOCKED",
		b.arg(limit))

	sql := fmt.Sprintf(`
UPDATE mempher.jobs SET
    state        = 'running',
    attempts     = attempts + 1,
    leased_until = %s,
    leased_by    = %s,
    updated_at   = %s
WHERE id IN (
        %s
)
RETURNING %s`, until, worker, now, candidates.String(), jobColumns)

	rows, err := s.pool.Query(ctx, sql, b.args...)
	if err != nil {
		if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
			return nil, fmt.Errorf("mempher/postgres: lease: run Migrate first: %w",
				ErrSchemaNotReady)
		}
		return nil, fmt.Errorf("mempher/postgres: lease for %q: %w", req.Worker, err)
	}
	defer rows.Close()

	out := make([]mempher.Job, 0, limit)
	for rows.Next() {
		job, err := scanJob(rows)
		if err != nil {
			return nil, fmt.Errorf("mempher/postgres: lease for %q: %w", req.Worker, err)
		}
		out = append(out, job)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: lease for %q: %w", req.Worker, err)
	}
	return out, nil
}

// Succeed marks a leased job done.
func (s *Store) Succeed(
	ctx context.Context,
	id mempher.JobID,
	worker mempher.WorkerID,
	at time.Time,
) error {
	if err := validateJobUpdate(id, worker, at); err != nil {
		return fmt.Errorf("mempher/postgres: succeed: %w", err)
	}

	tag, err := s.pool.Exec(ctx, `
		UPDATE mempher.jobs SET
		    state        = 'done',
		    leased_until = NULL,
		    leased_by    = NULL,
		    updated_at   = $3
		WHERE id = $1 AND state = 'running' AND leased_by = $2`,
		uuid.UUID(id), string(worker), at)
	if err != nil {
		return fmt.Errorf("mempher/postgres: succeed job %s: %w", id, err)
	}
	if tag.RowsAffected() == 0 {
		return s.explainLostLease(ctx, "succeed", id, worker)
	}
	return nil
}

// Fail records that an attempt failed.
//
// The attempt was already counted at lease time, so a job whose attempts have
// reached MaxAttempts is dead rather than retried: the alternative is a job that
// retries for ever, which is how a broken extractor becomes a bill.
func (s *Store) Fail(
	ctx context.Context,
	id mempher.JobID,
	worker mempher.WorkerID,
	cause error,
	at, retryAt time.Time,
) error {
	if err := validateJobUpdate(id, worker, at); err != nil {
		return fmt.Errorf("mempher/postgres: fail: %w", err)
	}
	if retryAt.IsZero() {
		return fmt.Errorf("mempher/postgres: fail: retryAt is unset: %w", mempher.ErrInvalidConfig)
	}

	tag, err := s.pool.Exec(ctx, `
		UPDATE mempher.jobs SET
		    state        = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'pending' END,
		    run_after    = CASE WHEN attempts >= max_attempts THEN run_after ELSE $4 END,
		    leased_until = NULL,
		    leased_by    = NULL,
		    last_error   = $5,
		    updated_at   = $3
		WHERE id = $1 AND state = 'running' AND leased_by = $2`,
		uuid.UUID(id), string(worker), at, retryAt, truncateError(cause))
	if err != nil {
		return fmt.Errorf("mempher/postgres: fail job %s: %w", id, err)
	}
	if tag.RowsAffected() == 0 {
		return s.explainLostLease(ctx, "fail", id, worker)
	}
	return nil
}

// Reclaim releases jobs whose leases expired.
//
// A job with attempts left returns to pending. One with none goes dead: attempts
// is counted at lease time, so returning an exhausted job to pending would leave
// a row the next lease could not legally claim.
func (s *Store) Reclaim(ctx context.Context, now time.Time) (int, error) {
	if now.IsZero() {
		return 0, fmt.Errorf("mempher/postgres: reclaim: now is unset: %w",
			mempher.ErrInvalidConfig)
	}

	tag, err := s.pool.Exec(ctx, `
		UPDATE mempher.jobs SET
		    state        = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'pending' END,
		    leased_until = NULL,
		    leased_by    = NULL,
		    last_error   = 'lease expired; the worker holding it is presumed dead',
		    updated_at   = $1
		WHERE state = 'running' AND leased_until < $1`, now)
	if err != nil {
		if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
			return 0, fmt.Errorf("mempher/postgres: reclaim: run Migrate first: %w",
				ErrSchemaNotReady)
		}
		return 0, fmt.Errorf("mempher/postgres: reclaim: %w", err)
	}
	return int(tag.RowsAffected()), nil
}

// Job returns one job by id.
func (s *Store) Job(ctx context.Context, id mempher.JobID) (mempher.Job, error) {
	if id.IsZero() {
		return mempher.Job{}, fmt.Errorf("mempher/postgres: job: id is unset: %w",
			mempher.ErrNotFound)
	}

	job, err := scanJob(s.pool.QueryRow(ctx,
		`SELECT `+jobColumns+` FROM mempher.jobs WHERE id = $1`, uuid.UUID(id)))
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return mempher.Job{}, fmt.Errorf("mempher/postgres: job %s: %w", id, mempher.ErrNotFound)
	case isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema):
		return mempher.Job{}, fmt.Errorf("mempher/postgres: job: run Migrate first: %w",
			ErrSchemaNotReady)
	case err != nil:
		return mempher.Job{}, fmt.Errorf("mempher/postgres: job %s: %w", id, err)
	}
	return job, nil
}

// explainLostLease turns "no rows updated" into which of the two reasons it was.
// It costs a query, but only on a path that is already going to be investigated.
func (s *Store) explainLostLease(
	ctx context.Context,
	op string,
	id mempher.JobID,
	worker mempher.WorkerID,
) error {
	existing, err := s.Job(ctx, id)
	if err != nil {
		return fmt.Errorf("mempher/postgres: %s job %s: %w", op, id, err)
	}
	return fmt.Errorf(
		"mempher/postgres: %s job %s: worker %q does not hold it (state %s, held by %q): %w",
		op, id, worker, existing.State, existing.LeasedBy, mempher.ErrJobNotLeased)
}

// scanJob reads one job in [jobColumns] order.
func scanJob(row scanner) (mempher.Job, error) {
	var (
		job         mempher.Job
		id          uuid.UUID
		kind        string
		state       string
		episodeIDs  []uuid.UUID
		attempts    int32
		maxAttempts int32
		leasedUntil *time.Time
		leasedBy    *string
		lastError   *string
	)
	if err := row.Scan(&id, &kind, &job.Scope, &episodeIDs, &state, &attempts, &maxAttempts,
		&job.RunAfter, &leasedUntil, &leasedBy, &lastError,
		&job.CreatedAt, &job.UpdatedAt); err != nil {
		// Wrapped, not replaced: callers test for pgx.ErrNoRows through it.
		return mempher.Job{}, fmt.Errorf("scan job: %w", err)
	}

	job.ID = mempher.JobID(id)
	job.Kind = mempher.JobKind(kind)
	job.State = mempher.JobState(state)
	if !job.Kind.Valid() {
		return mempher.Job{}, fmt.Errorf("job %s has kind %q: %w",
			job.ID, kind, mempher.ErrInvalidJobKind)
	}
	if !job.State.Valid() {
		return mempher.Job{}, fmt.Errorf("job %s has state %q: %w",
			job.ID, state, mempher.ErrInvalidConfig)
	}

	job.Episodes = make([]mempher.EpisodeID, len(episodeIDs))
	for i, episodeID := range episodeIDs {
		job.Episodes[i] = mempher.EpisodeID(episodeID)
	}
	job.Attempts = int(attempts)
	job.MaxAttempts = int(maxAttempts)
	job.RunAfter = job.RunAfter.UTC()
	job.CreatedAt = job.CreatedAt.UTC()
	job.UpdatedAt = job.UpdatedAt.UTC()
	if leasedUntil != nil {
		job.LeasedUntil = leasedUntil.UTC()
	}
	job.LeasedBy = mempher.WorkerID(deref(leasedBy))
	job.LastError = deref(lastError)
	return job, nil
}

// episodeUUIDs converts episode ids for the driver.
func episodeUUIDs(ids []mempher.EpisodeID) []uuid.UUID {
	out := make([]uuid.UUID, len(ids))
	for i, id := range ids {
		out[i] = uuid.UUID(id)
	}
	return out
}

// truncateError renders a cause for storage, bounded.
func truncateError(cause error) *string {
	if cause == nil {
		return nil
	}
	message := cause.Error()
	if len(message) > maxLastErrorLen {
		message = message[:maxLastErrorLen] + "... (truncated)"
	}
	return &message
}

// validateNewJob rejects a job the schema would reject anyway.
func validateNewJob(job mempher.NewJob, now time.Time) error {
	if err := job.Scope.Validate(); err != nil {
		return fmt.Errorf("scope: %w", err)
	}
	switch {
	case !job.Kind.Valid():
		return fmt.Errorf("kind %q: %w", job.Kind, mempher.ErrInvalidJobKind)
	case len(job.Episodes) == 0:
		return fmt.Errorf("a job must name at least one episode: %w", mempher.ErrInvalidConfig)
	case job.MaxAttempts < 0:
		return fmt.Errorf("MaxAttempts is %d: %w", job.MaxAttempts, mempher.ErrInvalidConfig)
	case now.IsZero():
		return fmt.Errorf("now must be resolved by the caller: %w", mempher.ErrInvalidConfig)
	}
	for i, id := range job.Episodes {
		if id.IsZero() {
			return fmt.Errorf("episode %d is unset: %w", i, mempher.ErrInvalidConfig)
		}
	}
	return nil
}

// validateLease rejects a claim that cannot be made.
func validateLease(req mempher.LeaseRequest) error {
	switch {
	case req.Worker == "":
		return fmt.Errorf("worker is empty: %w", mempher.ErrInvalidConfig)
	case req.Now.IsZero():
		return fmt.Errorf("the Now field must be resolved by the caller: %w", mempher.ErrInvalidConfig)
	case req.Limit < 0:
		return fmt.Errorf("limit is %d: %w", req.Limit, mempher.ErrInvalidConfig)
	case req.Duration < 0:
		return fmt.Errorf("duration is %s: %w", req.Duration, mempher.ErrInvalidConfig)
	}
	for _, kind := range req.Kinds {
		if !kind.Valid() {
			return fmt.Errorf("kind %q: %w", kind, mempher.ErrInvalidJobKind)
		}
	}
	return nil
}

// validateJobUpdate rejects what Succeed and Fail both need.
func validateJobUpdate(id mempher.JobID, worker mempher.WorkerID, at time.Time) error {
	switch {
	case id.IsZero():
		return fmt.Errorf("job id is unset: %w", mempher.ErrInvalidConfig)
	case worker == "":
		return fmt.Errorf("worker is empty: %w", mempher.ErrInvalidConfig)
	case at.IsZero():
		return fmt.Errorf("the time must be resolved by the caller: %w", mempher.ErrInvalidConfig)
	}
	return nil
}

// Jobs lists jobs matching q, oldest first.
//
// Ordering is by id, which is a uuidv7 and therefore the order the jobs were
// created in. That makes the cursor a single column and keeps a listing stable
// while jobs around it change state.
func (s *Store) Jobs(ctx context.Context, q mempher.JobQuery) ([]mempher.Job, error) {
	if err := q.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: jobs: %w", err)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultJobsLimit
	}

	b := &queryBuilder{}
	var sql strings.Builder
	sql.WriteString(`SELECT ` + jobColumns + ` FROM mempher.jobs WHERE true`)
	if q.Scope != "" {
		fmt.Fprintf(&sql, "\n  AND scope_id = %s", b.arg(string(q.Scope)))
	}
	if len(q.Kinds) > 0 {
		kinds := make([]string, len(q.Kinds))
		for i, kind := range q.Kinds {
			kinds[i] = string(kind)
		}
		fmt.Fprintf(&sql, "\n  AND kind = ANY(%s)", b.arg(kinds))
	}
	if len(q.States) > 0 {
		states := make([]string, len(q.States))
		for i, state := range q.States {
			states[i] = string(state)
		}
		fmt.Fprintf(&sql, "\n  AND state = ANY(%s)", b.arg(states))
	}
	if !q.After.IsZero() {
		fmt.Fprintf(&sql, "\n  AND id > %s", b.arg(uuid.UUID(q.After)))
	}
	fmt.Fprintf(&sql, "\nORDER BY id\nLIMIT %s", b.arg(limit))

	rows, err := s.pool.Query(ctx, sql.String(), b.args...)
	if err != nil {
		if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
			return nil, fmt.Errorf("mempher/postgres: jobs: run Migrate first: %w",
				ErrSchemaNotReady)
		}
		return nil, fmt.Errorf("mempher/postgres: jobs: %w", err)
	}
	defer rows.Close()

	out := make([]mempher.Job, 0, min(limit, 128))
	for rows.Next() {
		job, err := scanJob(rows)
		if err != nil {
			return nil, fmt.Errorf("mempher/postgres: jobs: %w", err)
		}
		out = append(out, job)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: jobs: %w", err)
	}
	return out, nil
}

// Stats counts the queue by kind and state, and reports the age of the oldest
// job in each bucket.
//
// It is an aggregate over the whole table, so it costs what the table's history
// costs. That is the argument for [Store.Purge]: a queue nobody prunes makes
// this slower every day, for rows that only ever get counted.
func (s *Store) Stats(
	ctx context.Context,
	scope mempher.ScopeID,
) ([]mempher.JobCount, error) {
	if scope != "" {
		if err := scope.Validate(); err != nil {
			return nil, fmt.Errorf("mempher/postgres: job stats: %w", err)
		}
	}

	b := &queryBuilder{}
	var sql strings.Builder
	sql.WriteString(`SELECT kind, state, count(*), min(created_at)
FROM mempher.jobs
WHERE true`)
	if scope != "" {
		fmt.Fprintf(&sql, "\n  AND scope_id = %s", b.arg(string(scope)))
	}
	sql.WriteString("\nGROUP BY kind, state\nORDER BY kind, state")

	rows, err := s.pool.Query(ctx, sql.String(), b.args...)
	if err != nil {
		if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
			return nil, fmt.Errorf("mempher/postgres: job stats: run Migrate first: %w",
				ErrSchemaNotReady)
		}
		return nil, fmt.Errorf("mempher/postgres: job stats: %w", err)
	}
	defer rows.Close()

	out := make([]mempher.JobCount, 0, 8)
	for rows.Next() {
		var (
			count  mempher.JobCount
			kind   string
			state  string
			jobs   int64
			oldest *time.Time
		)
		if err := rows.Scan(&kind, &state, &jobs, &oldest); err != nil {
			return nil, fmt.Errorf("mempher/postgres: job stats: scan bucket: %w", err)
		}
		count.Kind = mempher.JobKind(kind)
		count.State = mempher.JobState(state)
		count.Jobs = int(jobs)
		if oldest != nil {
			count.Oldest = oldest.UTC()
		}
		out = append(out, count)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: job stats: %w", err)
	}
	return out, nil
}

// Retry returns one dead job to pending at time now, with its attempts reset.
//
// The reset is the point: the attempts were spent on a cause that has since been
// dealt with, and a revived job that could only try once more would die again on
// the first flake. last_error is left standing, exactly as [Store.Fail] leaves it
// on a job it returns to pending, so the row still says what went wrong.
//
// Reviving work the queue is already holding is refused rather than merged: the
// partial unique index that collapses duplicate enqueues would otherwise be
// broken by this one statement.
func (s *Store) Retry(
	ctx context.Context,
	id mempher.JobID,
	now time.Time,
) (mempher.Job, error) {
	if id.IsZero() {
		return mempher.Job{}, fmt.Errorf("mempher/postgres: retry: id is unset: %w",
			mempher.ErrNotFound)
	}
	if now.IsZero() {
		return mempher.Job{}, fmt.Errorf("mempher/postgres: retry: now is unset: %w",
			mempher.ErrInvalidConfig)
	}

	job, err := scanJob(s.pool.QueryRow(ctx, `
		UPDATE mempher.jobs SET
		    state        = 'pending',
		    attempts     = 0,
		    run_after    = $2,
		    leased_until = NULL,
		    leased_by    = NULL,
		    updated_at   = $2
		WHERE id = $1 AND state = 'dead'
		RETURNING `+jobColumns, uuid.UUID(id), now))
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		// Either it does not exist or it was not dead, and which of the two
		// decides whether the caller has a bug or a race.
		return mempher.Job{}, s.explainNotRevived(ctx, id)
	case isCode(err, pgerrcodeUniqueViolation):
		return mempher.Job{}, fmt.Errorf(
			"mempher/postgres: retry job %s: the same work is already pending or running: %w",
			id, mempher.ErrJobOutstanding)
	case isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema):
		return mempher.Job{}, fmt.Errorf("mempher/postgres: retry: run Migrate first: %w",
			ErrSchemaNotReady)
	case err != nil:
		return mempher.Job{}, fmt.Errorf("mempher/postgres: retry job %s: %w", id, err)
	}
	return job, nil
}

// explainNotRevived turns "no rows updated" into which of the two reasons it
// was. Like explainLostLease it costs a query, on a path already headed for an
// investigation.
func (s *Store) explainNotRevived(ctx context.Context, id mempher.JobID) error {
	existing, err := s.Job(ctx, id)
	if err != nil {
		return fmt.Errorf("mempher/postgres: retry job %s: %w", id, err)
	}
	return fmt.Errorf("mempher/postgres: retry job %s: it is %s, not dead: %w",
		id, existing.State, mempher.ErrJobNotDead)
}

// Purge deletes finished jobs matching req and reports how many rows went.
//
// It is the one DELETE in this adapter. It is permitted because the queue is a
// work list rather than a layer of memory: nothing is derived from a done job,
// and a dead one has already been read by whoever it was left for. The inner
// SELECT is what makes Limit mean anything -- a bare DELETE ... LIMIT is not
// SQL, and a purge that cannot be bounded is a purge that holds locks for as
// long as the history is deep.
func (s *Store) Purge(ctx context.Context, req mempher.PurgeRequest) (int, error) {
	if err := req.Validate(); err != nil {
		return 0, fmt.Errorf("mempher/postgres: purge: %w", err)
	}
	limit := req.Limit
	if limit == 0 {
		limit = DefaultPurgeLimit
	}
	states := req.States
	if len(states) == 0 {
		states = []mempher.JobState{mempher.JobStateDone, mempher.JobStateDead}
	}
	names := make([]string, len(states))
	for i, state := range states {
		names[i] = string(state)
	}

	tag, err := s.pool.Exec(ctx, `
		DELETE FROM mempher.jobs
		WHERE id IN (
		    SELECT id FROM mempher.jobs
		    WHERE state = ANY($1) AND updated_at < $2
		    ORDER BY id
		    LIMIT $3
		)`, names, req.Before, limit)
	if err != nil {
		if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
			return 0, fmt.Errorf("mempher/postgres: purge: run Migrate first: %w",
				ErrSchemaNotReady)
		}
		return 0, fmt.Errorf("mempher/postgres: purge: %w", err)
	}
	return int(tag.RowsAffected()), nil
}
