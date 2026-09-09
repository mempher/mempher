// L0: appending episodes, reading them back, and replaying them in order.

package postgres

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/mempher/mempher"
)

// episodeColumns is the read shape of an episode, in the order scanEpisode
// expects.
const episodeColumns = `id, scope_id, seq, content, role, actor, source,
	occurred_at, ingested_at, binding`

// appendSQL inserts a scope row, an episode and the episode's jobs in one
// statement, so an episode can never commit without the work it implies.
//
// The scope upsert doubles as the sequence allocator: ON CONFLICT DO UPDATE
// returns last_seq + 1, which makes per-scope sequences dense and gap-free with
// no explicit lock and no extra round trip. Concurrent appends to one scope
// serialise on that row, which is the right semantics for a totally ordered log.
//
// Only id and seq come back. Everything else was supplied by the caller and the
// schema transforms nothing, so reading a 1 MiB content column back would be
// pure waste. The jobs arrive one row each through a LEFT JOIN, which still
// yields the episode row when there are none.
const appendSQL = `
WITH scope AS (
    INSERT INTO mempher.scopes (id, last_seq, created_at)
    VALUES ($1, 1, $7)
    ON CONFLICT (id) DO UPDATE SET last_seq = scopes.last_seq + 1
    RETURNING last_seq
), episode AS (
    INSERT INTO mempher.episodes
        (scope_id, seq, content, role, actor, source, occurred_at, ingested_at, binding)
    SELECT $1, scope.last_seq, $2, $3, $4, $5, $6, $7, $8
    FROM scope
    RETURNING id, seq
), job AS (
    INSERT INTO mempher.jobs
        (kind, scope_id, episode_ids, run_after, max_attempts, created_at, updated_at)
    SELECT spec.kind, $1, ARRAY[episode.id], spec.run_after, spec.max_attempts, $7, $7
    FROM episode,
         unnest($9::text[], $10::timestamptz[], $11::int[])
             AS spec(kind, run_after, max_attempts)
    ON CONFLICT (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running')
        DO NOTHING
    RETURNING id, kind, state, attempts, max_attempts, run_after, created_at, updated_at
)
SELECT episode.id, episode.seq,
       job.id, job.kind, job.state, job.attempts, job.max_attempts,
       job.run_after, job.created_at, job.updated_at
FROM episode LEFT JOIN job ON true
ORDER BY job.kind NULLS LAST`

// Append inserts one episode and the jobs it implies, atomically, creating the
// scope if this is its first episode.
func (s *Store) Append(ctx context.Context, cmd mempher.AppendCommand) (mempher.AppendResult, error) {
	if err := validateAppend(cmd); err != nil {
		return mempher.AppendResult{}, fmt.Errorf("mempher/postgres: append: %w", err)
	}
	ep := cmd.Episode

	kinds := make([]string, len(cmd.Jobs))
	runAfters := make([]time.Time, len(cmd.Jobs))
	maxAttempts := make([]int32, len(cmd.Jobs))
	for i, job := range cmd.Jobs {
		kinds[i] = string(job.Kind)
		runAfters[i] = job.RunAfter
		if runAfters[i].IsZero() {
			runAfters[i] = ep.IngestedAt
		}
		maxAttempts[i] = int32(job.MaxAttempts)
		if maxAttempts[i] == 0 {
			maxAttempts[i] = mempher.DefaultMaxAttempts
		}
	}

	// The column is NOT NULL and CHECKed to be a JSON object, and a nil map
	// marshals to null, so an absent binding is sent as an empty object.
	binding := ep.Binding
	if binding == nil {
		binding = mempher.Binding{}
	}

	rows, err := s.pool.Query(ctx, appendSQL,
		string(ep.Scope), ep.Content, string(ep.Role), string(ep.Actor), string(ep.Source),
		ep.OccurredAt, ep.IngestedAt, binding,
		kinds, runAfters, maxAttempts,
	)
	if err != nil {
		return mempher.AppendResult{}, wrapAppendError(err, ep)
	}
	defer rows.Close()

	result := mempher.AppendResult{Episode: mempher.Episode{
		Scope:      ep.Scope,
		Content:    ep.Content,
		Role:       ep.Role,
		Actor:      ep.Actor,
		Source:     ep.Source,
		OccurredAt: ep.OccurredAt.UTC(),
		IngestedAt: ep.IngestedAt.UTC(),
		Binding:    ep.Binding.Clone(),
	}}

	seen := false
	for rows.Next() {
		var (
			episodeID   uuid.UUID
			seq         int64
			jobID       *uuid.UUID
			jobKind     *string
			jobState    *string
			attempts    *int32
			jobMaxTries *int32
			runAfter    *time.Time
			createdAt   *time.Time
			updatedAt   *time.Time
		)
		if err := rows.Scan(&episodeID, &seq, &jobID, &jobKind, &jobState,
			&attempts, &jobMaxTries, &runAfter, &createdAt, &updatedAt); err != nil {
			return mempher.AppendResult{}, fmt.Errorf("mempher/postgres: append: scan: %w", err)
		}
		if !seen {
			result.Episode.ID = mempher.EpisodeID(episodeID)
			result.Episode.Seq = seq
			seen = true
		}
		if jobID == nil {
			continue
		}
		result.Jobs = append(result.Jobs, mempher.Job{
			ID:          mempher.JobID(*jobID),
			Kind:        mempher.JobKind(deref(jobKind)),
			Scope:       ep.Scope,
			Episodes:    []mempher.EpisodeID{mempher.EpisodeID(episodeID)},
			State:       mempher.JobState(deref(jobState)),
			Attempts:    int(deref(attempts)),
			MaxAttempts: int(deref(jobMaxTries)),
			RunAfter:    deref(runAfter).UTC(),
			CreatedAt:   deref(createdAt).UTC(),
			UpdatedAt:   deref(updatedAt).UTC(),
		})
	}
	if err := rows.Err(); err != nil {
		return mempher.AppendResult{}, wrapAppendError(err, ep)
	}
	if !seen {
		return mempher.AppendResult{}, fmt.Errorf(
			"mempher/postgres: append: the insert returned no row, which should be impossible")
	}
	return result, nil
}

// Episode returns one episode by id within a scope.
//
// The scope is part of the predicate rather than checked afterwards, so an
// episode addressed from the wrong scope is indistinguishable from one that does
// not exist.
func (s *Store) Episode(
	ctx context.Context,
	scope mempher.ScopeID,
	id mempher.EpisodeID,
) (mempher.Episode, error) {
	if err := scope.Validate(); err != nil {
		return mempher.Episode{}, fmt.Errorf("mempher/postgres: episode: %w", err)
	}
	if id.IsZero() {
		return mempher.Episode{}, fmt.Errorf(
			"mempher/postgres: episode: id is unset: %w", mempher.ErrNotFound)
	}

	row := s.pool.QueryRow(ctx,
		`SELECT `+episodeColumns+` FROM mempher.episodes WHERE scope_id = $1 AND id = $2`,
		string(scope), uuid.UUID(id))

	ep, err := scanEpisode(row)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return mempher.Episode{}, fmt.Errorf(
			"mempher/postgres: episode %s in scope %q: %w", id, scope, mempher.ErrNotFound)
	case err != nil:
		return mempher.Episode{}, fmt.Errorf("mempher/postgres: episode %s: %w", id, err)
	}
	return ep, nil
}

// Replay returns up to limit episodes of a scope in Seq order, starting after
// afterSeq. The ordering is the point: consolidation has to see history as it
// happened.
func (s *Store) Replay(
	ctx context.Context,
	scope mempher.ScopeID,
	afterSeq int64,
	limit int,
) ([]mempher.Episode, error) {
	if err := scope.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: replay: %w", err)
	}
	if afterSeq < 0 {
		return nil, fmt.Errorf("mempher/postgres: replay: afterSeq is %d: %w",
			afterSeq, mempher.ErrInvalidConfig)
	}
	if limit <= 0 {
		return nil, fmt.Errorf("mempher/postgres: replay: limit is %d, must be positive: %w",
			limit, mempher.ErrInvalidConfig)
	}

	rows, err := s.pool.Query(ctx,
		`SELECT `+episodeColumns+` FROM mempher.episodes
		 WHERE scope_id = $1 AND seq > $2 ORDER BY seq LIMIT $3`,
		string(scope), afterSeq, limit)
	if err != nil {
		return nil, fmt.Errorf("mempher/postgres: replay scope %q: %w", scope, err)
	}
	defer rows.Close()

	out := make([]mempher.Episode, 0, min(limit, 64))
	for rows.Next() {
		ep, err := scanEpisode(rows)
		if err != nil {
			return nil, fmt.Errorf("mempher/postgres: replay scope %q: %w", scope, err)
		}
		out = append(out, ep)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: replay scope %q: %w", scope, err)
	}
	return out, nil
}

// scanner is what both a single row and a row of a result set satisfy.
type scanner interface {
	Scan(dest ...any) error
}

// scanEpisode reads one episode in [episodeColumns] order.
//
// Timestamps come back as UTC. pgx returns timestamptz at the session's offset,
// which is the same instant but a different rendering, and normalising here makes
// round-trips and test comparisons predictable.
func scanEpisode(row scanner) (mempher.Episode, error) {
	var (
		ep      mempher.Episode
		id      uuid.UUID
		role    string
		binding mempher.Binding
	)
	if err := row.Scan(&id, &ep.Scope, &ep.Seq, &ep.Content, &role, &ep.Actor, &ep.Source,
		&ep.OccurredAt, &ep.IngestedAt, &binding); err != nil {
		// Wrapped, not replaced: callers test for pgx.ErrNoRows through it.
		return mempher.Episode{}, fmt.Errorf("scan episode: %w", err)
	}

	ep.ID = mempher.EpisodeID(id)
	ep.Role = mempher.Role(role)
	if !ep.Role.Valid() {
		return mempher.Episode{}, fmt.Errorf("episode %s has role %q: %w",
			ep.ID, role, mempher.ErrInvalidRole)
	}
	ep.OccurredAt = ep.OccurredAt.UTC()
	ep.IngestedAt = ep.IngestedAt.UTC()
	if len(binding) > 0 {
		ep.Binding = binding
	}
	return ep, nil
}

// validateAppend rejects a command the schema would reject anyway, so the error
// names the field instead of quoting a constraint.
func validateAppend(cmd mempher.AppendCommand) error {
	ep := cmd.Episode

	// The caller-facing rules live in one place; borrow them.
	probe := mempher.AppendRequest{
		Scope:   ep.Scope,
		Content: ep.Content,
		Role:    ep.Role,
		Actor:   ep.Actor,
		Source:  ep.Source,
		Binding: ep.Binding,
	}
	if err := probe.Validate(); err != nil {
		return fmt.Errorf("episode: %w", err)
	}
	if ep.OccurredAt.IsZero() || ep.IngestedAt.IsZero() {
		return fmt.Errorf(
			"OccurredAt and IngestedAt must be resolved by the caller, "+
				"because a Store never reads a clock: %w", mempher.ErrInvalidConfig)
	}

	kinds := make(map[mempher.JobKind]struct{}, len(cmd.Jobs))
	for i, job := range cmd.Jobs {
		switch {
		case !job.Kind.Valid():
			return fmt.Errorf("job %d kind %q: %w",
				i, job.Kind, mempher.ErrInvalidJobKind)
		case len(job.Episodes) > 0:
			return fmt.Errorf(
				"job %d already names episodes, but the Store fills "+
					"that in with the id it mints: %w", i, mempher.ErrInvalidConfig)
		case job.MaxAttempts < 0:
			return fmt.Errorf("job %d MaxAttempts is %d: %w",
				i, job.MaxAttempts, mempher.ErrInvalidConfig)
		}
		if _, dup := kinds[job.Kind]; dup {
			return fmt.Errorf(
				"two %q jobs for one episode would collapse into one: %w",
				job.Kind, mempher.ErrInvalidConfig)
		}
		kinds[job.Kind] = struct{}{}
	}
	return nil
}

// wrapAppendError turns the constraint violations that mean something specific
// into the matching sentinel.
func wrapAppendError(err error, ep mempher.NewEpisode) error {
	switch {
	case isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema):
		return fmt.Errorf("mempher/postgres: append: run Migrate first: %w", ErrSchemaNotReady)
	case isCode(err, pgerrcodeCheckViolation):
		return fmt.Errorf("mempher/postgres: append to scope %q: the schema rejected the row: %w",
			ep.Scope, err)
	default:
		return fmt.Errorf("mempher/postgres: append to scope %q: %w", ep.Scope, err)
	}
}

// deref reads a nullable scan target, treating NULL as the zero value.
func deref[T any](p *T) T {
	if p == nil {
		var zero T
		return zero
	}
	return *p
}
