// Erasure: the one way content leaves L0.

package postgres

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/mempher/mempher"
)

// erasureDeclaration is how this transaction tells the append-only trigger that
// it is the exception the trigger allows.
//
// LOCAL is the whole of its safety: it cannot outlive the transaction that set
// it, and it cannot reach a session that did not. A connection returned to the
// pool carries nothing.
const erasureDeclaration = `SET LOCAL mempher.erasing = 'on'`

// erasureStep is one table's share of an erasure. Two statements per table
// rather than one with an OR, exactly as the pending queries are split, so that
// each can use its index: a whole-scope erasure is a scope lookup, and a
// narrowed one adds a containment test the scope filter has already reduced.
type erasureStep struct {
	// what names the table in an error message.
	what string
	// count is where in the result this step's tally lands.
	count func(*mempher.ForgetResult) *int
	// whole erases everything the scope holds; some erases only the rows
	// belonging to the named episodes.
	whole, some string
}

// erasureSteps is the order an erasure runs in, and the order matters twice
// over.
//
// Foreign keys require it: an encoding and an extraction marker both point at an
// episode, so both go first. And recoverability requires it: every projection is
// removed before the episodes it came from, so an erasure that fails part-way
// has taken derived data and left its source, which re-running repairs. The
// reverse would leave a claim about someone whose episodes are gone, and nothing
// to notice it by.
//
// Facts and jobs name their episodes in an array, which PostgreSQL cannot put a
// foreign key behind. They are removed by containment here instead, which is why
// erasure has to be one supervised operation rather than a DELETE anyone can
// write. A job losing one erased episode is dropped whole, even if it named
// others that still need doing: what it named is no longer what it would find,
// and [Worker.Backfill] enqueues the rest again.
var erasureSteps = []erasureStep{
	{
		what:  "facts",
		count: func(r *mempher.ForgetResult) *int { return &r.Facts },
		whole: `DELETE FROM mempher.facts WHERE scope_id = $1`,
		some:  `DELETE FROM mempher.facts WHERE scope_id = $1 AND episode_ids && $2`,
	},
	{
		what:  "extraction markers",
		count: func(r *mempher.ForgetResult) *int { return &r.Extractions },
		whole: `DELETE FROM mempher.fact_extractions WHERE scope_id = $1`,
		some:  `DELETE FROM mempher.fact_extractions WHERE scope_id = $1 AND episode_id = ANY($2)`,
	},
	{
		what:  "encodings",
		count: func(r *mempher.ForgetResult) *int { return &r.Encodings },
		whole: `DELETE FROM mempher.episode_encodings WHERE scope_id = $1`,
		some:  `DELETE FROM mempher.episode_encodings WHERE scope_id = $1 AND episode_id = ANY($2)`,
	},
	{
		what:  "jobs",
		count: func(r *mempher.ForgetResult) *int { return &r.Jobs },
		whole: `DELETE FROM mempher.jobs WHERE scope_id = $1`,
		some:  `DELETE FROM mempher.jobs WHERE scope_id = $1 AND episode_ids && $2`,
	},
	{
		what:  "episodes",
		count: func(r *mempher.ForgetResult) *int { return &r.Episodes },
		whole: `DELETE FROM mempher.episodes WHERE scope_id = $1`,
		some:  `DELETE FROM mempher.episodes WHERE scope_id = $1 AND id = ANY($2)`,
	},
}

// Forget erases episodes and every row this store holds that derives from them.
//
// The whole erasure is one transaction, because a half-erased scope is not a
// smaller erasure but a failed one. It declares itself to the append-only
// trigger, removes each projection before its source, and takes the scope row
// last when the whole scope is going.
//
// An episode named from the wrong scope is not erased and is not an error, for
// the same reason reading one is [mempher.ErrNotFound] rather than a permission
// failure: a Store never crosses a scope. What actually went is in the result.
func (s *Store) Forget(
	ctx context.Context,
	req mempher.ForgetRequest,
) (mempher.ForgetResult, error) {
	if err := req.Validate(); err != nil {
		return mempher.ForgetResult{}, fmt.Errorf("mempher/postgres: forget: %w", err)
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return mempher.ForgetResult{}, fmt.Errorf("mempher/postgres: forget: begin: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }() // no-op once committed

	if _, err := tx.Exec(ctx, erasureDeclaration); err != nil {
		return mempher.ForgetResult{}, s.wrapForgetError("declare the erasure", req.Scope, err)
	}

	scope := string(req.Scope)
	wholeScope := len(req.Episodes) == 0
	episodes := make([]uuid.UUID, len(req.Episodes))
	for i, id := range req.Episodes {
		episodes[i] = uuid.UUID(id)
	}

	out := mempher.ForgetResult{Scope: req.Scope}
	for _, step := range erasureSteps {
		sql, args := step.whole, []any{scope}
		if !wholeScope {
			sql, args = step.some, []any{scope, episodes}
		}
		tag, err := tx.Exec(ctx, sql, args...)
		if err != nil {
			return mempher.ForgetResult{}, s.wrapForgetError(step.what, req.Scope, err)
		}
		*step.count(&out) = int(tag.RowsAffected())
	}

	// The catalogue row goes only when the partition itself is going. Nothing
	// references it by then, and appending to the same id afterwards begins a
	// new log at Seq 1.
	if wholeScope {
		tag, err := tx.Exec(ctx, `DELETE FROM mempher.scopes WHERE id = $1`, scope)
		if err != nil {
			return mempher.ForgetResult{}, s.wrapForgetError("the scope", req.Scope, err)
		}
		out.ScopeRemoved = tag.RowsAffected() > 0
	}

	if err := tx.Commit(ctx); err != nil {
		return mempher.ForgetResult{}, fmt.Errorf("mempher/postgres: forget scope %q: commit: %w",
			req.Scope, err)
	}
	return out, nil
}

// wrapForgetError names which part of the erasure failed, because "forget
// failed" does not say whether anything went.
func (s *Store) wrapForgetError(what string, scope mempher.ScopeID, err error) error {
	if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
		return fmt.Errorf("mempher/postgres: forget: run Migrate first: %w", ErrSchemaNotReady)
	}
	return fmt.Errorf("mempher/postgres: forget %s in scope %q: %w", what, scope, err)
}
