// Input validation for the operational requests.

package ops

import (
	"fmt"

	"github.com/mempher/mempher"
)

// Validate reports whether the query can list jobs. A [Queue] calls it, so
// calling it yourself is only useful to reject input earlier.
func (q JobQuery) Validate() error {
	if q.Scope != "" {
		if err := q.Scope.Validate(); err != nil {
			return err
		}
	}
	for _, kind := range q.Kinds {
		if !kind.Valid() {
			return fmt.Errorf("mempher: job kind filter %q: %w", string(kind), mempher.ErrInvalidJobKind)
		}
	}
	for _, state := range q.States {
		if !state.Valid() {
			return fmt.Errorf("mempher: job state filter %q: %w", string(state), mempher.ErrInvalidJobState)
		}
	}
	if q.Limit < 0 {
		return fmt.Errorf("mempher: job query limit is %d: %w", q.Limit, mempher.ErrInvalidConfig)
	}
	return nil
}

// Validate reports whether the request can purge. It refuses to delete work that
// is still owed, which is the whole reason the states are checked rather than
// passed to the database as written.
func (r PurgeRequest) Validate() error {
	if r.Before.IsZero() {
		return fmt.Errorf("mempher: purge has no horizon: %w", mempher.ErrInvalidConfig)
	}
	for _, state := range r.States {
		switch state {
		case mempher.JobStateDone, mempher.JobStateDead:
		case mempher.JobStatePending, mempher.JobStateRunning:
			return fmt.Errorf(
				"mempher: purge names %s jobs, which are still owed: %w",
				state, mempher.ErrInvalidJobState)
		default:
			return fmt.Errorf("mempher: purge state %q: %w", string(state), mempher.ErrInvalidJobState)
		}
	}
	if r.Limit < 0 {
		return fmt.Errorf("mempher: purge limit is %d: %w", r.Limit, mempher.ErrInvalidConfig)
	}
	return nil
}

// Validate reports whether the query can list scopes.
func (q ScopeQuery) Validate() error {
	// Both are optional, and both are scope ids when given: a prefix that
	// could never match any scope is a typo worth reporting here rather than
	// as an empty page.
	if q.Prefix != "" {
		if err := q.Prefix.Validate(); err != nil {
			return err
		}
	}
	if q.After != "" {
		if err := q.After.Validate(); err != nil {
			return err
		}
	}
	if q.Limit < 0 {
		return fmt.Errorf("mempher: scope query limit is %d: %w", q.Limit, mempher.ErrInvalidConfig)
	}
	return nil
}

// Validate reports whether the request can be backfilled. [Worker.Backfill]
// calls it, so calling it yourself is only useful to reject input earlier.
func (r BackfillRequest) Validate() error {
	if r.Scope != "" {
		if err := r.Scope.Validate(); err != nil {
			return err
		}
	}
	for _, kind := range r.Kinds {
		if !kind.Valid() {
			return fmt.Errorf("mempher: backfill kind %q: %w", string(kind), mempher.ErrInvalidJobKind)
		}
	}
	switch {
	case r.AfterSeq < 0:
		return fmt.Errorf("mempher: backfill AfterSeq is %d: %w", r.AfterSeq, mempher.ErrInvalidConfig)
	case r.AfterSeq > 0 && r.Scope == "":
		return fmt.Errorf(
			"mempher: backfill AfterSeq needs a Scope, because seq is only ordered "+
				"within one: %w", mempher.ErrInvalidConfig)
	case r.BatchSize < 0:
		return fmt.Errorf("mempher: backfill batch size is %d: %w", r.BatchSize, mempher.ErrInvalidConfig)
	case r.Limit < 0:
		return fmt.Errorf("mempher: backfill limit is %d: %w", r.Limit, mempher.ErrInvalidConfig)
	}
	return nil
}
