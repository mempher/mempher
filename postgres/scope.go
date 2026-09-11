// The catalogue of L0 partitions.

package postgres

import (
	"context"
	"fmt"
	"strings"

	"github.com/mempher/mempher/ops"
)

// DefaultScopesLimit is how many scopes [Store.Scopes] returns when a query does
// not say.
const DefaultScopesLimit = 100

// scopeColumns is the read shape of a scope, in the order scanScope expects.
const scopeColumns = `id, last_seq, created_at`

// Scopes lists the partitions of L0 in id order.
//
// The rows come from mempher.scopes rather than from a DISTINCT over the
// episodes, because that table already holds each scope's last_seq: the length
// of a log is read, never counted.
func (s *Store) Scopes(
	ctx context.Context,
	q ops.ScopeQuery,
) ([]ops.Scope, error) {
	if err := q.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: scopes: %w", err)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultScopesLimit
	}

	b := &queryBuilder{}
	var sql strings.Builder
	sql.WriteString(`SELECT ` + scopeColumns + ` FROM mempher.scopes WHERE true`)
	if q.After != "" {
		fmt.Fprintf(&sql, "\n  AND id > %s", b.arg(string(q.After)))
	}
	if q.Prefix != "" {
		// starts_with rather than LIKE, so a scope id containing % or _ is
		// matched as itself instead of as a pattern. It cannot seek the
		// primary key, which the query type says plainly.
		fmt.Fprintf(&sql, "\n  AND starts_with(id, %s)", b.arg(string(q.Prefix)))
	}
	fmt.Fprintf(&sql, "\nORDER BY id\nLIMIT %s", b.arg(limit))

	rows, err := s.pool.Query(ctx, sql.String(), b.args...)
	if err != nil {
		if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
			return nil, fmt.Errorf("mempher/postgres: scopes: run Migrate first: %w",
				ErrSchemaNotReady)
		}
		return nil, fmt.Errorf("mempher/postgres: scopes: %w", err)
	}
	defer rows.Close()

	out := make([]ops.Scope, 0, min(limit, 128))
	for rows.Next() {
		scope, err := scanScope(rows)
		if err != nil {
			return nil, fmt.Errorf("mempher/postgres: scopes: %w", err)
		}
		out = append(out, scope)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: scopes: %w", err)
	}
	return out, nil
}

// scanScope reads one scope in [scopeColumns] order.
func scanScope(row scanner) (ops.Scope, error) {
	var scope ops.Scope
	if err := row.Scan(&scope.ID, &scope.LastSeq, &scope.CreatedAt); err != nil {
		return ops.Scope{}, fmt.Errorf("scan scope: %w", err)
	}
	scope.CreatedAt = scope.CreatedAt.UTC()
	return scope, nil
}
