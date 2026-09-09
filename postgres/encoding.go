// The vector projection: writing encodings, and finding episodes that lack one.

package postgres

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/google/uuid"
	"github.com/mempher/mempher"
)

// DefaultPendingEncodingsLimit is how many ids [Store.PendingEncodings] returns
// when a query does not say.
const DefaultPendingEncodingsLimit = 100

// putEncodingSQL takes scope_id and ingested_at from the episode row rather than
// from the caller. Those two columns are denormalised so the vector channel can
// filter by scope and as-of without joining, and deriving them here is what
// guarantees they cannot disagree with L0. %s is the qualified vector type, so
// the cast does not depend on the pool's search_path.
//
// The upsert is what makes an encode job safe to retry, and what makes
// re-embedding with the same model idempotent.
const putEncodingSQL = `
INSERT INTO mempher.episode_encodings
    (episode_id, model, scope_id, ingested_at, content_vec, surprise, encoded_at)
SELECT e.id, $2, e.scope_id, e.ingested_at, $3::%s, $4, $5
FROM mempher.episodes e
WHERE e.id = $1
ON CONFLICT (episode_id, model) DO UPDATE
SET content_vec = excluded.content_vec,
    surprise    = excluded.surprise,
    encoded_at  = excluded.encoded_at`

// PutEncoding writes one encoding, replacing any existing one for the same
// episode and model.
func (s *Store) PutEncoding(ctx context.Context, enc mempher.Encoding) error {
	if err := s.validateEncoding(enc); err != nil {
		return err
	}

	tag, err := s.pool.Exec(ctx, fmt.Sprintf(putEncodingSQL, s.vectorType),
		uuid.UUID(enc.Episode), string(enc.Model), formatVector(enc.Vector),
		enc.Surprise, enc.EncodedAt)
	switch {
	case isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema):
		return fmt.Errorf("mempher/postgres: put encoding: run Migrate first: %w", ErrSchemaNotReady)
	case err != nil:
		return fmt.Errorf("mempher/postgres: put encoding for episode %s: %w", enc.Episode, err)
	}

	// No episode matched, so the SELECT produced no row to insert.
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("mempher/postgres: put encoding: episode %s: %w",
			enc.Episode, mempher.ErrNotFound)
	}
	return nil
}

// validateEncoding rejects what pgvector or a CHECK would reject anyway, with an
// error that names the problem.
func (s *Store) validateEncoding(enc mempher.Encoding) error {
	switch {
	case enc.Episode.IsZero():
		return fmt.Errorf("mempher/postgres: put encoding: episode id is unset: %w",
			mempher.ErrInvalidConfig)
	case enc.Model == "":
		return fmt.Errorf("mempher/postgres: put encoding: model is empty: %w",
			mempher.ErrInvalidConfig)
	case enc.EncodedAt.IsZero():
		return fmt.Errorf(
			"mempher/postgres: put encoding: EncodedAt must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	case enc.Surprise < 0 || enc.Surprise > 1:
		return fmt.Errorf("mempher/postgres: put encoding: surprise is %v, must be in [0,1]: %w",
			enc.Surprise, mempher.ErrInvalidConfig)
	}

	if got := enc.Vector.Dimensions(); got != s.vectorDimensions {
		return fmt.Errorf(
			"mempher/postgres: put encoding: vector has %d dimensions but the schema was migrated for %d: %w",
			got, s.vectorDimensions, mempher.ErrDimensionMismatch)
	}
	// pgvector rejects these, but as a parse error on a 20 KB literal, which
	// says nothing about which model produced the bad value.
	for i, f := range enc.Vector {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			return fmt.Errorf(
				"mempher/postgres: put encoding: vector element %d from model %q is %v: %w",
				i, enc.Model, f, mempher.ErrInvalidConfig)
		}
	}
	return nil
}

// pendingEncodingsSQL finds episodes with no encoding for a model. The anti-join
// runs against L0 and the encodings table alone, which is what makes the queue
// rebuildable: enqueue what this returns and any episode missed by a lost job, or
// left behind by a change of model, gets encoded.
const (
	pendingAllScopesSQL = `
SELECT e.id
FROM mempher.episodes e
WHERE NOT EXISTS (
    SELECT 1 FROM mempher.episode_encodings x
    WHERE x.episode_id = e.id AND x.model = $1
)
ORDER BY e.scope_id, e.seq
LIMIT $2`

	pendingOneScopeSQL = `
SELECT e.id
FROM mempher.episodes e
WHERE e.scope_id = $1
  AND e.seq > $2
  AND NOT EXISTS (
    SELECT 1 FROM mempher.episode_encodings x
    WHERE x.episode_id = e.id AND x.model = $3
)
ORDER BY e.seq
LIMIT $4`
)

// PendingEncodings returns ids of episodes with no encoding for a model, in Seq
// order.
func (s *Store) PendingEncodings(
	ctx context.Context,
	q mempher.PendingEncodings,
) ([]mempher.EpisodeID, error) {
	if q.Model == "" {
		return nil, fmt.Errorf("mempher/postgres: pending encodings: model is empty: %w",
			mempher.ErrInvalidConfig)
	}
	if q.Scope != "" {
		if err := q.Scope.Validate(); err != nil {
			return nil, fmt.Errorf("mempher/postgres: pending encodings: %w", err)
		}
	}
	switch {
	case q.AfterSeq < 0:
		return nil, fmt.Errorf("mempher/postgres: pending encodings: AfterSeq is %d: %w",
			q.AfterSeq, mempher.ErrInvalidConfig)
	case q.AfterSeq > 0 && q.Scope == "":
		return nil, fmt.Errorf(
			"mempher/postgres: pending encodings: AfterSeq needs a Scope, because seq is only "+
				"ordered within one: %w", mempher.ErrInvalidConfig)
	case q.Limit < 0:
		return nil, fmt.Errorf("mempher/postgres: pending encodings: Limit is %d: %w",
			q.Limit, mempher.ErrInvalidConfig)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultPendingEncodingsLimit
	}

	// Two statements rather than one with an OR, so each can use its index.
	sql, args := pendingAllScopesSQL, []any{string(q.Model), limit}
	if q.Scope != "" {
		sql = pendingOneScopeSQL
		args = []any{string(q.Scope), q.AfterSeq, string(q.Model), limit}
	}

	rows, err := s.pool.Query(ctx, sql, args...)
	if err != nil {
		if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
			return nil, fmt.Errorf("mempher/postgres: pending encodings: run Migrate first: %w",
				ErrSchemaNotReady)
		}
		return nil, fmt.Errorf("mempher/postgres: pending encodings for model %q: %w", q.Model, err)
	}
	defer rows.Close()

	out := make([]mempher.EpisodeID, 0, min(limit, 128))
	for rows.Next() {
		var id uuid.UUID
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("mempher/postgres: pending encodings: scan: %w", err)
		}
		out = append(out, mempher.EpisodeID(id))
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: pending encodings for model %q: %w", q.Model, err)
	}
	return out, nil
}

// formatVector renders a vector in pgvector's text input form.
//
// Text rather than the binary protocol, because binary would need the vector type
// registered on every pooled connection and this library does not own the pool.
// The cost is a few extra bytes per write and nothing in correctness.
func formatVector(v mempher.Vector) string {
	var b strings.Builder
	b.Grow(len(v)*10 + 2)
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		// 32-bit shortest round-trip: the value read back equals the value
		// written, without printing digits float32 does not hold.
		b.WriteString(strconv.FormatFloat(float64(f), 'g', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}
