// L1: applying an extraction, reading facts back, and finding episodes no
// extractor has read.

package postgres

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/mempher/mempher"
)

// Defaults used when the corresponding query field is left zero.
const (
	// DefaultFactsLimit is how many facts [Store.Facts] returns when a query
	// does not say.
	DefaultFactsLimit = 200
	// DefaultPendingExtractionsLimit is how many ids
	// [Store.PendingExtractions] returns when a query does not say.
	DefaultPendingExtractionsLimit = 100
)

// pgerrcodeExclusionViolation is what a WITHOUT OVERLAPS key raises when two
// windows for one claim collide.
const pgerrcodeExclusionViolation = "23P01"

// factColumns is the read shape of a fact, in the order scanFact expects.
const factColumns = `id, scope_id, extractor, subject, predicate, object,
	statement, valid, confidence, episode_ids, asserted_at, updated_at`

// markExtractionSQL records that an extractor read these episodes, and reports
// both how many of them exist in the scope and how many markers were new.
//
// The two counts are what make an extraction idempotent without a second round
// trip. A short `found` means an episode was named that this scope does not
// have. A zero `marked` means every one of these episodes had already been read
// by this extractor, so the extraction it produced is already in the table and
// applying it again would be at best a no-op and at worst a conflict against
// itself.
//
// scope_id and ingested_at are taken from the episode row for the same reason as
// in episode_encodings: it makes the pending anti-join a single-table scan, and
// they cannot drift from an immutable row.
const markExtractionSQL = `
WITH found AS (
    SELECT e.id, e.scope_id, e.ingested_at
    FROM mempher.episodes e
    WHERE e.id = ANY($1::uuid[]) AND e.scope_id = $2
), marked AS (
    INSERT INTO mempher.fact_extractions
        (episode_id, extractor, scope_id, ingested_at, fact_count, extracted_at)
    SELECT found.id, $3, found.scope_id, found.ingested_at, $4, $5
    FROM found
    ON CONFLICT (episode_id, extractor) DO NOTHING
    RETURNING episode_id
)
SELECT (SELECT count(*) FROM found), (SELECT count(*) FROM marked)`

// applyAssertionsSQL upserts a batch of claims.
//
// It is written as UPDATE-then-INSERT rather than ON CONFLICT because the
// target is a temporal primary key, which PostgreSQL implements as a GiST
// exclusion constraint, and ON CONFLICT cannot name one. The UPDATE matches on
// the full identity including an exact window, so re-asserting the same claim
// refreshes it and asserting the same claim over a different overlapping window
// falls through to the INSERT and is rejected by the constraint. That is the
// intended split: a re-run is free, a contradiction is loud.
//
// Provenance accumulates rather than being replaced. A claim two episodes
// support is evidenced by both, and an update that overwrote episode_ids would
// quietly narrow the trail back to L0.
const applyAssertionsSQL = `
WITH input AS (
    SELECT * FROM unnest(
        $3::text[], $4::text[], $5::text[], $6::text[],
        $7::timestamptz[], $8::timestamptz[], $9::real[]
    ) AS a(subject, predicate, object, statement, valid_from, valid_to, confidence)
), updated AS (
    UPDATE mempher.facts f
    SET statement   = i.statement,
        confidence  = i.confidence,
        episode_ids = (
            SELECT array_agg(DISTINCT e ORDER BY e)
            FROM unnest(f.episode_ids || $10::uuid[]) AS e
        ),
        updated_at  = $11
    FROM input i
    WHERE f.scope_id  = $1
      AND f.extractor = $2
      AND f.subject   = i.subject
      AND f.predicate = i.predicate
      AND f.object    = i.object
      AND f.valid     = tstzrange(i.valid_from, i.valid_to, '[)')
    RETURNING f.subject, f.predicate, f.object
)
INSERT INTO mempher.facts
    (scope_id, extractor, subject, predicate, object, statement, valid,
     confidence, episode_ids, asserted_at, updated_at)
SELECT $1, $2, i.subject, i.predicate, i.object, i.statement,
       tstzrange(i.valid_from, i.valid_to, '[)'), i.confidence, $10::uuid[], $11, $11
FROM input i
WHERE NOT EXISTS (
    SELECT 1 FROM updated u
    WHERE u.subject = i.subject AND u.predicate = i.predicate AND u.object = i.object
)`

// retractSQL closes a fact's window at an instant.
//
// The guard is what makes a retraction idempotent and keeps it from running
// backwards: a window already closed at or before that instant is left alone
// rather than reopened to a later one, and Go reads the row back to tell that
// case apart from a fact that is simply not there.
const retractSQL = `
UPDATE mempher.facts
SET valid      = tstzrange(lower(valid), $3, '[)'),
    updated_at = $4
WHERE id = $1
  AND scope_id = $2
  AND lower(valid) < $3
  AND (upper_inf(valid) OR upper(valid) > $3)`

// factsOfExtractionSQL reads back everything one extraction's episodes support.
// It answers both paths through ApplyExtraction: the one that just wrote, and
// the one that found the work already done.
const factsOfExtractionSQL = `
SELECT ` + factColumns + `
FROM mempher.facts
WHERE scope_id = $1 AND extractor = $2 AND episode_ids && $3::uuid[]
ORDER BY subject, predicate, object, lower(valid)`

// ApplyExtraction records one extraction atomically: the assertions, the
// retractions, and the markers saying those episodes were read.
func (s *Store) ApplyExtraction(
	ctx context.Context,
	cmd mempher.ExtractCommand,
) ([]mempher.Fact, error) {
	if err := cmd.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: apply extraction: %w", err)
	}
	assertions, err := dedupeAssertions(cmd.Assert)
	if err != nil {
		return nil, err
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, fmt.Errorf("mempher/postgres: apply extraction: begin: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	episodes := make([]uuid.UUID, len(cmd.Episodes))
	for i, id := range cmd.Episodes {
		episodes[i] = uuid.UUID(id)
	}

	var found, marked int
	err = tx.QueryRow(ctx, markExtractionSQL,
		episodes, string(cmd.Scope), string(cmd.Extractor), len(assertions), cmd.At,
	).Scan(&found, &marked)
	if err != nil {
		return nil, s.wrapFactError("apply extraction", cmd.Scope, err)
	}
	if found < len(cmd.Episodes) {
		return nil, fmt.Errorf(
			"mempher/postgres: apply extraction: %d of %d episodes are not in scope %q: %w",
			len(cmd.Episodes)-found, len(cmd.Episodes), cmd.Scope, mempher.ErrNotFound)
	}

	// A zero marked count means every episode had already been read by this
	// extractor, so the work is done and replaying it would at best change
	// nothing. Skipping is what makes a reclaimed job free, which is what
	// at-least-once leasing needs from every job body.
	if marked > 0 {
		if err := applyRetractions(ctx, tx, cmd); err != nil {
			return nil, err
		}
		if err := s.applyAssertions(ctx, tx, cmd, assertions, episodes); err != nil {
			return nil, err
		}
	}

	facts, err := scanFacts(ctx, tx, factsOfExtractionSQL,
		string(cmd.Scope), string(cmd.Extractor), episodes)
	if err != nil {
		return nil, s.wrapFactError("apply extraction", cmd.Scope, err)
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, fmt.Errorf("mempher/postgres: apply extraction: commit: %w", err)
	}
	return facts, nil
}

// applyAssertions writes the batch of claims, translating the constraint that
// catches a self-contradicting extraction into its sentinel.
func (s *Store) applyAssertions(
	ctx context.Context,
	tx pgx.Tx,
	cmd mempher.ExtractCommand,
	assertions []mempher.Assertion,
	episodes []uuid.UUID,
) error {
	if len(assertions) == 0 {
		return nil
	}

	n := len(assertions)
	subjects := make([]string, n)
	predicates := make([]string, n)
	objects := make([]string, n)
	statements := make([]string, n)
	from := make([]time.Time, n)
	to := make([]*time.Time, n)
	confidence := make([]float32, n)
	for i, a := range assertions {
		subjects[i] = string(a.Subject)
		predicates[i] = string(a.Predicate)
		objects[i] = a.Object
		statements[i] = a.Statement
		from[i] = a.Valid.From
		// A NULL upper bound is an unbounded one: the claim is still true.
		if !a.Valid.IsOpen() {
			upper := a.Valid.To
			to[i] = &upper
		}
		confidence[i] = a.Confidence
	}

	_, err := tx.Exec(ctx, applyAssertionsSQL,
		string(cmd.Scope), string(cmd.Extractor),
		subjects, predicates, objects, statements, from, to, confidence,
		episodes, cmd.At)
	if err != nil {
		return s.wrapFactError("apply extraction", cmd.Scope, err)
	}
	return nil
}

// applyRetractions closes the windows an extraction contradicted.
//
// Each is checked rather than fired and forgotten, because a retraction naming a
// fact that is not there, or an instant before the fact began, means the
// extractor is working from a view of the scope that no longer holds -- and
// silently dropping it would leave two claims standing that cannot both be true.
func applyRetractions(ctx context.Context, tx pgx.Tx, cmd mempher.ExtractCommand) error {
	for _, r := range cmd.Retract {
		tag, err := tx.Exec(ctx, retractSQL,
			uuid.UUID(r.Fact), string(cmd.Scope), r.At, cmd.At)
		if err != nil {
			return fmt.Errorf("mempher/postgres: apply extraction: retract fact %s: %w",
				r.Fact, err)
		}
		if tag.RowsAffected() > 0 {
			continue
		}
		if err := explainRetraction(ctx, tx, cmd.Scope, r); err != nil {
			return err
		}
	}
	return nil
}

// explainRetraction says why a retraction changed nothing: the fact is absent,
// the instant precedes its start, or it was already closed by then. Only the
// last of those is acceptable, and only because a retried job must be free.
func explainRetraction(
	ctx context.Context,
	tx pgx.Tx,
	scope mempher.ScopeID,
	r mempher.Retraction,
) error {
	var lower time.Time
	var upper *time.Time
	err := tx.QueryRow(ctx,
		`SELECT lower(valid), upper(valid) FROM mempher.facts WHERE id = $1 AND scope_id = $2`,
		uuid.UUID(r.Fact), string(scope)).Scan(&lower, &upper)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return fmt.Errorf("mempher/postgres: apply extraction: retract fact %s in scope %q: %w",
			r.Fact, scope, mempher.ErrNotFound)
	case err != nil:
		return fmt.Errorf("mempher/postgres: apply extraction: read fact %s: %w", r.Fact, err)
	}
	if !r.At.After(lower) {
		return fmt.Errorf(
			"mempher/postgres: apply extraction: fact %s stops being true at %s, "+
				"but only became true at %s: %w",
			r.Fact, r.At.UTC().Format(time.RFC3339), lower.UTC().Format(time.RFC3339),
			mempher.ErrInvalidValidity)
	}
	// Already closed at or before this instant: the retraction has been applied
	// before, which is exactly what a reclaimed job replays.
	return nil
}

// dedupeAssertions drops exact repeats and rejects a batch that contradicts
// itself over the same claim.
//
// Both are the extractor's mistakes, and both would otherwise arrive as one
// opaque constraint violation from a statement that inserted several rows at
// once. The same claim twice is a model repeating itself and is simply dropped;
// the same claim over two different windows is a model disagreeing with itself
// inside one answer, which nothing downstream can resolve.
func dedupeAssertions(in []mempher.Assertion) ([]mempher.Assertion, error) {
	type identity struct {
		subject   mempher.Subject
		predicate mempher.Predicate
		object    string
	}

	out := make([]mempher.Assertion, 0, len(in))
	windows := make(map[identity][]mempher.Validity, len(in))
	for _, a := range in {
		key := identity{a.Subject, a.Predicate, a.Object}
		if slices.ContainsFunc(windows[key], func(v mempher.Validity) bool {
			return v == a.Valid
		}) {
			continue
		}
		for _, seen := range windows[key] {
			if seen.Overlaps(a.Valid) {
				return nil, fmt.Errorf(
					"mempher/postgres: apply extraction: %q %q %q is asserted over two "+
						"overlapping windows in one extraction: %w",
					a.Subject, a.Predicate, a.Object, mempher.ErrFactConflict)
			}
		}
		windows[key] = append(windows[key], a.Valid)
		out = append(out, a)
	}
	return out, nil
}

// Facts returns the facts matching q.
func (s *Store) Facts(ctx context.Context, q mempher.FactQuery) ([]mempher.Fact, error) {
	if err := q.Validate(); err != nil {
		return nil, fmt.Errorf("mempher/postgres: facts: %w", err)
	}
	if q.AsOf.IsZero() {
		return nil, fmt.Errorf(
			"mempher/postgres: facts: AsOf must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	}
	if q.At.IsZero() {
		return nil, fmt.Errorf(
			"mempher/postgres: facts: At must be resolved by the caller: %w",
			mempher.ErrInvalidConfig)
	}

	b := &queryBuilder{}
	scope := b.arg(string(q.Scope))
	extractor := b.arg(string(q.Extractor))

	var sql strings.Builder
	var rank string
	if strings.TrimSpace(q.Text) != "" {
		// Ranked, but never filtered: relevance decides which facts survive the
		// limit, it does not decide which are true. A fact the query does not
		// mention is still a fact about the scope.
		rank = fmt.Sprintf("ts_rank_cd(f.statement_tsv, %s)",
			anyLexemeQuery(b.arg(s.textSearchConfig), b.arg(q.Text)))
	}
	fmt.Fprintf(&sql, "SELECT %s\nFROM mempher.facts f\nWHERE f.scope_id = %s AND f.extractor = %s",
		qualify(factColumns, "f."), scope, extractor)

	fmt.Fprintf(&sql, "\n  AND f.asserted_at <= %s", b.arg(q.AsOf))
	if q.IncludeClosed {
		// Everything the scope had begun to believe by then, whether or not it
		// still holds: the "what was true then" question.
		fmt.Fprintf(&sql, "\n  AND lower(f.valid) <= %s", b.arg(q.At))
	} else {
		fmt.Fprintf(&sql, "\n  AND f.valid @> %s::timestamptz", b.arg(q.At))
	}
	if len(q.Subjects) > 0 {
		subjects := make([]string, len(q.Subjects))
		for i, subject := range q.Subjects {
			subjects[i] = string(subject)
		}
		fmt.Fprintf(&sql, "\n  AND f.subject = ANY(%s)", b.arg(subjects))
	}
	if len(q.Predicates) > 0 {
		predicates := make([]string, len(q.Predicates))
		for i, predicate := range q.Predicates {
			predicates[i] = string(predicate)
		}
		fmt.Fprintf(&sql, "\n  AND f.predicate = ANY(%s)", b.arg(predicates))
	}
	if q.MinConfidence > 0 {
		fmt.Fprintf(&sql, "\n  AND f.confidence >= %s", b.arg(q.MinConfidence))
	}

	limit := q.Limit
	if limit == 0 {
		limit = DefaultFactsLimit
	}
	// Subject and predicate last in either ordering, so the result is fully
	// determined: two facts of equal rank come back in the same order every run.
	if rank != "" {
		fmt.Fprintf(&sql, "\nORDER BY %s DESC, f.subject, f.predicate, f.object", rank)
	} else {
		sql.WriteString("\nORDER BY f.subject, f.predicate, f.object")
	}
	fmt.Fprintf(&sql, "\nLIMIT %s", b.arg(limit))

	facts, err := scanFacts(ctx, s.pool, sql.String(), b.args...)
	if err != nil {
		return nil, s.wrapFactError("facts", q.Scope, err)
	}
	return facts, nil
}

// anyLexemeQuery builds a tsquery that matches any one of a text's lexemes.
//
// The episode channel parses its query with websearch_to_tsquery, which puts AND
// between terms, and that is right there: it decides which episodes match at
// all. Here the query only orders, so AND is the wrong operator entirely -- a
// two-word question would score every fact zero unless one statement happened to
// contain both words, and the ordering would silently collapse back to
// alphabetical. OR gives what ordering actually wants: the facts sharing the
// most with the question, first.
//
// The lexemes come from to_tsvector under the schema's own configuration, so
// they are stemmed exactly as the indexed column was, and quote_literal makes
// each one a literal rather than something the tsquery parser might read as an
// operator. Text that is nothing but stop words yields the empty tsquery, which
// ranks everything zero and leaves the ordering to the tie-break.
func anyLexemeQuery(config, text string) string {
	return fmt.Sprintf(
		`COALESCE((SELECT string_agg(quote_literal(lexeme), ' | ')`+
			` FROM unnest(to_tsvector(%s::regconfig, %s)))::tsquery, ''::tsquery)`,
		config, text)
}

// factByIDSQL reads one fact within a scope. The scope is part of the predicate,
// not checked afterwards, so a fact addressed from the wrong scope is absent
// rather than refused.
const factByIDSQL = `SELECT ` + factColumns + `
FROM mempher.facts WHERE id = $1 AND scope_id = $2`

// Fact returns one fact by id within a scope, or [mempher.ErrNotFound].
func (s *Store) Fact(
	ctx context.Context,
	scope mempher.ScopeID,
	id mempher.FactID,
) (mempher.Fact, error) {
	if err := scope.Validate(); err != nil {
		return mempher.Fact{}, fmt.Errorf("mempher/postgres: fact: %w", err)
	}
	if id.IsZero() {
		return mempher.Fact{}, fmt.Errorf("mempher/postgres: fact: id is unset: %w",
			mempher.ErrInvalidConfig)
	}

	facts, err := scanFacts(ctx, s.pool, factByIDSQL, uuid.UUID(id), string(scope))
	if err != nil {
		return mempher.Fact{}, s.wrapFactError("fact", scope, err)
	}
	if len(facts) == 0 {
		return mempher.Fact{}, fmt.Errorf("mempher/postgres: fact %s in scope %q: %w",
			id, scope, mempher.ErrNotFound)
	}
	return facts[0], nil
}

// pendingExtractionsSQL finds episodes no extraction marker covers. Like the
// encoding anti-join it runs against L0 and one projection table, which is what
// lets a lost job, or a change of extractor, be recovered by enqueueing what it
// returns.
const (
	pendingExtractAllScopesSQL = `
SELECT e.id
FROM mempher.episodes e
WHERE NOT EXISTS (
    SELECT 1 FROM mempher.fact_extractions x
    WHERE x.episode_id = e.id AND x.extractor = $1
)
ORDER BY e.scope_id, e.seq
LIMIT $2`

	pendingExtractOneScopeSQL = `
SELECT e.id
FROM mempher.episodes e
WHERE e.scope_id = $1
  AND e.seq > $2
  AND NOT EXISTS (
    SELECT 1 FROM mempher.fact_extractions x
    WHERE x.episode_id = e.id AND x.extractor = $3
)
ORDER BY e.seq
LIMIT $4`
)

// PendingExtractions returns ids of episodes no extraction marker covers, in Seq
// order.
func (s *Store) PendingExtractions(
	ctx context.Context,
	q mempher.PendingExtractions,
) ([]mempher.EpisodeID, error) {
	if q.Extractor == "" {
		return nil, fmt.Errorf("mempher/postgres: pending extractions: extractor is empty: %w",
			mempher.ErrInvalidConfig)
	}
	if q.Scope != "" {
		if err := q.Scope.Validate(); err != nil {
			return nil, fmt.Errorf("mempher/postgres: pending extractions: %w", err)
		}
	}
	switch {
	case q.AfterSeq < 0:
		return nil, fmt.Errorf("mempher/postgres: pending extractions: AfterSeq is %d: %w",
			q.AfterSeq, mempher.ErrInvalidConfig)
	case q.AfterSeq > 0 && q.Scope == "":
		return nil, fmt.Errorf(
			"mempher/postgres: pending extractions: AfterSeq needs a Scope, because seq is "+
				"only ordered within one: %w", mempher.ErrInvalidConfig)
	case q.Limit < 0:
		return nil, fmt.Errorf("mempher/postgres: pending extractions: Limit is %d: %w",
			q.Limit, mempher.ErrInvalidConfig)
	}
	limit := q.Limit
	if limit == 0 {
		limit = DefaultPendingExtractionsLimit
	}

	sql, args := pendingExtractAllScopesSQL, []any{string(q.Extractor), limit}
	if q.Scope != "" {
		sql = pendingExtractOneScopeSQL
		args = []any{string(q.Scope), q.AfterSeq, string(q.Extractor), limit}
	}

	rows, err := s.pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, s.wrapFactError("pending extractions", q.Scope, err)
	}
	defer rows.Close()

	out := make([]mempher.EpisodeID, 0, min(limit, 128))
	for rows.Next() {
		var id uuid.UUID
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("mempher/postgres: pending extractions: scan: %w", err)
		}
		out = append(out, mempher.EpisodeID(id))
	}
	if err := rows.Err(); err != nil {
		return nil, s.wrapFactError("pending extractions", q.Scope, err)
	}
	return out, nil
}

// querier is the part of pgx a fact read needs, so the same scan serves the pool
// and a transaction.
type querier interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

// scanFacts reads a fact result set.
func scanFacts(ctx context.Context, q querier, sql string, args ...any) ([]mempher.Fact, error) {
	rows, err := q.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]mempher.Fact, 0, 16)
	for rows.Next() {
		fact, err := scanFact(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, fact)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read facts: %w", err)
	}
	return out, nil
}

// scanFact reads one row in the order of [factColumns].
func scanFact(rows pgx.Rows) (mempher.Fact, error) {
	var (
		f         mempher.Fact
		id        [16]byte
		subject   string
		predicate string
		valid     pgtype.Range[pgtype.Timestamptz]
		episodes  []uuid.UUID
	)
	if err := rows.Scan(&id, &f.Scope, &f.Extractor, &subject, &predicate, &f.Object,
		&f.Statement, &valid, &f.Confidence, &episodes, &f.AssertedAt, &f.UpdatedAt); err != nil {
		return mempher.Fact{}, fmt.Errorf("scan fact: %w", err)
	}

	f.ID = mempher.FactID(id)
	f.Subject = mempher.Subject(subject)
	f.Predicate = mempher.Predicate(predicate)
	f.AssertedAt = f.AssertedAt.UTC()
	f.UpdatedAt = f.UpdatedAt.UTC()

	validity, err := toValidity(valid)
	if err != nil {
		return mempher.Fact{}, fmt.Errorf("fact %s: %w", f.ID, err)
	}
	f.Valid = validity

	if len(episodes) > 0 {
		f.Episodes = make([]mempher.EpisodeID, len(episodes))
		for i, e := range episodes {
			f.Episodes[i] = mempher.EpisodeID(e)
		}
	}
	return f, nil
}

// toValidity converts the stored range to the Go window.
//
// The schema constrains the range to be non-empty, lower-bounded and half-open,
// so anything else here means a row was written by something other than this
// library, and reading it as a plausible window would hide that.
func toValidity(r pgtype.Range[pgtype.Timestamptz]) (mempher.Validity, error) {
	if r.LowerType != pgtype.Inclusive || !r.Lower.Valid {
		return mempher.Validity{}, fmt.Errorf(
			"validity window is not lower-bounded and inclusive: %w", mempher.ErrInvalidValidity)
	}
	out := mempher.Validity{From: r.Lower.Time.UTC()}
	switch r.UpperType {
	case pgtype.Unbounded:
		// Still true, which the Go type spells as a zero To.
	case pgtype.Exclusive:
		if !r.Upper.Valid {
			return mempher.Validity{}, fmt.Errorf(
				"validity window has an exclusive but absent upper bound: %w",
				mempher.ErrInvalidValidity)
		}
		out.To = r.Upper.Time.UTC()
	default:
		return mempher.Validity{}, fmt.Errorf(
			"validity window is not half-open: %w", mempher.ErrInvalidValidity)
	}
	return out, nil
}

// qualify prefixes each column in a select list with a table alias.
func qualify(columns, prefix string) string {
	parts := strings.Split(columns, ",")
	for i, part := range parts {
		parts[i] = prefix + strings.TrimSpace(part)
	}
	return strings.Join(parts, ", ")
}

// wrapFactError turns the conditions L1 has sentinels for into them, and names
// the scope on everything else.
func (s *Store) wrapFactError(op string, scope mempher.ScopeID, err error) error {
	switch {
	case isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema):
		return fmt.Errorf("mempher/postgres: %s: run Migrate first: %w", op, ErrSchemaNotReady)
	case isCode(err, pgerrcodeExclusionViolation):
		return fmt.Errorf(
			"mempher/postgres: %s in scope %q: a claim is already believed over an "+
				"overlapping window; retract it before asserting a new one: %w",
			op, scope, mempher.ErrFactConflict)
	case isCode(err, pgerrcodeCheckViolation):
		return fmt.Errorf("mempher/postgres: %s in scope %q: %w: %w",
			op, scope, mempher.ErrInvalidFact, err)
	case isCode(err, pgerrcodeForeignKeyViolat):
		return fmt.Errorf("mempher/postgres: %s in scope %q: %w",
			op, scope, mempher.ErrNotFound)
	}
	return fmt.Errorf("mempher/postgres: %s in scope %q: %w", op, scope, err)
}

// The adapter serves every port in the library, so a signature drifting from an
// interface is a compile error here rather than a wiring error in New.
var (
	_ mempher.Store     = (*Store)(nil)
	_ mempher.Queue     = (*Store)(nil)
	_ mempher.FactStore = (*Store)(nil)
)
