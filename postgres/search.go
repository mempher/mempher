// The two retrieval channels. Each pushes its whole filter into SQL, because
// filtering after a LIMIT would drop matches ranked below an excluded row.

package postgres

import (
	"bytes"
	"cmp"
	"context"
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/mempher/mempher"
)

// episodeColumnsQualified is [episodeColumns] with the alias the search joins use.
const episodeColumnsQualified = `e.id, e.scope_id, e.seq, e.content, e.role, e.actor,
	e.source, e.occurred_at, e.ingested_at, e.binding`

// setIterativeScan is sent with every semantic search, in the same round trip.
//
// Without it a scope-filtered vector search is quietly wrong. pgvector's HNSW
// scan yields at most hnsw.ef_search candidates and the executor then applies
// the scope filter, so in a database holding many scopes a search can return
// fewer rows than asked, or none at all, while the scope's episodes sit in the
// index unvisited. strict_order makes the scan continue until the limit is
// genuinely satisfied, in distance order.
//
// SET LOCAL needs a transaction, and a pgx batch runs its statements inside one
// implicit transaction, so this costs no extra round trip and touches no
// connection state the caller might observe afterwards.
const setIterativeScan = `SET LOCAL hnsw.iterative_scan = strict_order`

// SearchSemantic returns the nearest episodes by cosine distance over one model's
// encodings, best first.
func (s *Store) SearchSemantic(
	ctx context.Context,
	q mempher.SemanticQuery,
) ([]mempher.Candidate, error) {
	if err := s.validateSemantic(q); err != nil {
		return nil, err
	}

	b := &queryBuilder{}
	model := b.arg(string(q.Model))
	scope := b.arg(string(q.Scope))
	asOf := b.arg(q.AsOf)
	vector := b.arg(formatVector(q.Vector)) + "::" + s.vectorType

	// The score is cosine similarity, which reads the right way round; the
	// ordering uses the distance operator so the index can serve it.
	var sql strings.Builder
	fmt.Fprintf(&sql, `SELECT %s, 1 - (enc.content_vec <=> %s) AS score
FROM mempher.episode_encodings enc
JOIN mempher.episodes e ON e.id = enc.episode_id
WHERE enc.model = %s AND enc.scope_id = %s AND enc.ingested_at <= %s`,
		episodeColumnsQualified, vector, model, scope, asOf)
	b.appendEpisodeFilters(&sql, q.ChannelQuery)
	fmt.Fprintf(&sql, "\nORDER BY enc.content_vec <=> %s\nLIMIT %s",
		vector, b.arg(candidateLimit(q.Limit)))

	batch := &pgx.Batch{}
	batch.Queue(setIterativeScan)
	batch.Queue(sql.String(), b.args...)

	results := s.pool.SendBatch(ctx, batch)
	candidates, scanErr := func() ([]mempher.Candidate, error) {
		if _, err := results.Exec(); err != nil {
			return nil, fmt.Errorf("enable iterative index scans: %w", err)
		}
		rows, err := results.Query()
		if err != nil {
			return nil, fmt.Errorf("query: %w", err)
		}
		return scanCandidates(rows, mempher.ChannelSemantic, q.Limit)
	}()
	closeErr := results.Close()

	switch {
	case scanErr != nil:
		return nil, s.wrapSearchError("semantic", q.Scope, scanErr)
	case closeErr != nil:
		return nil, s.wrapSearchError("semantic", q.Scope, closeErr)
	}
	return candidates, nil
}

// SearchLexical returns the best full-text matches, best first.
//
// The query is parsed with the text search configuration the schema was migrated
// with, using websearch_to_tsquery so a user-supplied string can never be a
// syntax error. A query of nothing but stop words yields an empty tsquery and so
// no matches, which is an empty result rather than a failure.
func (s *Store) SearchLexical(
	ctx context.Context,
	q mempher.LexicalQuery,
) ([]mempher.Candidate, error) {
	if err := validateLexical(q); err != nil {
		return nil, err
	}

	b := &queryBuilder{}
	config := b.arg(s.textSearchConfig)
	text := b.arg(q.Text)
	scope := b.arg(string(q.Scope))
	asOf := b.arg(q.AsOf)

	var sql strings.Builder
	fmt.Fprintf(&sql, `SELECT %s, ts_rank_cd(e.content_tsv, tsq.query) AS score
FROM mempher.episodes e, websearch_to_tsquery(%s::regconfig, %s) AS tsq(query)
WHERE e.scope_id = %s AND e.ingested_at <= %s AND e.content_tsv @@ tsq.query`,
		episodeColumnsQualified, config, text, scope, asOf)
	b.appendEpisodeFilters(&sql, q.ChannelQuery)
	// seq descending is a deterministic tie-break within a scope: on equal
	// rank, the more recent episode wins.
	fmt.Fprintf(&sql, "\nORDER BY score DESC, e.seq DESC\nLIMIT %s",
		b.arg(candidateLimit(q.Limit)))

	rows, err := s.pool.Query(ctx, sql.String(), b.args...)
	if err != nil {
		return nil, s.wrapSearchError("lexical", q.Scope, err)
	}
	defer rows.Close()

	candidates, err := scanCandidates(rows, mempher.ChannelLexical, q.Limit)
	if err != nil {
		return nil, s.wrapSearchError("lexical", q.Scope, err)
	}
	return candidates, nil
}

// queryBuilder numbers bind parameters as the SQL is assembled, so a filter can
// be added without renumbering everything after it.
type queryBuilder struct {
	args []any
}

// arg records a value and returns its placeholder.
func (b *queryBuilder) arg(v any) string {
	b.args = append(b.args, v)
	return "$" + strconv.Itoa(len(b.args))
}

// appendEpisodeFilters adds the predicates both channels share. They read
// columns of the episode, so both channels join to it.
func (b *queryBuilder) appendEpisodeFilters(sql *strings.Builder, q mempher.ChannelQuery) {
	if !q.OccurredFrom.IsZero() {
		fmt.Fprintf(sql, "\n  AND e.occurred_at >= %s", b.arg(q.OccurredFrom))
	}
	if !q.OccurredTo.IsZero() {
		fmt.Fprintf(sql, "\n  AND e.occurred_at <= %s", b.arg(q.OccurredTo))
	}
	if len(q.Binding) > 0 {
		// Containment, so the GIN jsonb_path_ops index serves it.
		fmt.Fprintf(sql, "\n  AND e.binding @> %s", b.arg(q.Binding))
	}
	if len(q.Roles) > 0 {
		roles := make([]string, len(q.Roles))
		for i, role := range q.Roles {
			roles[i] = string(role)
		}
		fmt.Fprintf(sql, "\n  AND e.role = ANY(%s)", b.arg(roles))
	}
}

// scanCandidates reads a channel's result set and ranks it.
//
// Rank comes from position rather than from a window function: one source of
// truth, and it cannot disagree with the slice fusion actually consumes.
//
// The rows are re-sorted before ranking, because equal scores otherwise arrive
// in whatever order the scan produced, and a vector index has no reason to be
// consistent about that between runs. Recall promises that replaying a request
// at the same AsOf reproduces the result, and ties are the ordinary case, not a
// corner: four episodes that mention a term equally often are exactly tied.
// Sorting here costs nothing at candidate-set sizes and cannot change the plan.
func scanCandidates(
	rows pgx.Rows,
	channel mempher.Channel,
	limit int,
) ([]mempher.Candidate, error) {
	out := make([]mempher.Candidate, 0, min(candidateLimit(limit), 128))
	for rows.Next() {
		var (
			ep      mempher.Episode
			id      [16]byte
			role    string
			binding mempher.Binding
			score   float64
		)
		if err := rows.Scan(&id, &ep.Scope, &ep.Seq, &ep.Content, &role, &ep.Actor,
			&ep.Source, &ep.OccurredAt, &ep.IngestedAt, &binding, &score); err != nil {
			return nil, fmt.Errorf("scan candidate: %w", err)
		}

		ep.ID = mempher.EpisodeID(id)
		ep.Role = mempher.Role(role)
		if !ep.Role.Valid() {
			return nil, fmt.Errorf("episode %s has role %q: %w",
				ep.ID, role, mempher.ErrInvalidRole)
		}
		ep.OccurredAt = ep.OccurredAt.UTC()
		ep.IngestedAt = ep.IngestedAt.UTC()
		if len(binding) > 0 {
			ep.Binding = binding
		}

		out = append(out, mempher.Candidate{
			Episode: ep,
			Channel: channel,
			Score:   score,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read candidates: %w", err)
	}

	// Score first, then the newer episode: the same rule fusion uses to break
	// its own ties, so the two cannot disagree.
	slices.SortStableFunc(out, func(a, b mempher.Candidate) int {
		if c := cmp.Compare(b.Score, a.Score); c != 0 {
			return c
		}
		return bytes.Compare(b.Episode.ID[:], a.Episode.ID[:])
	})
	for i := range out {
		out[i].Rank = i + 1
	}
	return out, nil
}

// candidateLimit resolves how deep a channel searches.
func candidateLimit(limit int) int {
	if limit <= 0 {
		return mempher.DefaultCandidatesPerChannel
	}
	return limit
}

// validateSemantic rejects a query that cannot be answered.
func (s *Store) validateSemantic(q mempher.SemanticQuery) error {
	if err := validateChannelQuery(q.ChannelQuery); err != nil {
		return fmt.Errorf("mempher/postgres: search semantic: %w", err)
	}
	if q.Model == "" {
		return fmt.Errorf("mempher/postgres: search semantic: model is empty: %w",
			mempher.ErrInvalidConfig)
	}
	if got := q.Vector.Dimensions(); got != s.vectorDimensions {
		return fmt.Errorf(
			"mempher/postgres: search semantic: query vector has %d dimensions but the schema was migrated for %d: %w",
			got, s.vectorDimensions, mempher.ErrDimensionMismatch)
	}
	return nil
}

// validateLexical rejects a query that cannot be answered.
func validateLexical(q mempher.LexicalQuery) error {
	if err := validateChannelQuery(q.ChannelQuery); err != nil {
		return fmt.Errorf("mempher/postgres: search lexical: %w", err)
	}
	if strings.TrimSpace(q.Text) == "" {
		return fmt.Errorf("mempher/postgres: search lexical: text is empty: %w",
			mempher.ErrInvalidContent)
	}
	return nil
}

// validateChannelQuery checks what both channels need.
func validateChannelQuery(q mempher.ChannelQuery) error {
	if err := q.Scope.Validate(); err != nil {
		return fmt.Errorf("scope: %w", err)
	}
	// A zero AsOf would silently match nothing, since every episode was
	// ingested after the zero time. Recall resolves it from the clock; a Store
	// insists on having been told.
	if q.AsOf.IsZero() {
		return fmt.Errorf("AsOf must be resolved by the caller: %w", mempher.ErrInvalidConfig)
	}
	if q.Limit < 0 {
		return fmt.Errorf("limit is %d: %w", q.Limit, mempher.ErrInvalidConfig)
	}
	if !q.OccurredFrom.IsZero() && !q.OccurredTo.IsZero() &&
		q.OccurredTo.Before(q.OccurredFrom) {
		return fmt.Errorf("event-time window ends before it starts: %w", mempher.ErrInvalidConfig)
	}
	for _, role := range q.Roles {
		if !role.Valid() {
			return fmt.Errorf("role filter %q: %w", role, mempher.ErrInvalidRole)
		}
	}
	if err := q.Binding.Validate(); err != nil {
		return fmt.Errorf("binding filter: %w", err)
	}
	return nil
}

// wrapSearchError names the channel and turns an absent schema into its sentinel.
func (s *Store) wrapSearchError(channel string, scope mempher.ScopeID, err error) error {
	if isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema) {
		return fmt.Errorf("mempher/postgres: search %s: run Migrate first: %w",
			channel, ErrSchemaNotReady)
	}
	return fmt.Errorf("mempher/postgres: search %s in scope %q: %w", channel, scope, err)
}
