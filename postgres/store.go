// The adapter: one Store over one pool, serving both mempher ports.

package postgres

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mempher/mempher"
)

// SQLSTATEs this adapter reacts to rather than passes through.
const (
	pgerrcodeUndefinedTable   = "42P01"
	pgerrcodeInvalidSchema    = "3F000"
	pgerrcodeUniqueViolation  = "23505"
	pgerrcodeCheckViolation   = "23514"
	pgerrcodeForeignKeyViolat = "23503"
)

// Store persists episodes and their projections in PostgreSQL. It satisfies both
// [mempher.Store] and [mempher.Queue], and is safe for concurrent use.
//
// It does not own the pool: closing that is the caller's business.
type Store struct {
	pool *pgxpool.Pool

	// Read once in New, because each is fixed for the life of the schema.
	extensionSchema  string
	vectorDimensions int
	textSearchConfig string

	// vectorType is the qualified pgvector type name for casts, so queries do
	// not depend on the pool's search_path.
	vectorType string
}

// New connects a Store to a database that [Migrate] has already prepared.
//
// It reads the schema's fixed properties once: the server version, where the
// extensions live, and the embedding width and text search configuration the
// schema was built for. A database that is unmigrated, too old or missing an
// extension therefore fails here with one legible error, rather than on every
// query.
func New(ctx context.Context, pool *pgxpool.Pool) (*Store, error) {
	if pool == nil {
		return nil, fmt.Errorf("mempher/postgres: new: pool is nil: %w", mempher.ErrInvalidConfig)
	}

	info, err := inspect(ctx, pool)
	if err != nil {
		return nil, fmt.Errorf("mempher/postgres: new: %w", err)
	}
	if err := info.checkVersion(); err != nil {
		return nil, err
	}
	if len(info.missing) > 0 {
		return nil, fmt.Errorf(
			"mempher/postgres: new: %v not installed; run Migrate, or have a privileged role create them: %w",
			info.missing, ErrMissingExtension)
	}
	if err := validateIdentifier("extension schema", info.extensionSchema); err != nil {
		return nil, fmt.Errorf("mempher/postgres: new: %w", err)
	}

	s := &Store{
		pool:            pool,
		extensionSchema: info.extensionSchema,
		vectorType:      quoteIdentifier(info.extensionSchema) + ".vector",
	}
	if err := s.readSchemaConfig(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

// readSchemaConfig loads what the schema was migrated for.
func (s *Store) readSchemaConfig(ctx context.Context) error {
	err := s.pool.QueryRow(ctx,
		`SELECT vector_dimensions, text_search_config FROM mempher.schema_config`).
		Scan(&s.vectorDimensions, &s.textSearchConfig)

	switch {
	case isCode(err, pgerrcodeUndefinedTable, pgerrcodeInvalidSchema):
		return fmt.Errorf(
			"mempher/postgres: new: the %s schema is absent or incomplete; run Migrate first: %w",
			Schema, ErrSchemaNotReady)
	case errors.Is(err, pgx.ErrNoRows):
		return fmt.Errorf("mempher/postgres: new: %s.schema_config is empty: %w",
			Schema, ErrSchemaNotReady)
	case err != nil:
		return fmt.Errorf("mempher/postgres: new: read schema config: %w", err)
	}

	if s.vectorDimensions <= 0 {
		return fmt.Errorf("mempher/postgres: new: schema records %d dimensions: %w",
			s.vectorDimensions, ErrSchemaNotReady)
	}
	if err := validateIdentifier("text search config", s.textSearchConfig); err != nil {
		return fmt.Errorf("mempher/postgres: new: %w", err)
	}
	return nil
}

// Dimensions reports the embedding width the schema was migrated for.
func (s *Store) Dimensions() int { return s.vectorDimensions }

// TextSearchConfig reports the configuration the lexical channel parses with.
func (s *Store) TextSearchConfig() string { return s.textSearchConfig }

// ExtensionSchema reports where pgvector and btree_gin were found.
func (s *Store) ExtensionSchema() string { return s.extensionSchema }

// VerifyEmbedder reports whether emb agrees with the schema, and is what turns a
// swapped embedding model into one clear error at startup instead of a failure
// on every write. It needs no context: the width was read in [New].
func (s *Store) VerifyEmbedder(emb mempher.Embedder) error {
	if emb == nil {
		return fmt.Errorf("mempher/postgres: verify embedder: embedder is nil: %w",
			mempher.ErrInvalidConfig)
	}
	if got := emb.Dimensions(); got != s.vectorDimensions {
		return fmt.Errorf(
			"mempher/postgres: embedder %q produces %d dimensions but the schema was migrated for %d: %w",
			emb.Model(), got, s.vectorDimensions, mempher.ErrEmbedderMismatch)
	}
	return nil
}

// isCode reports whether err is a PostgreSQL error carrying one of the SQLSTATEs.
func isCode(err error, codes ...string) bool {
	var pgErr *pgconn.PgError
	if !errors.As(err, &pgErr) {
		return false
	}
	for _, code := range codes {
		if pgErr.Code == code {
			return true
		}
	}
	return false
}

// The methods this stage implements, asserted against the shape [mempher.Store]
// requires. The full interface assertion lands with the search channels.
var _ interface {
	Append(context.Context, mempher.AppendCommand) (mempher.AppendResult, error)
	Episode(context.Context, mempher.ScopeID, mempher.EpisodeID) (mempher.Episode, error)
	Replay(context.Context, mempher.ScopeID, int64, int) ([]mempher.Episode, error)
	PutEncoding(context.Context, mempher.Encoding) error
	PendingEncodings(context.Context, mempher.PendingEncodings) ([]mempher.EpisodeID, error)
} = (*Store)(nil)
