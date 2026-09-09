// Applying the embedded schema: forward-only, serialised, and idempotent.

package postgres

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/mempher/mempher"
)

// Schema is the single PostgreSQL schema mempher owns, so it installs beside an
// application's own tables and uninstalls by dropping one schema.
const Schema = "mempher"

// Defaults for [MigrateOptions].
const (
	// DefaultTextSearchConfig is the text search configuration the lexical
	// channel uses.
	DefaultTextSearchConfig = "english"
	// DefaultExtensionSchema is where extensions are created when they are
	// not installed already.
	DefaultExtensionSchema = "public"
	// DefaultMigrateLockTimeout is how long [Migrate] waits for another
	// process to finish migrating before giving up.
	DefaultMigrateLockTimeout = time.Minute
)

// MaxVectorDimensions is the widest embedding this schema accepts. The limit is
// pgvector's HNSW index, not the vector type: an unindexed vector column would
// make the semantic channel a sequential scan.
const MaxVectorDimensions = 2000

// advisoryLockKey serialises migrations across every process on one database. It
// is the FNV-1a 64 hash of "mempher.schema_migrations" as a signed integer, and
// must never change: two builds with different keys would migrate concurrently.
const advisoryLockKey int64 = 8348320501231727730

// Polling with pg_try_advisory_lock rather than blocking in pg_advisory_lock
// keeps the outcome unambiguous: a cancelled wait cannot leave a lock this
// process holds but does not know about.
const (
	lockPollInterval = 50 * time.Millisecond
	lockPollMax      = 500 * time.Millisecond
	lockReleaseGrace = 5 * time.Second
)

// MigrateOptions carries the two decisions that must be baked into DDL, plus how
// long to wait for a concurrent migration.
//
// Both are permanent for the life of a database: a generated tsvector column
// needs an immutable expression, and an HNSW index needs a declared width.
// [Migrate] records them and refuses to proceed if they later disagree.
type MigrateOptions struct {
	// VectorDimensions is the embedding width, normally taken from
	// [mempher.Embedder.Dimensions]. Required, and at most
	// [MaxVectorDimensions].
	VectorDimensions int

	// TextSearchConfig is the text search configuration for the lexical
	// channel, such as "english" or "french". Empty means
	// [DefaultTextSearchConfig]. It is verified against the server, so a typo
	// is reported before any DDL runs.
	//
	// One configuration serves the whole database; per-scope languages are a
	// later stage's problem.
	TextSearchConfig string

	// ExtensionSchema is where the vector and btree_gin extensions live.
	// Empty means: wherever they are already installed, or
	// [DefaultExtensionSchema] if they are not. Setting it explicitly when
	// they are installed elsewhere is reported as a mismatch rather than
	// silently ignored.
	ExtensionSchema string

	// LockTimeout is how long to wait for another process's migration to
	// finish. Zero means [DefaultMigrateLockTimeout]. Exceeding it returns
	// [ErrLockTimeout], which is safe to retry.
	LockTimeout time.Duration
}

// templateData is what a migration template sees, kept distinct so adding an
// option does not silently expose it to DDL.
type templateData struct {
	VectorDimensions int
	TextSearchConfig string
}

func (o MigrateOptions) templateData() templateData {
	return templateData{
		VectorDimensions: o.VectorDimensions,
		TextSearchConfig: o.TextSearchConfig,
	}
}

// validate fills in defaults and rejects options that cannot produce valid DDL.
// Checks needing a server happen in [Migrate].
func (o MigrateOptions) validate() (MigrateOptions, error) {
	out := o
	switch {
	case out.VectorDimensions <= 0:
		return MigrateOptions{}, fmt.Errorf(
			"mempher/postgres: VectorDimensions is %d, must be positive: %w",
			out.VectorDimensions, mempher.ErrInvalidConfig)
	case out.VectorDimensions > MaxVectorDimensions:
		return MigrateOptions{}, fmt.Errorf(
			"mempher/postgres: VectorDimensions is %d, at most %d is indexable by HNSW: %w",
			out.VectorDimensions, MaxVectorDimensions, mempher.ErrInvalidConfig)
	}
	if out.TextSearchConfig == "" {
		out.TextSearchConfig = DefaultTextSearchConfig
	}
	if err := validateIdentifier("TextSearchConfig", out.TextSearchConfig); err != nil {
		return MigrateOptions{}, fmt.Errorf("%w: %w", err, mempher.ErrInvalidConfig)
	}
	if out.ExtensionSchema != "" {
		if err := validateIdentifier("ExtensionSchema", out.ExtensionSchema); err != nil {
			return MigrateOptions{}, fmt.Errorf("%w: %w", err, mempher.ErrInvalidConfig)
		}
	}
	if out.LockTimeout < 0 {
		return MigrateOptions{}, fmt.Errorf(
			"mempher/postgres: LockTimeout is %s: %w", out.LockTimeout, mempher.ErrInvalidConfig)
	}
	if out.LockTimeout == 0 {
		out.LockTimeout = DefaultMigrateLockTimeout
	}
	return out, nil
}

// MigrateResult reports what a run did, so a caller can log it without querying
// the database again.
type MigrateResult struct {
	// Applied lists the versions this call applied, in order. Empty means
	// the schema was already current.
	Applied []int
	// Version is the schema version after the call.
	Version int
	// ExtensionSchema is where the extensions were found or created.
	ExtensionSchema string
}

// Migrate brings the database up to the schema this build embeds and reports what
// it did.
//
// It is safe to call from every instance of an application at startup, and safe
// to call repeatedly: work is serialised on an advisory lock, each pending
// migration commits in its own transaction, and a current database is untouched.
//
// It refuses rather than guesses when the database disagrees with the build:
// [ErrChecksumMismatch] for edited history, [ErrDatabaseAhead] for a newer
// database, [ErrSchemaConfigMismatch] for different DDL options. All three are
// cheaper to hear about at startup than to diagnose from query results.
func Migrate(
	ctx context.Context,
	pool *pgxpool.Pool,
	opts MigrateOptions,
) (result MigrateResult, err error) {
	opts, err = opts.validate()
	if err != nil {
		return MigrateResult{}, err
	}
	migrations, err := loadMigrations()
	if err != nil {
		return MigrateResult{}, fmt.Errorf("mempher/postgres: %w", err)
	}
	if pool == nil {
		return MigrateResult{}, fmt.Errorf(
			"mempher/postgres: migrate: pool is nil: %w", mempher.ErrInvalidConfig)
	}

	// One connection for the whole run: the lock is session-scoped, so every
	// migration must travel over the session holding it.
	conn, err := pool.Acquire(ctx)
	if err != nil {
		return MigrateResult{}, fmt.Errorf("mempher/postgres: migrate: acquire connection: %w", err)
	}
	locked := false
	defer func() {
		err = errors.Join(err, finishMigrationSession(ctx, conn, locked))
	}()

	info, err := inspect(ctx, conn)
	if err != nil {
		return MigrateResult{}, fmt.Errorf("mempher/postgres: migrate: %w", err)
	}
	if err := info.checkVersion(); err != nil {
		return MigrateResult{}, err
	}
	if opts, err = resolveExtensionSchema(opts, info); err != nil {
		return MigrateResult{}, err
	}
	if err := checkTextSearchConfig(ctx, conn, opts.TextSearchConfig); err != nil {
		return MigrateResult{}, err
	}

	if err := acquireMigrationLock(ctx, conn, opts.LockTimeout); err != nil {
		return MigrateResult{}, err
	}
	locked = true

	if err := bootstrap(ctx, conn); err != nil {
		return MigrateResult{}, fmt.Errorf("mempher/postgres: migrate: %w", err)
	}
	applied, err := appliedMigrations(ctx, conn)
	if err != nil {
		return MigrateResult{}, fmt.Errorf("mempher/postgres: migrate: %w", err)
	}
	if err := checkHistory(migrations, applied); err != nil {
		return MigrateResult{}, err
	}

	result = MigrateResult{ExtensionSchema: opts.ExtensionSchema}
	for _, m := range migrations {
		if _, done := applied[m.version]; done {
			result.Version = m.version
			continue
		}
		if err := applyMigration(ctx, conn, m, opts, info); err != nil {
			return MigrateResult{}, err
		}
		result.Applied = append(result.Applied, m.version)
		result.Version = m.version
	}

	if err := checkSchemaConfig(ctx, conn, opts); err != nil {
		return MigrateResult{}, err
	}
	return result, nil
}

// finishMigrationSession releases the advisory lock and hands the connection
// back.
//
// A pooled connection still holding a session lock would poison the pool: every
// later borrower would silently hold the migration lock, and the next migration
// anywhere would block until that connection closed. So if the release does not
// succeed, the connection is discarded rather than returned.
func finishMigrationSession(ctx context.Context, conn *pgxpool.Conn, locked bool) error {
	if !locked {
		conn.Release()
		return nil
	}
	// ctx may already be cancelled, and the lock still has to come off.
	releaseCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), lockReleaseGrace)
	defer cancel()

	var released bool
	err := conn.QueryRow(releaseCtx, "SELECT pg_advisory_unlock($1)", advisoryLockKey).Scan(&released)
	switch {
	case err != nil:
		raw := conn.Hijack()
		_ = raw.Close(releaseCtx)
		return fmt.Errorf(
			"mempher/postgres: release migration lock (connection discarded): %w", err)
	case !released:
		// Unreachable: this session took the lock. If it happens, the
		// session's state is not what we believe, so discard it.
		raw := conn.Hijack()
		_ = raw.Close(releaseCtx)
		return errors.New(
			"mempher/postgres: migration lock was not held at release (connection discarded)")
	}
	conn.Release()
	return nil
}

// resolveExtensionSchema prefers what the database already does over what the
// caller assumed.
func resolveExtensionSchema(opts MigrateOptions, info serverInfo) (MigrateOptions, error) {
	switch {
	case info.extensionSchema == "":
		// Not installed: the migration creates it, in the first schema on
		// the search_path applyMigration sets.
		if opts.ExtensionSchema == "" {
			opts.ExtensionSchema = DefaultExtensionSchema
		}
	case opts.ExtensionSchema == "":
		opts.ExtensionSchema = info.extensionSchema
	case opts.ExtensionSchema != info.extensionSchema:
		return MigrateOptions{}, fmt.Errorf(
			"mempher/postgres: ExtensionSchema is %q but the vector extension is installed in %q: %w",
			opts.ExtensionSchema, info.extensionSchema, ErrSchemaConfigMismatch)
	}
	if err := validateIdentifier("ExtensionSchema", opts.ExtensionSchema); err != nil {
		return MigrateOptions{}, fmt.Errorf("%w: %w", err, mempher.ErrInvalidConfig)
	}
	return opts, nil
}

// checkTextSearchConfig asks the server whether the configuration exists, so a
// typo fails before any DDL runs.
func checkTextSearchConfig(ctx context.Context, conn *pgxpool.Conn, name string) error {
	// There is no to_regconfig() to mirror to_regclass(), and casting to
	// regconfig raises rather than returning NULL.
	var exists bool
	if err := conn.QueryRow(ctx,
		"SELECT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname::text = $1)",
		name).Scan(&exists); err != nil {
		return fmt.Errorf("mempher/postgres: check text search config %q: %w", name, err)
	}
	if !exists {
		return fmt.Errorf(
			"mempher/postgres: text search configuration %q does not exist on this server "+
				"(see pg_ts_config for the available ones): %w", name, mempher.ErrInvalidConfig)
	}
	return nil
}

// acquireMigrationLock waits for exclusive rights to migrate.
func acquireMigrationLock(ctx context.Context, conn *pgxpool.Conn, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	backoff := lockPollInterval
	for {
		var got bool
		if err := conn.QueryRow(ctx,
			"SELECT pg_try_advisory_lock($1)", advisoryLockKey).Scan(&got); err != nil {
			return fmt.Errorf("mempher/postgres: acquire migration lock: %w", err)
		}
		if got {
			return nil
		}
		if !time.Now().Add(backoff).Before(deadline) {
			return fmt.Errorf(
				"mempher/postgres: another process has been migrating for more than %s: %w",
				timeout, ErrLockTimeout)
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("mempher/postgres: waiting for migration lock: %w", ctx.Err())
		case <-time.After(backoff):
		}
		if backoff < lockPollMax {
			backoff = min(backoff*2, lockPollMax)
		}
	}
}

// bootstrap creates the schema and the ledger. It runs under the advisory lock,
// so IF NOT EXISTS is belt-and-braces rather than the mechanism.
func bootstrap(ctx context.Context, conn *pgxpool.Conn) error {
	const ddl = `
		CREATE SCHEMA IF NOT EXISTS ` + Schema + `;

		CREATE TABLE IF NOT EXISTS ` + Schema + `.schema_migrations (
		    version     int         NOT NULL,
		    name        text        NOT NULL,
		    checksum    text        NOT NULL,
		    applied_at  timestamptz NOT NULL DEFAULT now(),
		    duration_ms bigint      NOT NULL,

		    CONSTRAINT schema_migrations_pkey PRIMARY KEY (version),
		    CONSTRAINT schema_migrations_version_positive CHECK (version > 0),
		    CONSTRAINT schema_migrations_duration_nonneg CHECK (duration_ms >= 0)
		);

		COMMENT ON TABLE ` + Schema + `.schema_migrations IS
		    'Forward-only migration ledger. checksum fingerprints the migration template, not the rendered SQL.';
	`
	if _, err := conn.Exec(ctx, ddl); err != nil {
		return fmt.Errorf("bootstrap schema: %w", err)
	}
	return nil
}

// appliedMigrations reads the ledger, keyed by version.
func appliedMigrations(ctx context.Context, conn *pgxpool.Conn) (map[int]appliedMigration, error) {
	rows, err := conn.Query(ctx,
		`SELECT version, name, checksum FROM `+Schema+`.schema_migrations ORDER BY version`)
	if err != nil {
		return nil, fmt.Errorf("read migration ledger: %w", err)
	}
	defer rows.Close()

	out := make(map[int]appliedMigration)
	for rows.Next() {
		var a appliedMigration
		if err := rows.Scan(&a.version, &a.name, &a.checksum); err != nil {
			return nil, fmt.Errorf("scan migration ledger: %w", err)
		}
		out[a.version] = a
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read migration ledger: %w", err)
	}
	return out, nil
}

// appliedMigration is one row of the ledger.
type appliedMigration struct {
	version  int
	name     string
	checksum string
}

// checkHistory compares the ledger against what this build embeds.
func checkHistory(migrations []migration, applied map[int]appliedMigration) error {
	embedded := make(map[int]migration, len(migrations))
	for _, m := range migrations {
		embedded[m.version] = m
	}

	// Both passes walk sorted versions rather than the map, so a database with
	// several problems always reports the same one first.
	for _, version := range slices.Sorted(maps.Keys(applied)) {
		if _, known := embedded[version]; !known {
			return fmt.Errorf(
				"mempher/postgres: database has migration %d (%s) which this build does not embed, "+
					"so it was migrated by a newer mempher: %w",
				version, applied[version].name, ErrDatabaseAhead)
		}
	}
	for _, m := range migrations {
		a, done := applied[m.version]
		if !done {
			continue
		}
		if a.checksum != m.checksum {
			return fmt.Errorf(
				"mempher/postgres: migration %s was applied as checksum %s but now hashes to %s; "+
					"migrations are forward-only, so add a new migration instead of editing it: %w",
				m.filename(), short(a.checksum), short(m.checksum), ErrChecksumMismatch)
		}
	}
	return nil
}

// applyMigration renders, runs and records one migration in a single
// transaction: the DDL and its ledger row land together or not at all.
func applyMigration(
	ctx context.Context,
	conn *pgxpool.Conn,
	m migration,
	opts MigrateOptions,
	info serverInfo,
) error {
	sql, err := m.render(opts)
	if err != nil {
		return fmt.Errorf("mempher/postgres: %w", err)
	}

	tx, err := conn.Begin(ctx)
	if err != nil {
		return fmt.Errorf("mempher/postgres: begin migration %s: %w", m.filename(), err)
	}
	defer func() { _ = tx.Rollback(ctx) }() // no-op once committed

	// The extension schema goes first so CREATE EXTENSION lands there and the
	// unqualified vector type and btree_gin operator classes resolve.
	// pg_catalog is searched ahead of both, so this shadows no built-in.
	setPath := fmt.Sprintf("SET LOCAL search_path TO %s, %s",
		quoteIdentifier(opts.ExtensionSchema), quoteIdentifier(Schema))
	if _, err := tx.Exec(ctx, setPath); err != nil {
		return fmt.Errorf("mempher/postgres: set search_path for %s: %w", m.filename(), err)
	}

	started := time.Now()
	if _, err := tx.Exec(ctx, sql); err != nil {
		return fmt.Errorf("mempher/postgres: apply migration %s: %w",
			m.filename(), annotateMigrationError(err, info))
	}
	elapsed := time.Since(started)

	if _, err := tx.Exec(ctx, `
		INSERT INTO `+Schema+`.schema_migrations (version, name, checksum, duration_ms)
		VALUES ($1, $2, $3, $4)
	`, m.version, m.name, m.checksum, elapsed.Milliseconds()); err != nil {
		return fmt.Errorf("mempher/postgres: record migration %s: %w", m.filename(), err)
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("mempher/postgres: commit migration %s: %w", m.filename(), err)
	}
	return nil
}

// annotateMigrationError turns a privilege failure into a message that says what
// to do about it.
func annotateMigrationError(err error, info serverInfo) error {
	var pgErr *pgconn.PgError
	if !errors.As(err, &pgErr) {
		return err
	}
	if pgErr.Code == pgerrcodeInsufficientPrivilege && len(info.missing) > 0 {
		return fmt.Errorf(
			"%w (this database is missing %v, and the current role may not create extensions; "+
				"ask a privileged role to run CREATE EXTENSION for each, then migrate again): %w",
			err, info.missing, ErrMissingExtension)
	}
	return err
}

// pgerrcodeInsufficientPrivilege is SQLSTATE 42501.
const pgerrcodeInsufficientPrivilege = "42501"

// checkSchemaConfig confirms the database was built for the options in hand.
// This is what catches a swapped embedding model; without it a mismatch surfaces
// as every write failing on width, or as the wrong language quietly stemming.
func checkSchemaConfig(ctx context.Context, conn *pgxpool.Conn, opts MigrateOptions) error {
	var dims int
	var config string
	err := conn.QueryRow(ctx,
		`SELECT vector_dimensions, text_search_config FROM `+Schema+`.schema_config`).
		Scan(&dims, &config)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return fmt.Errorf(
			"mempher/postgres: %s.schema_config is empty: %w", Schema, ErrSchemaNotReady)
	case err != nil:
		return fmt.Errorf("mempher/postgres: read schema config: %w", err)
	}
	if dims != opts.VectorDimensions || config != opts.TextSearchConfig {
		return fmt.Errorf(
			"mempher/postgres: database was migrated for %d dimensions and %q text search, "+
				"but %d dimensions and %q were requested; both are baked into DDL, so this "+
				"needs a new database or a migration that rewrites the schema: %w",
			dims, config, opts.VectorDimensions, opts.TextSearchConfig, ErrSchemaConfigMismatch)
	}
	return nil
}

// short truncates a checksum for an error message.
func short(checksum string) string {
	if len(checksum) <= 12 {
		return checksum
	}
	return checksum[:12]
}
