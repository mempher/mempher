// Expected conditions specific to this adapter.

package postgres

import "errors"

var (
	// ErrUnsupportedServer means the server is older than PostgreSQL 18,
	// whose uuidv7() and temporal constraints are not negotiable here.
	ErrUnsupportedServer = errors.New("unsupported PostgreSQL server")

	// ErrMissingExtension means a required extension is absent and could not
	// be created, which usually means a privileged role has to create it.
	ErrMissingExtension = errors.New("required extension is missing")

	// ErrChecksumMismatch means an applied migration's source has changed.
	// Migrations are forward-only: add a new one instead of editing an old one.
	ErrChecksumMismatch = errors.New("applied migration has changed")

	// ErrDatabaseAhead means the database holds a migration this build does
	// not know, so running this build would be a silent downgrade.
	ErrDatabaseAhead = errors.New("database schema is newer than this build")

	// ErrSchemaConfigMismatch means the database was migrated with a different
	// embedding width or text search configuration. Either is baked into DDL.
	ErrSchemaConfigMismatch = errors.New("schema was migrated with different options")

	// ErrSchemaNotReady means the mempher schema is absent or incomplete, so
	// [Migrate] has not run against this database.
	ErrSchemaNotReady = errors.New("mempher schema is not ready")

	// ErrLockTimeout means another process held the migration lock for longer
	// than the configured wait. It is safe to retry.
	ErrLockTimeout = errors.New("timed out waiting for the migration lock")
)
