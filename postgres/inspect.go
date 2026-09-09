// Reading what the server is, and interpolating identifiers safely where SQL
// leaves no room for a parameter.

package postgres

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/jackc/pgx/v5"
)

// MinServerVersionNum is the lowest server_version_num this adapter runs on:
// PostgreSQL 18.0.
const MinServerVersionNum = 180000

// requiredExtensions are what the schema cannot be built without: pgvector for
// the vector type and HNSW index, btree_gin for the text operator class that
// lets one GIN index cover both scope_id and the tsvector.
var requiredExtensions = []string{"vector", "btree_gin"}

// serverInfo is what this adapter checks before it will talk to a database.
type serverInfo struct {
	// versionNum is server_version_num, e.g. 180003.
	versionNum int
	// version is the human-readable server version, for error messages.
	version string
	// extensionSchema is where the vector extension lives, or "" when it is
	// not installed yet.
	extensionSchema string
	// missing lists required extensions that are not installed.
	missing []string
}

// rowQuerier is the read-only subset of pgx used for inspection, so the same
// code serves a pool, a connection and a transaction.
type rowQuerier interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
	QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// inspect reads the server version and locates the required extensions.
func inspect(ctx context.Context, db rowQuerier) (serverInfo, error) {
	var info serverInfo
	err := db.QueryRow(ctx, `
		SELECT current_setting('server_version_num')::int, current_setting('server_version')
	`).Scan(&info.versionNum, &info.version)
	if err != nil {
		return serverInfo{}, fmt.Errorf("read server version: %w", err)
	}

	rows, err := db.Query(ctx, `
		SELECT e.extname, n.nspname
		FROM pg_extension e
		JOIN pg_namespace n ON n.oid = e.extnamespace
		WHERE e.extname = ANY($1)
	`, requiredExtensions)
	if err != nil {
		return serverInfo{}, fmt.Errorf("read installed extensions: %w", err)
	}
	installed := make(map[string]string, len(requiredExtensions))
	for rows.Next() {
		var name, schema string
		if err := rows.Scan(&name, &schema); err != nil {
			rows.Close()
			return serverInfo{}, fmt.Errorf("scan installed extension: %w", err)
		}
		installed[name] = schema
	}
	if err := rows.Err(); err != nil {
		return serverInfo{}, fmt.Errorf("read installed extensions: %w", err)
	}

	for _, name := range requiredExtensions {
		if _, ok := installed[name]; !ok {
			info.missing = append(info.missing, name)
		}
	}
	info.extensionSchema = installed["vector"]
	return info, nil
}

// checkVersion rejects a server too old for the schema.
func (info serverInfo) checkVersion() error {
	if info.versionNum < MinServerVersionNum {
		return fmt.Errorf("mempher/postgres: server is %s, need 18.0 or newer: %w",
			info.version, ErrUnsupportedServer)
	}
	return nil
}

// identifierPattern is narrower than PostgreSQL allows: anything matching it is
// safe to interpolate, and anything else is reported as a configuration error
// rather than hoped for.
var identifierPattern = regexp.MustCompile(`^[a-z_][a-z0-9_$]{0,62}$`)

// validateIdentifier rejects anything that should not be interpolated into SQL.
// Schema and text search configuration names cannot be bind parameters: SET and
// DDL take literals only.
func validateIdentifier(kind, name string) error {
	switch {
	case name == "":
		return fmt.Errorf("mempher/postgres: %s is empty", kind)
	case !identifierPattern.MatchString(name):
		return fmt.Errorf(
			"mempher/postgres: %s %q must match %s (lowercase, unquoted PostgreSQL identifier)",
			kind, name, identifierPattern)
	}
	return nil
}

// quoteIdentifier wraps an identifier in double quotes. Validation already
// excludes quotes, but a helper producing SQL should not rely on its caller.
func quoteIdentifier(name string) string {
	return `"` + strings.ReplaceAll(name, `"`, `""`) + `"`
}
