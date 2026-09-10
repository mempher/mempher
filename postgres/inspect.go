// Reading what the server is, and interpolating identifiers safely where SQL
// leaves no room for a parameter.

package postgres

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
)

// MinServerVersionNum is the lowest server_version_num this adapter runs on:
// PostgreSQL 18.0.
const MinServerVersionNum = 180000

// requiredExtensions are what the schema cannot be built without: pgvector for
// the vector type and HNSW index, btree_gin for the text operator class that
// lets one GIN index cover both scope_id and the tsvector, and btree_gist for
// the scalar operator classes a WITHOUT OVERLAPS temporal key needs.
//
// This is a pre-flight check, and it exists so a role without CREATE EXTENSION
// fails here with a legible message rather than partway through a migration.
var requiredExtensions = []string{"vector", "btree_gin", "btree_gist"}

// serverInfo is what this adapter checks before it will talk to a database.
type serverInfo struct {
	// versionNum is server_version_num, e.g. 180003.
	versionNum int
	// version is the human-readable server version, for error messages.
	version string
	// extensionSchema is where the vector extension lives, or "" when it is
	// not installed yet.
	extensionSchema string
	// vectorVersion is pgvector's extversion, or "" when it is not installed.
	vectorVersion string
	// missing lists required extensions that are not installed.
	missing []string
}

// MinVectorVersion is the lowest pgvector this adapter runs on. 0.8 introduced
// iterative index scans, which a scope-filtered search cannot be correct without.
const MinVectorVersion = "0.8"

// checkVectorVersion rejects a pgvector too old for a filtered vector search.
func (info serverInfo) checkVectorVersion() error {
	major, minor, ok := parseVersion(info.vectorVersion)
	if !ok {
		return fmt.Errorf("mempher/postgres: cannot read the pgvector version (%q): %w",
			info.vectorVersion, ErrUnsupportedExtension)
	}
	if major == 0 && minor < 8 {
		return fmt.Errorf(
			"mempher/postgres: pgvector is %s, need %s or newer for iterative index scans: %w",
			info.vectorVersion, MinVectorVersion, ErrUnsupportedExtension)
	}
	return nil
}

// parseVersion reads the leading major and minor numbers of an extension version.
func parseVersion(version string) (major, minor int, ok bool) {
	parts := strings.SplitN(version, ".", 3)
	if len(parts) < 2 {
		return 0, 0, false
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, false
	}
	minor, err = strconv.Atoi(parts[1])
	if err != nil {
		return 0, 0, false
	}
	return major, minor, true
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
		SELECT e.extname, n.nspname, e.extversion
		FROM pg_extension e
		JOIN pg_namespace n ON n.oid = e.extnamespace
		WHERE e.extname = ANY($1)
	`, requiredExtensions)
	if err != nil {
		return serverInfo{}, fmt.Errorf("read installed extensions: %w", err)
	}
	installed := make(map[string]string, len(requiredExtensions))
	versions := make(map[string]string, len(requiredExtensions))
	for rows.Next() {
		var name, schema, version string
		if err := rows.Scan(&name, &schema, &version); err != nil {
			rows.Close()
			return serverInfo{}, fmt.Errorf("scan installed extension: %w", err)
		}
		installed[name] = schema
		versions[name] = version
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
	info.vectorVersion = versions["vector"]
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
