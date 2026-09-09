// The embedded migration set: discovery, fingerprinting and rendering.

package postgres

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"fmt"
	"io/fs"
	"path"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"text/template"
)

// migrationFS holds the schema, so a binary carries the DDL it expects and the
// two cannot be deployed out of step. The files live here because go:embed
// cannot reach above its own directory.
//
//go:embed migrations/*.sql
var migrationFS embed.FS

const migrationDir = "migrations"

// migrationNamePattern matches NNNN_snake_case_name.sql, fixed width so a
// directory listing sorts the way this code does.
var migrationNamePattern = regexp.MustCompile(`^(\d{4})_([a-z0-9_]+)\.sql$`)

// migration is one forward-only step.
type migration struct {
	// version orders the step and is its identity in schema_migrations.
	version int
	// name is the human-readable half of the filename.
	name string
	// source is the unrendered template, and what checksum is taken over.
	source string
	// checksum fingerprints the template, not the rendered SQL, so the same
	// migration at a different embedding width is not read as edited history.
	checksum string
}

// render substitutes the deployment options into the template.
func (m migration) render(opts MigrateOptions) (string, error) {
	tmpl, err := template.New(m.filename()).
		Option("missingkey=error"). // a typo must fail here, not ship DDL with a hole
		Parse(m.source)
	if err != nil {
		return "", fmt.Errorf("parse migration %s: %w", m.filename(), err)
	}
	var out strings.Builder
	if err := tmpl.Execute(&out, opts.templateData()); err != nil {
		return "", fmt.Errorf("render migration %s: %w", m.filename(), err)
	}
	return out.String(), nil
}

// filename reconstructs the embedded name for error messages.
func (m migration) filename() string {
	return fmt.Sprintf("%04d_%s.sql", m.version, m.name)
}

// loadMigrations reads the embedded set in version order. A stray file, a
// duplicate version or a gap all mean the set is not what its author thought,
// and a migration runner is the wrong place to be forgiving.
func loadMigrations() ([]migration, error) {
	entries, err := fs.ReadDir(migrationFS, migrationDir)
	if err != nil {
		return nil, fmt.Errorf("read embedded migrations: %w", err)
	}

	out := make([]migration, 0, len(entries))
	seen := make(map[int]string, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			return nil, fmt.Errorf("unexpected directory %q in %s", entry.Name(), migrationDir)
		}
		match := migrationNamePattern.FindStringSubmatch(entry.Name())
		if match == nil {
			return nil, fmt.Errorf(
				"migration %q does not match NNNN_name.sql", entry.Name())
		}
		version, err := strconv.Atoi(match[1])
		if err != nil {
			return nil, fmt.Errorf("parse version of %q: %w", entry.Name(), err)
		}
		if version < 1 {
			return nil, fmt.Errorf("migration %q has version %d, must be positive",
				entry.Name(), version)
		}
		if other, dup := seen[version]; dup {
			return nil, fmt.Errorf("migrations %q and %q share version %d",
				other, entry.Name(), version)
		}
		seen[version] = entry.Name()

		source, err := migrationFS.ReadFile(path.Join(migrationDir, entry.Name()))
		if err != nil {
			return nil, fmt.Errorf("read %q: %w", entry.Name(), err)
		}
		sum := sha256.Sum256(source)
		out = append(out, migration{
			version:  version,
			name:     match[2],
			source:   string(source),
			checksum: hex.EncodeToString(sum[:]),
		})
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("no migrations embedded from %s", migrationDir)
	}
	slices.SortFunc(out, func(a, b migration) int { return a.version - b.version })

	// A deleted migration is a database that can never be rebuilt from this
	// source tree.
	for i, m := range out {
		if want := i + 1; m.version != want {
			return nil, fmt.Errorf(
				"migration versions must be contiguous from 1: found %d where %d was expected",
				m.version, want)
		}
	}
	return out, nil
}
