package postgres

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/mempher/mempher"
)

// TestLoadMigrations checks the embedded set is well formed. It runs on every
// build, so adding a badly named or non-contiguous migration fails immediately
// rather than on somebody's first deploy.
func TestLoadMigrations(t *testing.T) {
	t.Parallel()

	migrations, err := loadMigrations()
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	if len(migrations) == 0 {
		t.Fatal("no migrations embedded")
	}

	for i, m := range migrations {
		if got, want := m.version, i+1; got != want {
			t.Errorf("migration %d has version %d, want %d", i, got, want)
		}
		if m.name == "" {
			t.Errorf("migration %d has an empty name", m.version)
		}
		if len(m.checksum) != 64 {
			t.Errorf("%s checksum is %d chars, want a 64-char sha256",
				m.filename(), len(m.checksum))
		}
		if strings.TrimSpace(m.source) == "" {
			t.Errorf("%s is empty", m.filename())
		}
	}

	// The checksum must fingerprint the template, not the rendered SQL:
	// otherwise deploying the same migration with a different embedding
	// width would look like somebody edited history.
	first := migrations[0]
	narrow, err := first.render(MigrateOptions{VectorDimensions: 8, TextSearchConfig: "english"})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	wide, err := first.render(MigrateOptions{VectorDimensions: 1536, TextSearchConfig: "english"})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	if narrow == wide {
		t.Error("rendering with different widths produced identical SQL")
	}
	again, err := loadMigrations()
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	if again[0].checksum != first.checksum {
		t.Error("checksum is not stable across loads")
	}
}

// TestMigrationRenderSubstitutesEverything guards against a placeholder being
// added to the SQL and forgotten in templateData, which would otherwise ship DDL
// with a literal "{{" in it.
func TestMigrationRenderSubstitutesEverything(t *testing.T) {
	t.Parallel()

	migrations, err := loadMigrations()
	if err != nil {
		t.Fatalf("loadMigrations: %v", err)
	}
	opts := MigrateOptions{VectorDimensions: 384, TextSearchConfig: "simple"}

	for _, m := range migrations {
		sql, err := m.render(opts)
		if err != nil {
			t.Fatalf("render %s: %v", m.filename(), err)
		}
		if strings.Contains(sql, "{{") || strings.Contains(sql, "}}") {
			t.Errorf("%s still holds a template placeholder after rendering", m.filename())
		}
		if strings.Contains(sql, "<no value>") {
			t.Errorf("%s rendered a missing value", m.filename())
		}
	}

	sql, err := migrations[0].render(opts)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	for _, want := range []string{"vector(384)", "to_tsvector('simple'", "384"} {
		if !strings.Contains(sql, want) {
			t.Errorf("rendered SQL does not contain %q", want)
		}
	}
}

func TestMigrateOptionsValidate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		opts     MigrateOptions
		wantErr  error
		wantConf string
		wantLock time.Duration
	}{
		{
			name:     "defaults fill in",
			opts:     MigrateOptions{VectorDimensions: 8},
			wantConf: DefaultTextSearchConfig,
			wantLock: DefaultMigrateLockTimeout,
		},
		{
			name:     "explicit values are kept",
			opts:     MigrateOptions{VectorDimensions: 8, TextSearchConfig: "french", LockTimeout: time.Second},
			wantConf: "french",
			wantLock: time.Second,
		},
		{
			name:    "zero dimensions",
			opts:    MigrateOptions{},
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "negative dimensions",
			opts:    MigrateOptions{VectorDimensions: -1},
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "dimensions past what HNSW indexes",
			opts:    MigrateOptions{VectorDimensions: MaxVectorDimensions + 1},
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:     "dimensions at the limit are fine",
			opts:     MigrateOptions{VectorDimensions: MaxVectorDimensions},
			wantConf: DefaultTextSearchConfig,
			wantLock: DefaultMigrateLockTimeout,
		},
		{
			name:    "text search config with SQL in it",
			opts:    MigrateOptions{VectorDimensions: 8, TextSearchConfig: "english'; DROP TABLE x --"},
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "text search config with a capital",
			opts:    MigrateOptions{VectorDimensions: 8, TextSearchConfig: "English"},
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "extension schema with a quote",
			opts:    MigrateOptions{VectorDimensions: 8, ExtensionSchema: `we"ird`},
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "negative lock timeout",
			opts:    MigrateOptions{VectorDimensions: 8, LockTimeout: -time.Second},
			wantErr: mempher.ErrInvalidConfig,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := tc.opts.validate()
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.TextSearchConfig != tc.wantConf {
				t.Errorf("TextSearchConfig = %q, want %q", got.TextSearchConfig, tc.wantConf)
			}
			if got.LockTimeout != tc.wantLock {
				t.Errorf("LockTimeout = %s, want %s", got.LockTimeout, tc.wantLock)
			}
		})
	}
}

func TestValidateIdentifier(t *testing.T) {
	t.Parallel()

	valid := []string{"public", "mempher", "english", "a", "_x", "s1", "with_underscores", "a$b"}
	for _, name := range valid {
		if err := validateIdentifier("test", name); err != nil {
			t.Errorf("validateIdentifier(%q) = %v, want nil", name, err)
		}
	}

	invalid := []string{
		"",                      // empty
		"Public",                // uppercase
		"1abc",                  // leading digit
		"has space",             // space
		`has"quote`,             // quote
		"has-dash",              // dash
		"has;semicolon",         // statement separator
		"public.schema",         // qualified
		strings.Repeat("x", 64), // too long for an unquoted identifier
		"drop table x",          // SQL
		"pg_catalog\nsomething", // newline
	}
	for _, name := range invalid {
		if err := validateIdentifier("test", name); err == nil {
			t.Errorf("validateIdentifier(%q) = nil, want an error", name)
		}
	}
}

func TestQuoteIdentifier(t *testing.T) {
	t.Parallel()

	tests := map[string]string{
		"public":  `"public"`,
		"mempher": `"mempher"`,
		`a"b`:     `"a""b"`, // never reachable via validation, but the helper must not be the weak link
	}
	for in, want := range tests {
		if got := quoteIdentifier(in); got != want {
			t.Errorf("quoteIdentifier(%q) = %s, want %s", in, got, want)
		}
	}
}

// TestCheckHistory covers the two ways a database can disagree with the build,
// without needing a database to do it.
func TestCheckHistory(t *testing.T) {
	t.Parallel()

	migrations := []migration{
		{version: 1, name: "init", checksum: "aaa"},
		{version: 2, name: "next", checksum: "bbb"},
	}

	tests := []struct {
		name    string
		applied map[int]appliedMigration
		wantErr error
	}{
		{name: "nothing applied", applied: map[int]appliedMigration{}},
		{
			name:    "partially applied",
			applied: map[int]appliedMigration{1: {version: 1, name: "init", checksum: "aaa"}},
		},
		{
			name: "fully applied",
			applied: map[int]appliedMigration{
				1: {version: 1, name: "init", checksum: "aaa"},
				2: {version: 2, name: "next", checksum: "bbb"},
			},
		},
		{
			name:    "an applied migration was edited",
			applied: map[int]appliedMigration{1: {version: 1, name: "init", checksum: "tampered"}},
			wantErr: ErrChecksumMismatch,
		},
		{
			name: "the database knows a migration this build does not",
			applied: map[int]appliedMigration{
				1: {version: 1, name: "init", checksum: "aaa"},
				9: {version: 9, name: "from_the_future", checksum: "zzz"},
			},
			wantErr: ErrDatabaseAhead,
		},
		{
			name: "being ahead is reported before a checksum mismatch",
			applied: map[int]appliedMigration{
				1: {version: 1, name: "init", checksum: "tampered"},
				9: {version: 9, name: "from_the_future", checksum: "zzz"},
			},
			wantErr: ErrDatabaseAhead,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := checkHistory(migrations, tc.applied)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestResolveExtensionSchema(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		opts       MigrateOptions
		info       serverInfo
		wantSchema string
		wantErr    error
	}{
		{
			name:       "not installed, unset: falls back to the default",
			wantSchema: DefaultExtensionSchema,
		},
		{
			name:       "not installed, set: honours the request",
			opts:       MigrateOptions{ExtensionSchema: "extras"},
			wantSchema: "extras",
		},
		{
			name:       "installed, unset: discovers where it actually is",
			info:       serverInfo{extensionSchema: "extras"},
			wantSchema: "extras",
		},
		{
			name:       "installed, set to the same place: agrees",
			opts:       MigrateOptions{ExtensionSchema: "extras"},
			info:       serverInfo{extensionSchema: "extras"},
			wantSchema: "extras",
		},
		{
			name:    "installed, set somewhere else: reported, not ignored",
			opts:    MigrateOptions{ExtensionSchema: "public"},
			info:    serverInfo{extensionSchema: "extras"},
			wantErr: ErrSchemaConfigMismatch,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := resolveExtensionSchema(tc.opts, tc.info)
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.ExtensionSchema != tc.wantSchema {
				t.Errorf("ExtensionSchema = %q, want %q", got.ExtensionSchema, tc.wantSchema)
			}
		})
	}
}

func TestServerInfoCheckVersion(t *testing.T) {
	t.Parallel()

	if err := (serverInfo{versionNum: MinServerVersionNum, version: "18.0"}).checkVersion(); err != nil {
		t.Errorf("PostgreSQL 18.0 should be accepted: %v", err)
	}
	if err := (serverInfo{versionNum: 190000, version: "19.1"}).checkVersion(); err != nil {
		t.Errorf("PostgreSQL 19 should be accepted: %v", err)
	}
	err := (serverInfo{versionNum: 170004, version: "17.4"}).checkVersion()
	if !errors.Is(err, ErrUnsupportedServer) {
		t.Errorf("PostgreSQL 17.4 err = %v, want ErrUnsupportedServer", err)
	}
	if !strings.Contains(err.Error(), "17.4") {
		t.Errorf("error should name the version found, got: %v", err)
	}
}

func TestShort(t *testing.T) {
	t.Parallel()

	if got := short("abc"); got != "abc" {
		t.Errorf("short of a short string = %q", got)
	}
	if got := short(strings.Repeat("a", 64)); len(got) != 12 {
		t.Errorf("short returned %d chars, want 12", len(got))
	}
}
