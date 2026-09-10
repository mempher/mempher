// Named identifiers. Each is a distinct type, so a scope cannot be passed where
// an episode id belongs.

package mempher

import (
	"fmt"

	"github.com/google/uuid"
)

// Length limits on identifiers, enforced here and again by CHECK constraints.
const (
	// MaxScopeIDLen is the longest permitted [ScopeID], in bytes.
	MaxScopeIDLen = 200
	// MaxActorLen is the longest permitted [Actor], in bytes.
	MaxActorLen = 200
	// MaxSourceLen is the longest permitted [Source], in bytes.
	MaxSourceLen = 100
)

// ScopeID names an isolated partition of memory. It is the unit of tenancy, of
// per-agent memory and of deletion, and no query crosses one.
//
// Scopes are caller-defined labels ("user:8123", "agent:planner"), created
// implicitly by the first [Memory.Append] that names them.
type ScopeID string

// String returns the scope label.
func (s ScopeID) String() string { return string(s) }

// EpisodeID is an episode's primary key, minted by PostgreSQL 18's uuidv7(). It
// embeds its own creation instant and sorts chronologically, so B-tree locality
// holds under append-heavy load and uuid_extract_timestamp() recovers the moment
// the row arrived.
//
// That instant is physical arrival time, from the database. It is not
// [Episode.IngestedAt], which comes from the injected [Clock] and is what as-of
// filtering uses; under a test clock the two diverge.
//
// Its underlying type is [16]byte, so it converts directly to [uuid.UUID].
type EpisodeID uuid.UUID

// ParseEpisodeID parses the canonical textual form of an episode id.
func ParseEpisodeID(s string) (EpisodeID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return EpisodeID{}, fmt.Errorf("mempher: parse episode id %q: %w", s, err)
	}
	return EpisodeID(id), nil
}

// String returns the canonical textual form of the id.
func (id EpisodeID) String() string { return uuid.UUID(id).String() }

// IsZero reports whether the id is unset.
func (id EpisodeID) IsZero() bool { return id == EpisodeID{} }

// MarshalText implements [encoding.TextMarshaler].
func (id EpisodeID) MarshalText() ([]byte, error) { return []byte(id.String()), nil }

// UnmarshalText implements [encoding.TextUnmarshaler].
func (id *EpisodeID) UnmarshalText(text []byte) error {
	parsed, err := ParseEpisodeID(string(text))
	if err != nil {
		return err
	}
	*id = parsed
	return nil
}

// JobID is a queued job's primary key, also a uuidv7 minted by the database.
type JobID uuid.UUID

// ParseJobID parses the canonical textual form of a job id.
func ParseJobID(s string) (JobID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return JobID{}, fmt.Errorf("mempher: parse job id %q: %w", s, err)
	}
	return JobID(id), nil
}

// String returns the canonical textual form of the id.
func (id JobID) String() string { return uuid.UUID(id).String() }

// IsZero reports whether the id is unset.
func (id JobID) IsZero() bool { return id == JobID{} }

// FactID is a fact's surrogate key, a uuidv7 minted by the database.
//
// A fact's real identity is its scope, extractor, subject, predicate, object and
// validity window, which is what the temporal primary key enforces. This id sits
// beside that so a fact can be named by something that does not move when its
// window is closed.
type FactID uuid.UUID

// ParseFactID parses the canonical textual form of a fact id.
func ParseFactID(s string) (FactID, error) {
	id, err := uuid.Parse(s)
	if err != nil {
		return FactID{}, fmt.Errorf("mempher: parse fact id %q: %w", s, err)
	}
	return FactID(id), nil
}

// String returns the canonical textual form of the id.
func (id FactID) String() string { return uuid.UUID(id).String() }

// IsZero reports whether the id is unset.
func (id FactID) IsZero() bool { return id == FactID{} }

// MarshalText implements [encoding.TextMarshaler].
func (id FactID) MarshalText() ([]byte, error) { return []byte(id.String()), nil }

// UnmarshalText implements [encoding.TextUnmarshaler].
func (id *FactID) UnmarshalText(text []byte) error {
	parsed, err := ParseFactID(string(text))
	if err != nil {
		return err
	}
	*id = parsed
	return nil
}

// ModelID identifies an embedding model precisely enough to re-embed against,
// including any version that changes the vector space. Encodings are keyed by
// it, so several models can coexist and a model change is a backfill.
type ModelID string

// String returns the model identifier.
func (m ModelID) String() string { return string(m) }

// ExtractorID identifies an extractor precisely enough to re-extract against.
// It must change when the model changes, and also when the prompt or the output
// schema does: all three decide what facts come out, and a value that survives a
// prompt rewrite silently mixes two extractors' opinions in one scope.
type ExtractorID string

// String returns the extractor identifier.
func (e ExtractorID) String() string { return string(e) }

// WorkerID identifies one worker process holding a job lease.
type WorkerID string

// String returns the worker label.
func (w WorkerID) String() string { return string(w) }

// Actor names who produced an episode within its scope, such as "user:8123"
// or "tool:web_search". It refines [Role], and is optional.
type Actor string

// String returns the actor label.
func (a Actor) String() string { return string(a) }

// Source names the system that delivered an episode, such as "cli" or "slack".
// It is
// caller-defined rather than an enumeration, and required: provenance is the
// first thing asked of a recalled episode that looks wrong.
type Source string

// String returns the source label.
func (s Source) String() string { return string(s) }

// Subject names what a fact is about, within its scope: "user", "user:8123",
// "project:apollo". It is caller- and extractor-defined rather than an
// enumeration, and it is the first axis facts are grouped and looked up by.
type Subject string

// String returns the subject label.
func (s Subject) String() string { return string(s) }

// Predicate names the relation a fact asserts: "allergic_to", "prefers",
// "lives_in". Keeping it a short, stable slug rather than free text is what
// makes two extractions of the same claim recognisably the same claim.
type Predicate string

// String returns the predicate label.
func (p Predicate) String() string { return string(p) }
