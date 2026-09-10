// L0: the immutable episode log, the only source of truth.

package mempher

import (
	"fmt"
	"time"
)

// Content and binding limits, enforced here and again by CHECK constraints.
const (
	// MaxContentLen is the longest permitted episode content, in bytes.
	MaxContentLen = 1 << 20
	// MaxBindingKeys is the largest number of entries in a [Binding].
	MaxBindingKeys = 32
	// MaxBindingKeyLen is the longest permitted [Binding] key, in bytes.
	MaxBindingKeyLen = 64
	// MaxBindingValueLen is the longest permitted [Binding] value, in bytes.
	MaxBindingValueLen = 512
)

// Role is the kind of participant an episode came from. The zero Role is
// invalid, so an [AppendRequest] must state one.
//
// The set is closed and mirrored by a CHECK constraint. The values are readable
// strings rather than integers because a memory store gets debugged by hand in
// psql, and "assistant" beats 2.
type Role string

// The roles an episode may carry.
const (
	// RoleUser is input from the human the agent is serving.
	RoleUser Role = "user"
	// RoleAssistant is output the agent produced.
	RoleAssistant Role = "assistant"
	// RoleSystem is instruction or configuration given to the agent.
	RoleSystem Role = "system"
	// RoleTool is the result of a tool or function call.
	RoleTool Role = "tool"
	// RoleObservation is a non-conversational event the agent witnessed.
	RoleObservation Role = "observation"
)

// Valid reports whether r is one of the defined roles.
func (r Role) Valid() bool {
	switch r {
	case RoleUser, RoleAssistant, RoleSystem, RoleTool, RoleObservation:
		return true
	default:
		return false
	}
}

// String returns the role name.
func (r Role) String() string { return string(r) }

// MarshalText implements [encoding.TextMarshaler].
func (r Role) MarshalText() ([]byte, error) {
	if !r.Valid() {
		return nil, fmt.Errorf("mempher: marshal role %q: %w", string(r), ErrInvalidRole)
	}
	return []byte(r), nil
}

// UnmarshalText implements [encoding.TextUnmarshaler], rejecting any value
// outside the defined set.
func (r *Role) UnmarshalText(text []byte) error {
	candidate := Role(text)
	if !candidate.Valid() {
		return fmt.Errorf("mempher: unmarshal role %q: %w", string(text), ErrInvalidRole)
	}
	*r = candidate
	return nil
}

// Binding is the contextual features bound to an episode when it was recorded:
// the session, the task in flight, the thread. It is what later lets recall be
// cued by context rather than content alone.
//
// It is flat and string-to-string on purpose: that shape indexes as one GIN
// containment index, keeps the key space small enough to reason about, and keeps
// map[string]any out of this API. Nest structured payloads in the content.
type Binding map[string]string

// Clone returns an independent copy. A nil Binding clones to nil.
func (b Binding) Clone() Binding {
	if b == nil {
		return nil
	}
	out := make(Binding, len(b))
	for k, v := range b {
		out[k] = v
	}
	return out
}

// Episode is one immutable record of something that happened: the atom of L0.
// Once returned by [Memory.Append] its contents never change.
type Episode struct {
	// ID is the uuidv7 primary key, minted by the database.
	ID EpisodeID
	// Scope is the partition this episode belongs to.
	Scope ScopeID
	// Seq is its position in the scope's log, starting at 1 and allocated
	// densely. It is what consolidation replays in order.
	//
	// Allocated densely, not necessarily dense for ever: erasing an episode
	// leaves the number it held unused, and no later episode takes it. Nothing
	// reads Seq as a count.
	Seq int64
	// Content is the episode text, verbatim.
	Content string
	// Role is the kind of participant this came from.
	Role Role
	// Actor is who produced it, if the caller said.
	Actor Actor
	// Source is the system that delivered it.
	Source Source
	// OccurredAt is event time: when the thing happened in the world.
	OccurredAt time.Time
	// IngestedAt is system time, from the [Clock]. All as-of filtering uses
	// this, not the instant embedded in ID.
	IngestedAt time.Time
	// Binding is the context bound at recording time.
	Binding Binding
}

// NewEpisode is a validated episode ready to persist. Both timestamps are
// already decided: a [Store] never reads a clock, so a replay produces identical
// rows.
type NewEpisode struct {
	// Scope is the partition to append to, created if new.
	Scope ScopeID
	// Content is the episode text.
	Content string
	// Role is the kind of participant this came from.
	Role Role
	// Actor is who produced it; may be empty.
	Actor Actor
	// Source is the system that delivered it.
	Source Source
	// OccurredAt is event time, already resolved.
	OccurredAt time.Time
	// IngestedAt is system time, already resolved.
	IngestedAt time.Time
	// Binding is the context to bind; may be nil.
	Binding Binding
}
