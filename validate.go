// Input validation. Every rule here is mirrored by a CHECK constraint, so a bad
// row cannot arrive by another path; these run first only so the error is
// legible.

package mempher

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Validate reports whether the scope id is usable: non-empty, within
// [MaxScopeIDLen] bytes, valid UTF-8, and free of control characters. A scope id
// ends up in log lines and operator queries, where an embedded newline or escape
// sequence is at best confusing.
func (s ScopeID) Validate() error {
	switch {
	case len(s) == 0:
		return fmt.Errorf("mempher: scope id is empty: %w", ErrInvalidScope)
	case len(s) > MaxScopeIDLen:
		return fmt.Errorf("mempher: scope id is %d bytes, limit is %d: %w",
			len(s), MaxScopeIDLen, ErrInvalidScope)
	case !utf8.ValidString(string(s)):
		return fmt.Errorf("mempher: scope id is not valid UTF-8: %w", ErrInvalidScope)
	}
	if i := strings.IndexFunc(string(s), unicode.IsControl); i >= 0 {
		return fmt.Errorf("mempher: scope id holds a control character at byte %d: %w",
			i, ErrInvalidScope)
	}
	return nil
}

// Validate reports whether the actor label is usable. An empty actor is valid:
// it means the caller did not say who produced the episode.
func (a Actor) Validate() error {
	if len(a) > MaxActorLen {
		return fmt.Errorf("mempher: actor is %d bytes, limit is %d: %w",
			len(a), MaxActorLen, ErrInvalidActor)
	}
	if !utf8.ValidString(string(a)) {
		return fmt.Errorf("mempher: actor is not valid UTF-8: %w", ErrInvalidActor)
	}
	return nil
}

// Validate reports whether the source label is usable. Unlike an actor, a source
// is required: the episode whose provenance is unrecorded is the one you will
// want to trace.
func (s Source) Validate() error {
	switch {
	case len(s) == 0:
		return fmt.Errorf("mempher: source is empty: %w", ErrInvalidSource)
	case len(s) > MaxSourceLen:
		return fmt.Errorf("mempher: source is %d bytes, limit is %d: %w",
			len(s), MaxSourceLen, ErrInvalidSource)
	case !utf8.ValidString(string(s)):
		return fmt.Errorf("mempher: source is not valid UTF-8: %w", ErrInvalidSource)
	}
	return nil
}

// Validate reports whether the binding is usable: at most [MaxBindingKeys]
// entries, each key 1 to [MaxBindingKeyLen] bytes and each value at most
// [MaxBindingValueLen] bytes, all valid UTF-8. A nil binding is valid.
//
// The limits are deliberately tight: a binding is a retrieval cue, not a
// document store.
func (b Binding) Validate() error {
	if len(b) > MaxBindingKeys {
		return fmt.Errorf("mempher: binding has %d keys, limit is %d: %w",
			len(b), MaxBindingKeys, ErrInvalidBinding)
	}
	for key, value := range b {
		switch {
		case len(key) == 0:
			return fmt.Errorf("mempher: binding has an empty key: %w", ErrInvalidBinding)
		case len(key) > MaxBindingKeyLen:
			return fmt.Errorf("mempher: binding key %q is %d bytes, limit is %d: %w",
				key, len(key), MaxBindingKeyLen, ErrInvalidBinding)
		case len(value) > MaxBindingValueLen:
			return fmt.Errorf("mempher: binding value for %q is %d bytes, limit is %d: %w",
				key, len(value), MaxBindingValueLen, ErrInvalidBinding)
		case !utf8.ValidString(key) || !utf8.ValidString(value):
			return fmt.Errorf("mempher: binding entry %q is not valid UTF-8: %w",
				key, ErrInvalidBinding)
		}
	}
	return nil
}

// Validate reports whether the request can be appended. [Memory.Append] calls
// it, so calling it yourself is only useful to reject input earlier.
func (r AppendRequest) Validate() error {
	if err := r.Scope.Validate(); err != nil {
		return err
	}
	switch {
	case len(r.Content) == 0:
		return fmt.Errorf("mempher: episode content is empty: %w", ErrInvalidContent)
	case len(r.Content) > MaxContentLen:
		return fmt.Errorf("mempher: episode content is %d bytes, limit is %d: %w",
			len(r.Content), MaxContentLen, ErrInvalidContent)
	case !utf8.ValidString(r.Content):
		return fmt.Errorf("mempher: episode content is not valid UTF-8: %w", ErrInvalidContent)
	}
	if !r.Role.Valid() {
		return fmt.Errorf("mempher: role %q: %w", string(r.Role), ErrInvalidRole)
	}
	if err := r.Actor.Validate(); err != nil {
		return err
	}
	if err := r.Source.Validate(); err != nil {
		return err
	}
	return r.Binding.Validate()
}

// Validate reports whether the request can be recalled. [Memory.Recall] calls
// it, so calling it yourself is only useful to reject input earlier.
func (r RecallRequest) Validate() error {
	if err := r.Scope.Validate(); err != nil {
		return err
	}
	if strings.TrimSpace(r.Query) == "" {
		return fmt.Errorf("mempher: recall query is empty: %w", ErrInvalidContent)
	}
	if len(r.Query) > MaxContentLen {
		return fmt.Errorf("mempher: recall query is %d bytes, limit is %d: %w",
			len(r.Query), MaxContentLen, ErrInvalidContent)
	}
	for _, ch := range r.Channels {
		if !ch.Valid() {
			return fmt.Errorf("mempher: channel %q: %w", string(ch), ErrInvalidChannel)
		}
	}
	for _, role := range r.Roles {
		if !role.Valid() {
			return fmt.Errorf("mempher: role filter %q: %w", string(role), ErrInvalidRole)
		}
	}
	switch {
	case r.Limit < 0:
		return fmt.Errorf("mempher: recall limit is %d: %w", r.Limit, ErrInvalidConfig)
	case r.CandidatesPerChannel < 0:
		return fmt.Errorf("mempher: candidates per channel is %d: %w",
			r.CandidatesPerChannel, ErrInvalidConfig)
	case r.MaxTokens < 0:
		return fmt.Errorf("mempher: max tokens is %d: %w", r.MaxTokens, ErrInvalidConfig)
	}
	// An inverted window matches nothing, which is far more likely to be
	// swapped arguments than an intentional empty query.
	if !r.OccurredFrom.IsZero() && !r.OccurredTo.IsZero() && r.OccurredTo.Before(r.OccurredFrom) {
		return fmt.Errorf("mempher: event-time window ends (%s) before it starts (%s): %w",
			r.OccurredTo.Format(timeFormat), r.OccurredFrom.Format(timeFormat), ErrInvalidConfig)
	}
	return r.Binding.Validate()
}

// timeFormat renders instants in error messages: RFC 3339 with milliseconds.
const timeFormat = "2006-01-02T15:04:05.000Z07:00"
