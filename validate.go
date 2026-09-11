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
	for _, subject := range r.Subjects {
		if err := subject.Validate(); err != nil {
			return err
		}
	}
	switch {
	case r.Limit < 0:
		return fmt.Errorf("mempher: recall limit is %d: %w", r.Limit, ErrInvalidConfig)
	case r.FactLimit < 0:
		return fmt.Errorf("mempher: fact limit is %d: %w", r.FactLimit, ErrInvalidConfig)
	case r.MinFactConfidence < 0 || r.MinFactConfidence > 1:
		return fmt.Errorf("mempher: minimum fact confidence is %v, must be in [0,1]: %w",
			r.MinFactConfidence, ErrInvalidConfig)
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

// Validate reports whether the subject label is usable. Like a [ScopeID] it ends
// up in operator queries and log lines, so the same character rules apply.
func (s Subject) Validate() error {
	switch {
	case len(s) == 0:
		return fmt.Errorf("mempher: subject is empty: %w", ErrInvalidFact)
	case len(s) > MaxSubjectLen:
		return fmt.Errorf("mempher: subject is %d bytes, limit is %d: %w",
			len(s), MaxSubjectLen, ErrInvalidFact)
	case !utf8.ValidString(string(s)):
		return fmt.Errorf("mempher: subject is not valid UTF-8: %w", ErrInvalidFact)
	}
	if i := strings.IndexFunc(string(s), unicode.IsControl); i >= 0 {
		return fmt.Errorf("mempher: subject holds a control character at byte %d: %w",
			i, ErrInvalidFact)
	}
	return nil
}

// Validate reports whether the predicate label is usable.
//
// A predicate is the key that makes two extractions of one claim the same claim,
// so it is held to a tighter shape than the rest of a fact: no whitespace and no
// control characters. "lives_in" and "lives in" must not be two relations.
func (p Predicate) Validate() error {
	switch {
	case len(p) == 0:
		return fmt.Errorf("mempher: predicate is empty: %w", ErrInvalidFact)
	case len(p) > MaxPredicateLen:
		return fmt.Errorf("mempher: predicate is %d bytes, limit is %d: %w",
			len(p), MaxPredicateLen, ErrInvalidFact)
	case !utf8.ValidString(string(p)):
		return fmt.Errorf("mempher: predicate is not valid UTF-8: %w", ErrInvalidFact)
	}
	if i := strings.IndexFunc(string(p), unicode.IsSpace); i >= 0 {
		return fmt.Errorf("mempher: predicate %q holds whitespace at byte %d: %w",
			string(p), i, ErrInvalidFact)
	}
	if i := strings.IndexFunc(string(p), unicode.IsControl); i >= 0 {
		return fmt.Errorf("mempher: predicate holds a control character at byte %d: %w",
			i, ErrInvalidFact)
	}
	return nil
}

// Validate reports whether the validity window is usable: it must begin, and if
// it ends it must end after it began. A window that ends where it starts holds
// at no instant, which is a mistake rather than a way to say "never true".
func (v Validity) Validate() error {
	if v.From.IsZero() {
		return fmt.Errorf("mempher: validity window has no start: %w", ErrInvalidValidity)
	}
	if !v.IsOpen() && !v.From.Before(v.To) {
		return fmt.Errorf("mempher: validity window ends (%s) at or before it starts (%s): %w",
			v.To.Format(timeFormat), v.From.Format(timeFormat), ErrInvalidValidity)
	}
	return nil
}

// Validate reports whether the assertion can be recorded. [FactStore]
// implementations call it, so calling it yourself is only useful to reject a
// model's output earlier.
func (a Assertion) Validate() error {
	if err := a.Subject.Validate(); err != nil {
		return err
	}
	if err := a.Predicate.Validate(); err != nil {
		return err
	}
	switch {
	case len(a.Object) == 0:
		return fmt.Errorf("mempher: fact object is empty: %w", ErrInvalidFact)
	case len(a.Object) > MaxObjectLen:
		return fmt.Errorf("mempher: fact object is %d bytes, limit is %d: %w",
			len(a.Object), MaxObjectLen, ErrInvalidFact)
	case !utf8.ValidString(a.Object):
		return fmt.Errorf("mempher: fact object is not valid UTF-8: %w", ErrInvalidFact)
	}
	switch {
	case strings.TrimSpace(a.Statement) == "":
		return fmt.Errorf("mempher: fact statement is empty: %w", ErrInvalidFact)
	case len(a.Statement) > MaxStatementLen:
		return fmt.Errorf("mempher: fact statement is %d bytes, limit is %d: %w",
			len(a.Statement), MaxStatementLen, ErrInvalidFact)
	case !utf8.ValidString(a.Statement):
		return fmt.Errorf("mempher: fact statement is not valid UTF-8: %w", ErrInvalidFact)
	}
	if a.Confidence < 0 || a.Confidence > 1 {
		return fmt.Errorf("mempher: fact confidence is %v, must be in [0,1]: %w",
			a.Confidence, ErrInvalidFact)
	}
	return a.Valid.Validate()
}

// Validate reports whether the retraction can be applied. It cannot check that
// At falls after the fact's start, because only the store holds the fact; that
// is [ErrInvalidValidity] from ApplyExtraction.
func (r Retraction) Validate() error {
	if r.Fact.IsZero() {
		return fmt.Errorf("mempher: retraction names no fact: %w", ErrInvalidExtraction)
	}
	if r.At.IsZero() {
		return fmt.Errorf("mempher: retraction of fact %s has no instant: %w",
			r.Fact, ErrInvalidExtraction)
	}
	return nil
}

// Validate reports whether the extraction can be applied. [FactStore]
// implementations call it.
func (c ExtractCommand) Validate() error {
	if err := c.Scope.Validate(); err != nil {
		return err
	}
	switch {
	case c.Extractor == "":
		return fmt.Errorf("mempher: extraction names no extractor: %w", ErrInvalidExtraction)
	case len(c.Episodes) == 0:
		return fmt.Errorf("mempher: extraction names no episodes: %w", ErrInvalidExtraction)
	case len(c.Assert) > MaxAssertionsPerExtraction:
		return fmt.Errorf("mempher: extraction asserts %d facts, limit is %d: %w",
			len(c.Assert), MaxAssertionsPerExtraction, ErrInvalidExtraction)
	case len(c.Retract) > MaxRetractionsPerExtraction:
		return fmt.Errorf("mempher: extraction retracts %d facts, limit is %d: %w",
			len(c.Retract), MaxRetractionsPerExtraction, ErrInvalidExtraction)
	case c.At.IsZero():
		return fmt.Errorf("mempher: extraction has no system time: %w", ErrInvalidExtraction)
	}
	for i, episode := range c.Episodes {
		if episode.IsZero() {
			return fmt.Errorf("mempher: extraction episode %d is the zero id: %w",
				i, ErrInvalidExtraction)
		}
	}
	for i, assertion := range c.Assert {
		if err := assertion.Validate(); err != nil {
			return fmt.Errorf("mempher: extraction assertion %d: %w", i, err)
		}
	}
	for i, retraction := range c.Retract {
		if err := retraction.Validate(); err != nil {
			return fmt.Errorf("mempher: extraction retraction %d: %w", i, err)
		}
	}
	return nil
}

// Validate reports whether the fact query is usable.
func (q FactQuery) Validate() error {
	if err := q.Scope.Validate(); err != nil {
		return err
	}
	if q.Extractor == "" {
		return fmt.Errorf("mempher: fact query names no extractor: %w", ErrInvalidConfig)
	}
	for _, subject := range q.Subjects {
		if err := subject.Validate(); err != nil {
			return err
		}
	}
	for _, predicate := range q.Predicates {
		if err := predicate.Validate(); err != nil {
			return err
		}
	}
	switch {
	case q.MinConfidence < 0 || q.MinConfidence > 1:
		return fmt.Errorf("mempher: minimum confidence is %v, must be in [0,1]: %w",
			q.MinConfidence, ErrInvalidConfig)
	case q.Limit < 0:
		return fmt.Errorf("mempher: fact query limit is %d: %w", q.Limit, ErrInvalidConfig)
	case len(q.Text) > MaxContentLen:
		return fmt.Errorf("mempher: fact query text is %d bytes, limit is %d: %w",
			len(q.Text), MaxContentLen, ErrInvalidContent)
	}
	return nil
}

// Validate reports whether the request can be erased. [Memory.Forget] calls it,
// so calling it yourself is only useful to reject input earlier.
func (r ForgetRequest) Validate() error {
	if err := r.Scope.Validate(); err != nil {
		return err
	}
	if len(r.Episodes) > MaxForgetEpisodes {
		return fmt.Errorf("mempher: forget names %d episodes, limit is %d: %w",
			len(r.Episodes), MaxForgetEpisodes, ErrInvalidConfig)
	}
	for i, id := range r.Episodes {
		if id.IsZero() {
			return fmt.Errorf("mempher: forget episode %d is unset: %w", i, ErrInvalidConfig)
		}
	}
	return nil
}

// timeFormat renders instants in error messages: RFC 3339 with milliseconds.
const timeFormat = "2006-01-02T15:04:05.000Z07:00"
