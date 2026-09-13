// Reading the model's answer, and refusing the parts of it a FactStore would.

package extract

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/mempher/mempher"
)

// extraction is a reply to the extraction call, in the shape the schema asks
// for.
type extraction struct {
	Assertions  []jsonAssertion  `json:"assertions"`
	Retractions []jsonRetraction `json:"retractions"`
}

// topics is a reply to the first pass of [Options.Reconcile].
type topics struct {
	Subjects   []string `json:"subjects"`
	Predicates []string `json:"predicates"`
}

type jsonAssertion struct {
	Subject   string `json:"subject"`
	Predicate string `json:"predicate"`
	Object    string `json:"object"`
	Statement string `json:"statement"`
	ValidFrom string `json:"valid_from"`
	ValidTo   string `json:"valid_to"`
	// A pointer, because a missing confidence must not arrive as zero.
	// [mempher.Assertion.Confidence] takes zero at face value -- it means no
	// confidence at all, not "unset" -- so an omission would otherwise record
	// every fact as one the extractor believed worthless.
	Confidence *float64 `json:"confidence"`
}

type jsonRetraction struct {
	FactID string `json:"fact_id"`
	At     string `json:"at"`
}

// decode reads a provider's reply. Unknown fields are tolerated: a provider that
// wraps the answer in a key of its own is something to read around, not a job to
// dead-letter.
func decode[T any](raw []byte) (T, error) {
	var out T
	if err := json.Unmarshal(raw, &out); err != nil {
		return out, fmt.Errorf("mempher/extract: decode %d bytes of response: %w: %w",
			len(raw), err, ErrMalformedResponse)
	}
	return out, nil
}

// validate turns a decoded reply into what a [mempher.FactStore] will accept.
//
// Every rejection here drops one item and logs why, rather than failing the
// extraction. A model returns a bad triple beside good ones often enough that
// the alternative -- losing the whole turn to one hallucinated predicate, and
// re-losing it on every retry until the job dies -- costs more than it saves.
func (e *Extractor) validate(
	ctx context.Context,
	body extraction,
	req mempher.ExtractRequest,
) mempher.ExtractResult {
	return mempher.ExtractResult{
		Assert:  e.assertions(ctx, body.Assertions, req.Now),
		Retract: e.retractions(ctx, body.Retractions, req),
	}
}

func (e *Extractor) assertions(
	ctx context.Context,
	in []jsonAssertion,
	now time.Time,
) []mempher.Assertion {
	if len(in) == 0 {
		return nil
	}
	out := make([]mempher.Assertion, 0, len(in))
	for _, raw := range in {
		if len(out) >= e.opts.MaxAssertions {
			e.log.WarnContext(ctx, "mempher/extract: assertions over the cap were dropped",
				"cap", e.opts.MaxAssertions, "returned", len(in))
			break
		}
		a, err := e.assertion(raw, now)
		if err != nil {
			e.log.WarnContext(ctx, "mempher/extract: dropped an assertion",
				"reason", err,
				"subject", raw.Subject, "predicate", raw.Predicate, "object", raw.Object)
			continue
		}
		out = append(out, a)
	}
	return out
}

// assertion builds one claim and holds it to everything a FactStore would,
// plus the one rule only this package knows: the configured vocabulary.
func (e *Extractor) assertion(raw jsonAssertion, now time.Time) (mempher.Assertion, error) {
	if raw.Confidence == nil {
		return mempher.Assertion{}, fmt.Errorf(
			"no confidence was given: %w", mempher.ErrInvalidFact)
	}
	from, err := instant(raw.ValidFrom, now)
	if err != nil {
		return mempher.Assertion{}, fmt.Errorf("valid_from: %w", err)
	}
	var to time.Time
	if strings.TrimSpace(raw.ValidTo) != "" {
		if to, err = instant(raw.ValidTo, now); err != nil {
			return mempher.Assertion{}, fmt.Errorf("valid_to: %w", err)
		}
	}

	a := mempher.Assertion{
		Subject:    mempher.Subject(strings.TrimSpace(raw.Subject)),
		Predicate:  mempher.Predicate(strings.TrimSpace(raw.Predicate)),
		Object:     strings.TrimSpace(raw.Object),
		Statement:  strings.TrimSpace(raw.Statement),
		Valid:      mempher.Validity{From: from, To: to},
		Confidence: float32(*raw.Confidence),
	}
	if e.predicates != nil {
		if _, ok := e.predicates[a.Predicate]; !ok {
			return mempher.Assertion{}, fmt.Errorf(
				"predicate %q is outside the configured vocabulary: %w",
				a.Predicate, mempher.ErrInvalidFact)
		}
	}
	if err := a.Validate(); err != nil {
		return mempher.Assertion{}, err
	}
	return a, nil
}

func (e *Extractor) retractions(
	ctx context.Context,
	in []jsonRetraction,
	req mempher.ExtractRequest,
) []mempher.Retraction {
	if len(in) == 0 {
		return nil
	}
	// Checked against every fact the worker supplied, not against the narrowed
	// set an Options.Reconcile pass chose: a fact id is either real or invented,
	// and which facts the model was shown does not change which.
	known := make(map[mempher.FactID]mempher.Fact, len(req.Known))
	for _, f := range req.Known {
		known[f.ID] = f
	}

	out := make([]mempher.Retraction, 0, len(in))
	for _, raw := range in {
		if len(out) >= mempher.MaxRetractionsPerExtraction {
			e.log.WarnContext(ctx, "mempher/extract: retractions over the cap were dropped",
				"cap", mempher.MaxRetractionsPerExtraction, "returned", len(in))
			break
		}
		r, err := retraction(raw, known, req.Now)
		if err != nil {
			e.log.WarnContext(ctx, "mempher/extract: dropped a retraction",
				"reason", err, "fact", raw.FactID, "at", raw.At)
			continue
		}
		out = append(out, r)
	}
	return out
}

func retraction(
	raw jsonRetraction,
	known map[mempher.FactID]mempher.Fact,
	now time.Time,
) (mempher.Retraction, error) {
	id, err := mempher.ParseFactID(strings.TrimSpace(raw.FactID))
	if err != nil {
		return mempher.Retraction{}, fmt.Errorf(
			"fact_id %q is not an id: %w", raw.FactID, mempher.ErrInvalidExtraction)
	}
	fact, ok := known[id]
	if !ok {
		// A fact the model was never shown is a fact it invented, and closing a
		// window by a guessed id would close the wrong claim.
		return mempher.Retraction{}, fmt.Errorf(
			"fact %s is not among the known facts: %w", id, mempher.ErrInvalidExtraction)
	}
	at, err := instant(raw.At, now)
	if err != nil {
		return mempher.Retraction{}, fmt.Errorf("at: %w", err)
	}
	if at.Before(fact.Valid.From) {
		// Clamping to Valid.From would invent a window the world never had, and
		// the store would accept it without complaint.
		return mempher.Retraction{}, fmt.Errorf(
			"fact %s would close at %s, before it began at %s: %w",
			id, at.Format(time.RFC3339), fact.Valid.From.Format(time.RFC3339),
			mempher.ErrInvalidValidity)
	}

	r := mempher.Retraction{Fact: id, At: at}
	if err := r.Validate(); err != nil {
		return mempher.Retraction{}, err
	}
	return r, nil
}

// instant parses a date or a timestamp, as the schema asks for either.
//
// A bare date resolves at midnight in now's location. Natural language yields
// dates rather than timestamps, and demanding RFC 3339 would only make the model
// invent a time of day; taking the zone from the injected instant is what keeps
// that invention out of the data and out of a replay.
func instant(s string, now time.Time) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, fmt.Errorf("no instant was given: %w", mempher.ErrInvalidValidity)
	}
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t, nil
	}
	if t, err := time.ParseInLocation(time.DateOnly, s, now.Location()); err == nil {
		return t, nil
	}
	return time.Time{}, fmt.Errorf(
		"%q is neither a date nor an RFC 3339 timestamp: %w", s, mempher.ErrInvalidValidity)
}
