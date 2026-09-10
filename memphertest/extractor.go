package memphertest

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/mempher/mempher"
)

// Rule is one thing an [Extractor] knows how to notice.
//
// It is deliberately a substring match rather than anything cleverer. A test
// double for a model should be obvious about what it will do, so that a test
// which fails says something about the code under test rather than about the
// double's own matching.
type Rule struct {
	// Match is the substring that fires the rule, compared case-insensitively
	// against episode content.
	Match string
	// Assert is the claim to make. A zero Valid.From is filled in with the
	// matching episode's OccurredAt, which is what an extractor with no better
	// information should do.
	Assert mempher.Assertion
	// Replaces retracts any known fact with this rule's subject and predicate
	// but a different object, as of the matching episode's OccurredAt.
	//
	// It is how a test exercises the case the whole temporal design exists
	// for: the world changed, the old claim was true and stays answerable, and
	// the new one takes over from the moment it became true.
	Replaces bool
}

// Extractor is a deterministic [mempher.Extractor] that needs no model provider.
//
// It applies a fixed set of [Rule]s to episode content. That is enough to
// exercise everything L1 promises -- assertion, re-assertion, supersession by
// retraction, and finding nothing at all -- while producing the same output on
// every run, which a real extractor at any temperature does not.
//
// Extractor is safe for concurrent use.
type Extractor struct {
	id mempher.ExtractorID

	mu      sync.RWMutex
	rules   []Rule
	failure error
	calls   int
}

// NewExtractor returns an Extractor with no rules, which finds nothing. That is
// the useful default: most episodes yield no durable fact, and a test asserting
// that Recall stays empty should not have to arrange it.
func NewExtractor() *Extractor {
	return &Extractor{id: "memphertest/rules"}
}

// WithID returns a copy carrying a different extractor id and the same rules, so
// a test can put two extractors over one scope and check that their facts stay
// apart. It panics on an empty id, which is a bug in the test.
func (e *Extractor) WithID(id mempher.ExtractorID) *Extractor {
	if id == "" {
		panic("memphertest: WithID: extractor id must not be empty")
	}
	e.mu.RLock()
	defer e.mu.RUnlock()
	return &Extractor{id: id, rules: append([]Rule(nil), e.rules...)}
}

// On registers a rule and returns the Extractor, so rules chain.
func (e *Extractor) On(rules ...Rule) *Extractor {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.rules = append(e.rules, rules...)
	return e
}

// Model identifies this extractor.
func (e *Extractor) Model() mempher.ExtractorID { return e.id }

// Fail makes every later call return err, so a test can exercise the paths that
// only run when extraction is unavailable, such as a retried or dead job. Pass
// nil to stop failing.
func (e *Extractor) Fail(err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.failure = err
}

// Calls reports how many extraction calls have been made, which is how a test
// asserts that the write path made none, or that a replayed job made no second
// one.
func (e *Extractor) Calls() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.calls
}

// Extract applies the rules to each episode in order.
func (e *Extractor) Extract(
	ctx context.Context,
	req mempher.ExtractRequest,
) (mempher.ExtractResult, error) {
	if err := ctx.Err(); err != nil {
		return mempher.ExtractResult{}, fmt.Errorf("memphertest: extract: %w", err)
	}

	e.mu.Lock()
	e.calls++
	rules := append([]Rule(nil), e.rules...)
	failure := e.failure
	e.mu.Unlock()

	if failure != nil {
		return mempher.ExtractResult{}, fmt.Errorf("memphertest: extract: %w", failure)
	}

	var out mempher.ExtractResult
	retracted := make(map[mempher.FactID]struct{})
	for _, episode := range req.Episodes {
		content := strings.ToLower(episode.Content)
		for _, rule := range rules {
			if !strings.Contains(content, strings.ToLower(rule.Match)) {
				continue
			}

			assertion := rule.Assert
			if assertion.Valid.From.IsZero() {
				assertion.Valid.From = episode.OccurredAt
			}
			out.Assert = append(out.Assert, assertion)

			if !rule.Replaces {
				continue
			}
			// Only a different object is a contradiction. The same claim
			// asserted again is a re-assertion, and retracting it here would
			// close the very window the assertion above is about to open.
			for _, known := range req.Known {
				if known.Subject != assertion.Subject ||
					known.Predicate != assertion.Predicate ||
					known.Object == assertion.Object {
					continue
				}
				if _, done := retracted[known.ID]; done {
					continue
				}
				retracted[known.ID] = struct{}{}
				out.Retract = append(out.Retract, mempher.Retraction{
					Fact: known.ID,
					At:   assertion.Valid.From,
				})
			}
		}
	}
	return out, nil
}

var _ mempher.Extractor = (*Extractor)(nil)
