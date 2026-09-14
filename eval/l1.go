// Measuring L1: whether extracted facts are true, and whether they supersede.

package eval

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"
	"unicode"

	"github.com/mempher/mempher"
)

// L1Config assembles a fact-extraction run.
type L1Config struct {
	// Memory ingests the haystack. Required.
	Memory *mempher.Memory
	// Worker drains the extract queue. Required: without one no fact is ever
	// derived and every metric here is zero.
	Worker *mempher.Worker
	// Facts reads back what was derived. Required.
	Facts mempher.FactStore
	// Extractor names whose facts to read, and is called directly for the
	// replay check. Required.
	Extractor mempher.Extractor
	// Store erases each scope once it is scored. Optional.
	Store mempher.Store
	// Replay re-runs extraction over one question's episodes and compares, to
	// see whether a second pass would assert the same thing. Off by default
	// because it doubles the model calls.
	Replay bool
	// Progress is called after each question.
	Progress func(done int, last L1Result)
}

// L1Result is what one question's extraction produced.
type L1Result struct {
	ID   string
	Type string
	// Episodes is how many were ingested, and Facts how many claims came out
	// of them. The ratio is the first thing to look at when an extractor is
	// suspected of finding something durable in every "ok, thanks".
	Episodes int
	Facts    int
	// Open is how many facts are still true; Closed how many had their window
	// ended by a later episode. Closed is the number that matters: it is
	// supersession actually happening rather than being available.
	Open   int
	Closed int
	// Contradictions is how many subject+predicate pairs are left holding more
	// than one open fact with different objects -- the world changed and the
	// extractor asserted the new claim without retracting the old, so recall
	// now returns both and neither is marked wrong.
	Contradictions int
	// AnswerCovered reports that some fact's statement carries the benchmark's
	// expected answer.
	AnswerCovered bool
	// ReplayStable reports that a second extraction over the same episodes
	// asserted the same claims. Empty when [L1Config.Replay] is off.
	ReplayStable *bool
	// Duration is how long the question took, extraction included.
	Duration time.Duration
}

// L1Summary aggregates a fact-extraction run.
type L1Summary struct {
	Dataset   string
	Questions int
	Skipped   int
	// FactsPerEpisode is how many claims an episode yielded on average. The
	// useful direction is low: most turns of a conversation carry nothing
	// durable, and an extractor that disagrees is filling a scope with noise
	// that recall then has to spend its budget on.
	FactsPerEpisode float64
	// SupersessionRate is the fraction of questions where at least one window
	// was closed. On a knowledge-update corpus, where the world changing is
	// the whole point, a low number means the temporal model is decorative.
	SupersessionRate float64
	// ContradictionRate is the fraction of questions left holding two open
	// facts that disagree. It is the failure mode supersession exists to
	// prevent, and it is what a fact channel would serve to a model as though
	// both were true.
	ContradictionRate float64
	// AnswerCoverage is the fraction whose expected answer some fact carried.
	AnswerCoverage float64
	// ReplayStability is the fraction whose second extraction matched the
	// first. Zero questions checked leaves it at zero.
	ReplayStability float64
	Replayed        int
	// ByType breaks the same numbers down by question category.
	ByType map[string]*L1TypeSummary
	// P50 and P95 are per-question wall clock, extraction included.
	P50, P95 time.Duration

	episodes, facts          int
	superseded, contradicted int
	covered, stable          int
	durations                []time.Duration
}

// L1TypeSummary is one question category's share of an [L1Summary].
type L1TypeSummary struct {
	Type              string
	Questions         int
	SupersessionRate  float64
	ContradictionRate float64
	AnswerCoverage    float64

	superseded, contradicted, covered int
}

// RunL1 ingests each question's haystack one session at a time, lets a worker
// extract from it, and scores the facts that result.
//
// Session by session, deliberately. Extraction sees one conversation and the
// facts already believed, which is the shape a deployment has and the only shape
// in which supersession can happen at all: a single batch holding the whole
// history would let the extractor reconcile everything in one pass and never
// close a window.
func RunL1(ctx context.Context, cfg L1Config, ds Dataset) (L1Summary, error) {
	switch {
	case cfg.Memory == nil:
		return L1Summary{}, fmt.Errorf("eval: run l1: Memory is required: %w", mempher.ErrInvalidConfig)
	case cfg.Worker == nil:
		return L1Summary{}, fmt.Errorf("eval: run l1: Worker is required: %w", mempher.ErrInvalidConfig)
	case cfg.Facts == nil:
		return L1Summary{}, fmt.Errorf("eval: run l1: FactStore is required: %w", mempher.ErrInvalidConfig)
	case cfg.Extractor == nil:
		return L1Summary{}, fmt.Errorf("eval: run l1: Extractor is required: %w", mempher.ErrInvalidConfig)
	}

	summary := L1Summary{Dataset: ds.Name, ByType: map[string]*L1TypeSummary{}}
	for question, err := range ds.Questions {
		if err != nil {
			return L1Summary{}, err
		}
		if question.Answer == "" {
			summary.Skipped++
			continue
		}
		result, err := runL1(ctx, cfg, question)
		if err != nil {
			return L1Summary{}, err
		}
		summary.addL1(result)
		if cfg.Progress != nil {
			cfg.Progress(summary.Questions, result)
		}
	}
	summary.finishL1()
	return summary, nil
}

func runL1(ctx context.Context, cfg L1Config, q Question) (L1Result, error) {
	scope := mempher.ScopeID("l1:" + q.ID)
	started := time.Now()

	// One session at a time, in order, draining between them: that is what
	// gives extraction a "before" to contradict.
	for _, session := range sessions(q.Haystack) {
		if err := ingest(ctx, cfg.Memory, scope, session); err != nil {
			return L1Result{}, fmt.Errorf("eval: %s: ingest: %w", q.ID, err)
		}
		if err := drain(ctx, cfg.Worker); err != nil {
			return L1Result{}, fmt.Errorf("eval: %s: drain: %w", q.ID, err)
		}
	}

	facts, err := cfg.Facts.Facts(ctx, mempher.FactQuery{
		Scope:         scope,
		Extractor:     cfg.Extractor.Model(),
		AsOf:          time.Now(),
		At:            time.Now(),
		IncludeClosed: true,
		Limit:         1000,
	})
	if err != nil {
		return L1Result{}, fmt.Errorf("eval: %s: read facts: %w", q.ID, err)
	}

	result := L1Result{
		ID:       q.ID,
		Type:     q.Type,
		Episodes: len(q.Haystack),
		Facts:    len(facts),
	}
	open := map[string][]string{}
	for _, fact := range facts {
		if fact.Valid.IsOpen() {
			result.Open++
			key := strings.ToLower(string(fact.Subject) + "\x00" + string(fact.Predicate))
			if !slices.Contains(open[key], fact.Object) {
				open[key] = append(open[key], fact.Object)
			}
		} else {
			result.Closed++
		}
		if covers(fact.Statement, q.Answer) {
			result.AnswerCovered = true
		}
	}
	for _, objects := range open {
		if len(objects) > 1 {
			result.Contradictions++
		}
	}

	if cfg.Replay {
		stable, err := replayStable(ctx, cfg, scope, q, facts)
		if err != nil {
			return L1Result{}, err
		}
		result.ReplayStable = &stable
	}
	result.Duration = time.Since(started)

	if cfg.Store != nil {
		if _, err := cfg.Store.Forget(ctx, mempher.ForgetRequest{Scope: scope}); err != nil {
			return L1Result{}, fmt.Errorf("eval: %s: erase the scope: %w", q.ID, err)
		}
	}
	return result, nil
}

// replayStable asks the extractor to read the last session again, with the same
// facts in hand, and reports whether it asserted the same claims.
//
// This is the risk [mempher.Extractor] documents: a reclaimed lease replays a
// job, and an extractor that answers differently the second time asserts the
// same claim over an overlapping window, which is a conflict rather than a
// merged answer. It is cheap to check and nothing else checks it.
func replayStable(
	ctx context.Context,
	cfg L1Config,
	scope mempher.ScopeID,
	q Question,
	known []mempher.Fact,
) (bool, error) {
	last := sessions(q.Haystack)
	if len(last) == 0 {
		return true, nil
	}
	episodes, err := cfg.Memory.Recall(ctx, mempher.RecallRequest{
		Scope: scope, Query: q.Query, Channels: []mempher.Channel{mempher.ChannelLexical}, Limit: 8,
	})
	if err != nil {
		return false, fmt.Errorf("eval: %s: replay: %w", q.ID, err)
	}
	if len(episodes.Episodes) == 0 {
		return true, nil
	}
	reread := make([]mempher.Episode, len(episodes.Episodes))
	for i, recalled := range episodes.Episodes {
		reread[i] = recalled.Episode
	}
	slices.SortFunc(reread, func(a, b mempher.Episode) int { return int(a.Seq - b.Seq) })

	first, err := cfg.Extractor.Extract(ctx, mempher.ExtractRequest{
		Scope: scope, Episodes: reread, Known: known, Now: time.Now(),
	})
	if err != nil {
		return false, fmt.Errorf("eval: %s: replay extract: %w", q.ID, err)
	}
	second, err := cfg.Extractor.Extract(ctx, mempher.ExtractRequest{
		Scope: scope, Episodes: reread, Known: known, Now: time.Now(),
	})
	if err != nil {
		return false, fmt.Errorf("eval: %s: replay extract: %w", q.ID, err)
	}
	return sameClaims(first, second), nil
}

func sameClaims(a, b mempher.ExtractResult) bool {
	render := func(r mempher.ExtractResult) []string {
		out := make([]string, 0, len(r.Assert)+len(r.Retract))
		for _, assertion := range r.Assert {
			out = append(out, fmt.Sprintf("a|%s|%s|%s", assertion.Subject, assertion.Predicate, assertion.Object))
		}
		for _, retraction := range r.Retract {
			out = append(out, fmt.Sprintf("r|%s", retraction.Fact))
		}
		slices.Sort(out)
		return out
	}
	return slices.Equal(render(a), render(b))
}

// sessions splits a haystack back into the conversations it came from, in
// order, using the binding the loader put there.
func sessions(haystack []mempher.AppendRequest) [][]mempher.AppendRequest {
	var out [][]mempher.AppendRequest
	current := ""
	for _, request := range haystack {
		id := request.Binding["session"]
		if len(out) == 0 || id != current {
			out = append(out, nil)
			current = id
		}
		out[len(out)-1] = append(out[len(out)-1], request)
	}
	return out
}

// covers reports whether a statement carries the benchmark's expected answer.
//
// Deliberately crude, and stated as such wherever the number is reported. A fact
// is a paraphrase -- "the user drives a Tesla Model 3" against an expected
// "Tesla Model 3" -- so exact matching would score a correct extraction as a
// miss. Content-word overlap at two thirds is generous in the other direction
// and will pass a fact that shares vocabulary without sharing meaning, which is
// why this is the soft metric here and supersession is the hard one.
func covers(statement, answer string) bool {
	wanted := contentWords(answer)
	if len(wanted) == 0 {
		return false
	}
	have := contentWords(statement)
	found := 0
	for word := range wanted {
		if _, ok := have[word]; ok {
			found++
		}
	}
	return float64(found)/float64(len(wanted)) >= 0.66
}

// stopWords are dropped before overlap is counted, or "the" would carry a match
// on its own.
var stopWords = map[string]bool{
	"the": true, "a": true, "an": true, "and": true, "or": true, "of": true,
	"to": true, "in": true, "is": true, "was": true, "are": true, "were": true,
	"for": true, "on": true, "at": true, "it": true, "that": true, "this": true,
	"with": true, "as": true, "by": true, "from": true, "has": true, "have": true,
}

func contentWords(text string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, field := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(field) > 2 && !stopWords[field] {
			out[field] = struct{}{}
		}
	}
	return out
}

func (s *L1Summary) addL1(r L1Result) {
	s.Questions++
	s.episodes += r.Episodes
	s.facts += r.Facts
	if r.Closed > 0 {
		s.superseded++
	}
	if r.Contradictions > 0 {
		s.contradicted++
	}
	if r.AnswerCovered {
		s.covered++
	}
	if r.ReplayStable != nil {
		s.Replayed++
		if *r.ReplayStable {
			s.stable++
		}
	}
	s.durations = append(s.durations, r.Duration)

	byType := s.ByType[r.Type]
	if byType == nil {
		byType = &L1TypeSummary{Type: r.Type}
		s.ByType[r.Type] = byType
	}
	byType.Questions++
	if r.Closed > 0 {
		byType.superseded++
	}
	if r.Contradictions > 0 {
		byType.contradicted++
	}
	if r.AnswerCovered {
		byType.covered++
	}
}

func (s *L1Summary) finishL1() {
	if s.Questions > 0 {
		n := float64(s.Questions)
		s.SupersessionRate = float64(s.superseded) / n
		s.ContradictionRate = float64(s.contradicted) / n
		s.AnswerCoverage = float64(s.covered) / n
	}
	if s.episodes > 0 {
		s.FactsPerEpisode = float64(s.facts) / float64(s.episodes)
	}
	if s.Replayed > 0 {
		s.ReplayStability = float64(s.stable) / float64(s.Replayed)
	}
	for _, byType := range s.ByType {
		if byType.Questions == 0 {
			continue
		}
		n := float64(byType.Questions)
		byType.SupersessionRate = float64(byType.superseded) / n
		byType.ContradictionRate = float64(byType.contradicted) / n
		byType.AnswerCoverage = float64(byType.covered) / n
	}
	slices.Sort(s.durations)
	s.P50 = percentile(s.durations, 0.50)
	s.P95 = percentile(s.durations, 0.95)
}

// String renders the summary as the table a run is worth reporting as.
func (s L1Summary) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s -- L1 -- %d questions scored, %d skipped\n\n", s.Dataset, s.Questions, s.Skipped)
	fmt.Fprintf(&b, "  %-26s %5s %11s %13s %9s\n",
		"category", "n", "superseded", "contradicted", "answered")
	fmt.Fprintf(&b, "  %s\n", strings.Repeat("-", 70))
	fmt.Fprintf(&b, "  %-26s %5d %11.3f %13.3f %9.3f\n",
		"overall", s.Questions, s.SupersessionRate, s.ContradictionRate, s.AnswerCoverage)
	for _, name := range slices.Sorted(maps.Keys(s.ByType)) {
		t := s.ByType[name]
		fmt.Fprintf(&b, "  %-26s %5d %11.3f %13.3f %9.3f\n",
			name, t.Questions, t.SupersessionRate, t.ContradictionRate, t.AnswerCoverage)
	}
	fmt.Fprintf(&b, "\n  %.3f facts per episode (%d facts from %d episodes)\n",
		s.FactsPerEpisode, s.facts, s.episodes)
	if s.Replayed > 0 {
		fmt.Fprintf(&b, "  replay asserted the same claims on %.3f of %d checked\n",
			s.ReplayStability, s.Replayed)
	}
	fmt.Fprintf(&b, "  per question: p50 %s, p95 %s\n",
		s.P50.Round(time.Millisecond), s.P95.Round(time.Millisecond))
	return b.String()
}
