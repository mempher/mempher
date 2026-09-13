// The harness: labelled questions in, scored retrieval out.

package eval

import (
	"context"
	"fmt"
	"iter"
	"time"

	"github.com/mempher/mempher"
)

// Question is one labelled retrieval task.
type Question struct {
	// ID names the question, and names the scope its haystack is ingested
	// into.
	ID string
	// Type is the benchmark's own category, kept so that a summary can be
	// broken down by it. A number that is good overall and bad on one category
	// is the interesting case, and an average hides it.
	Type string
	// Query is what to recall with.
	Query string
	// Asked is when the question was put. It is recorded rather than applied:
	// see [Config.BoundByQuestionTime].
	Asked time.Time
	// Haystack is everything the scope holds, in the order it was recorded.
	Haystack []mempher.AppendRequest
	// Relevant indexes Haystack: the episodes that actually carry the answer.
	//
	// Indices rather than ids because the ids do not exist until the haystack
	// is ingested. Sequence numbers are allocated densely from 1 in append
	// order, so a returned episode's Seq-1 is its index here.
	Relevant []int
}

// Dataset is a labelled corpus.
type Dataset struct {
	// Name identifies it in a [Summary].
	Name string
	// Questions yields one question at a time.
	//
	// An iterator rather than a slice because a benchmark haystack runs to
	// hundreds of megabytes, and holding all of it at once costs more than
	// reading the file again would.
	Questions iter.Seq2[Question, error]
}

// Config assembles a run.
type Config struct {
	// Memory is what gets measured. Required.
	Memory *mempher.Memory
	// Store is used to erase each scope once its question has been scored, so
	// a run leaves the database as it found it. Optional: nil keeps every
	// scope, which is what to do when a failure needs looking at afterwards.
	Store mempher.Store
	// Worker, when set, drains the queue after ingesting each haystack so that
	// the semantic and fact channels have something to answer with. Nil leaves
	// projections unbuilt, which is a lexical-only run.
	Worker *mempher.Worker
	// Channels selects which channels to measure. Empty means whatever the
	// Memory would run by default.
	Channels []mempher.Channel
	// Limit is how many episodes recall returns, and the k in every metric.
	// Zero means [DefaultLimit].
	Limit int
	// BoundByQuestionTime applies [Question.Asked] as an event-time upper
	// bound, so that nothing dated after the question can answer it.
	//
	// Off by default, and the default is the honest one. In LongMemEval 6% of
	// haystack sessions are dated after the question they accompany, so
	// switching this on deletes a slice of the distractors and flatters the
	// result -- while also hiding the evidence for 43 of the 500 questions,
	// which depresses it. Neither effect is what the number is supposed to
	// describe, so the measurement is taken over the whole haystack.
	BoundByQuestionTime bool
	// Progress, when set, is called after each question. It exists because a
	// full run takes minutes and a silent one is indistinguishable from a
	// hung one.
	Progress func(done int, last Result)
}

// DefaultLimit is the k used when [Config.Limit] is zero. It matches
// [mempher.DefaultRecallLimit], so the measured number describes what a caller
// who set nothing would have got.
const DefaultLimit = mempher.DefaultRecallLimit

// Result is what one question scored.
type Result struct {
	// ID and Type identify the question.
	ID   string
	Type string
	// Relevant is how many episodes carried the answer, and Returned is how
	// many episodes recall gave back.
	Relevant int
	Returned int
	// Hit reports that at least one relevant episode came back. It is the
	// number that matters most: an agent needs the evidence in its context,
	// not all of it.
	Hit bool
	// Recall is the fraction of relevant episodes returned.
	Recall float64
	// ReciprocalRank is 1/rank of the first relevant episode, or zero if none
	// came back.
	ReciprocalRank float64
	// NDCG is normalised discounted cumulative gain over the returned
	// ranking, with binary relevance.
	NDCG float64
	// Channels is what each channel contributed, so a bad score can be told
	// apart from a broken channel.
	Channels []mempher.ChannelReport
	// Ingest is how long writing the haystack took, and Latency how long the
	// recall itself took.
	Ingest  time.Duration
	Latency time.Duration
}

// Run ingests each question's haystack, recalls against it, and scores what
// came back.
//
// Every question gets a scope of its own, named for it, so that one question's
// haystack can never answer another's. A question with no relevant episodes --
// an abstention case, where the right answer is that the memory holds nothing --
// is skipped rather than scored, because recall has no target to hit and any
// number for it would be arbitrary.
func Run(ctx context.Context, cfg Config, ds Dataset) (Summary, error) {
	if cfg.Memory == nil {
		return Summary{}, fmt.Errorf("eval: run: Memory is required: %w", mempher.ErrInvalidConfig)
	}
	limit := cfg.Limit
	if limit == 0 {
		limit = DefaultLimit
	}

	summary := Summary{Dataset: ds.Name, Limit: limit, ByType: map[string]*TypeSummary{}}
	for question, err := range ds.Questions {
		if err != nil {
			return Summary{}, err
		}
		if len(question.Relevant) == 0 {
			summary.Skipped++
			continue
		}
		result, err := run(ctx, cfg, question, limit)
		if err != nil {
			return Summary{}, err
		}
		summary.add(result)
		if cfg.Progress != nil {
			cfg.Progress(summary.Questions, result)
		}
	}
	summary.finish()
	return summary, nil
}

func run(ctx context.Context, cfg Config, q Question, limit int) (Result, error) {
	scope := mempher.ScopeID("eval:" + q.ID)

	started := time.Now()
	if err := ingest(ctx, cfg.Memory, scope, q.Haystack); err != nil {
		return Result{}, fmt.Errorf("eval: %s: ingest: %w", q.ID, err)
	}
	if cfg.Worker != nil {
		if _, err := cfg.Worker.DrainOnce(ctx); err != nil {
			return Result{}, fmt.Errorf("eval: %s: drain: %w", q.ID, err)
		}
	}
	ingested := time.Since(started)

	started = time.Now()
	request := mempher.RecallRequest{
		Scope:    scope,
		Query:    q.Query,
		Channels: cfg.Channels,
		Limit:    limit,
	}
	if cfg.BoundByQuestionTime {
		request.OccurredTo = q.Asked
	}
	recollection, err := cfg.Memory.Recall(ctx, request)
	if err != nil {
		return Result{}, fmt.Errorf("eval: %s: recall: %w", q.ID, err)
	}
	recalled := time.Since(started)

	result := score(q, recollection, limit)
	result.Channels = recollection.Channels
	result.Ingest = ingested
	result.Latency = recalled

	if cfg.Store != nil {
		if _, err := cfg.Store.Forget(ctx, mempher.ForgetRequest{Scope: scope}); err != nil {
			return Result{}, fmt.Errorf("eval: %s: erase the scope: %w", q.ID, err)
		}
	}
	return result, nil
}

// ingest writes the haystack in full batches, which is both faster and closer
// to how a real import would do it.
func ingest(
	ctx context.Context,
	mem *mempher.Memory,
	scope mempher.ScopeID,
	haystack []mempher.AppendRequest,
) error {
	for start := 0; start < len(haystack); start += mempher.MaxAppendBatch {
		end := min(start+mempher.MaxAppendBatch, len(haystack))
		batch := make([]mempher.AppendRequest, end-start)
		for i := range batch {
			batch[i] = haystack[start+i]
			batch[i].Scope = scope
		}
		if _, err := mem.AppendBatch(ctx, batch); err != nil {
			return err
		}
	}
	return nil
}
