// Memory: the public entry point, wiring the write and read loops.

package mempher

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"
)

// Config assembles a [Memory] from its ports. Store and Embedder are required;
// everything else has a default.
type Config struct {
	// Store persists L0 and its projections. Required.
	Store Store
	// Queue holds deferred work. Optional: when nil, and Store also
	// implements [Queue], the store is used for both. One backend normally
	// serves both, and the interfaces stay separate so that a caller who
	// wants them apart can still split them.
	Queue Queue
	// Embedder turns text into vectors. Required. [Memory.Append] never
	// calls it; only [Memory.Recall] and [Worker] do.
	Embedder Embedder
	// Extractor turns episodes into facts. Optional: nil leaves L1 switched
	// off, so [Memory.Append] enqueues no extract job and [Memory.Recall]
	// returns no facts. [Memory] never calls it -- only a [Worker] does --
	// but it is named here because recall reads the facts it is keyed by.
	Extractor Extractor
	// FactStore persists L1. Optional: when nil, and Store also implements
	// [FactStore], the store is used for both. Required once Extractor is
	// set, and meaningless without it.
	FactStore FactStore
	// Clock is the source of time. Nil means [SystemClock].
	Clock Clock
	// TokenCounter measures returned content. Nil means
	// [ApproxTokenCounter].
	TokenCounter TokenCounter
	// Fusion tunes reciprocal rank fusion. The zero value uses
	// [DefaultFusionK] and equal channel weights.
	Fusion FusionOptions
	// DefaultLimit is how many episodes [Memory.Recall] returns when a
	// request does not say. Zero means [DefaultRecallLimit].
	DefaultLimit int
}

// Memory is the public entry point: the write and read loops over one [Store].
// It is safe for concurrent use and holds no mutable global state.
type Memory struct {
	store        Store
	queue        Queue
	embedder     Embedder
	extractor    Extractor
	facts        FactStore
	clock        Clock
	tokens       TokenCounter
	fusion       FusionOptions
	defaultLimit int
}

// New assembles a [Memory] from cfg, resolving defaults and rejecting a
// configuration that cannot work.
//
// It performs no I/O, so it needs no context. Whether the database agrees with
// the configured embedder is the adapter's business, checked at startup there.
func New(cfg Config) (*Memory, error) {
	if cfg.Store == nil {
		return nil, fmt.Errorf("mempher: new: Store is required: %w", ErrInvalidConfig)
	}
	if cfg.Embedder == nil {
		return nil, fmt.Errorf("mempher: new: Embedder is required: %w", ErrInvalidConfig)
	}
	if dims := cfg.Embedder.Dimensions(); dims <= 0 {
		return nil, fmt.Errorf("mempher: new: embedder reports %d dimensions: %w", dims, ErrInvalidConfig)
	}
	if cfg.Embedder.Model() == "" {
		return nil, fmt.Errorf("mempher: new: embedder reports no model id: %w", ErrInvalidConfig)
	}

	queue := cfg.Queue
	if queue == nil {
		asQueue, ok := cfg.Store.(Queue)
		if !ok {
			return nil, fmt.Errorf(
				"mempher: new: Queue is required because %T does not implement mempher.Queue: %w",
				cfg.Store, ErrInvalidConfig)
		}
		queue = asQueue
	}

	facts, err := resolveFactStore(cfg.Store, cfg.FactStore, cfg.Extractor, "new")
	if err != nil {
		return nil, err
	}
	if cfg.Extractor != nil && cfg.Extractor.Model() == "" {
		return nil, fmt.Errorf("mempher: new: extractor reports no model id: %w", ErrInvalidConfig)
	}

	fusion, err := cfg.Fusion.resolve()
	if err != nil {
		return nil, fmt.Errorf("mempher: new: %w", err)
	}
	if cfg.DefaultLimit < 0 {
		return nil, fmt.Errorf("mempher: new: DefaultLimit is %d: %w", cfg.DefaultLimit, ErrInvalidConfig)
	}

	m := &Memory{
		store:        cfg.Store,
		queue:        queue,
		embedder:     cfg.Embedder,
		extractor:    cfg.Extractor,
		facts:        facts,
		clock:        cfg.Clock,
		tokens:       cfg.TokenCounter,
		fusion:       fusion,
		defaultLimit: cfg.DefaultLimit,
	}
	if m.clock == nil {
		m.clock = SystemClock{}
	}
	if m.tokens == nil {
		m.tokens = ApproxTokenCounter{}
	}
	if m.defaultLimit == 0 {
		m.defaultLimit = DefaultRecallLimit
	}
	return m, nil
}

// Append records one episode and returns it as persisted.
//
// This is the write loop: validate, insert the episode, enqueue the work it
// implies in the same statement, return. It makes no model call, so its latency
// is one round trip. The episode is lexically searchable the moment Append
// returns, semantically searchable once a [Worker] drains its encode job, and
// has yielded whatever facts it holds once a worker drains its extract job.
//
// Append never modifies an existing episode. Recording a correction means
// appending another one.
func (m *Memory) Append(ctx context.Context, req AppendRequest) (Episode, error) {
	if err := req.Validate(); err != nil {
		return Episode{}, err
	}

	now := m.clock.Now()
	occurred := req.OccurredAt
	if occurred.IsZero() {
		occurred = now
	}

	result, err := m.store.Append(ctx, AppendCommand{
		Episodes: []NewEpisode{{
			Scope:      req.Scope,
			Content:    req.Content,
			Role:       req.Role,
			Actor:      req.Actor,
			Source:     req.Source,
			OccurredAt: occurred,
			IngestedAt: now,
			// Cloned, so a caller reusing its map cannot change an episode
			// that has already been recorded.
			Binding: req.Binding.Clone(),
		}},
		// What this episode implies. They commit with it, so neither can be
		// lost to a crash between two statements.
		Jobs: m.jobsFor(req.Scope),
	})
	if err != nil {
		return Episode{}, err
	}
	return result.Episodes[0], nil
}

// AppendBatch records several episodes of one scope in a single round trip, and
// returns them as persisted, in the order given.
//
// It is [Memory.Append] for the case a caller already has more than one episode
// in hand -- a conversation turn that produced a question, a tool call and an
// answer, or a page of an import. Everything true of Append is true here: no
// model call, one round trip, and the episodes plus the work they imply commit
// together or not at all.
//
// What it buys is amortisation, and the saving is mostly not in this library.
// One round trip and one commit are shared by the whole batch, and so is the
// consolidation that follows: a batch yields one encode job naming all of its
// episodes, which a [Worker] satisfies with one call to
// [Embedder.EmbedDocuments] rather than one per episode. That is where the money
// goes.
//
// Every request must name the same scope. Sequence allocation is a lock on that
// scope's row, so a batch spanning two scopes would hold two of them at once and
// two such batches in opposite orders would deadlock. Appending across scopes is
// a loop of these calls, which holds one lock at a time.
//
// The episodes are sequenced in the order given and share one ingest instant:
// they were recorded together, so an as-of cut that returns one of them returns
// all of them.
func (m *Memory) AppendBatch(ctx context.Context, reqs []AppendRequest) ([]Episode, error) {
	switch {
	case len(reqs) == 0:
		return nil, fmt.Errorf("mempher: append batch: no episodes: %w", ErrInvalidConfig)
	case len(reqs) > MaxAppendBatch:
		return nil, fmt.Errorf("mempher: append batch holds %d episodes, limit is %d: %w",
			len(reqs), MaxAppendBatch, ErrInvalidConfig)
	}

	now := m.clock.Now()
	scope := reqs[0].Scope
	episodes := make([]NewEpisode, len(reqs))
	for i, req := range reqs {
		if err := req.Validate(); err != nil {
			return nil, fmt.Errorf("mempher: append batch: episode %d: %w", i, err)
		}
		if req.Scope != scope {
			return nil, fmt.Errorf(
				"mempher: append batch: episode %d names scope %q, not %q: a batch is one "+
					"scope, because sequencing locks its row: %w",
				i, req.Scope, scope, ErrInvalidConfig)
		}
		occurred := req.OccurredAt
		if occurred.IsZero() {
			occurred = now
		}
		episodes[i] = NewEpisode{
			Scope:      req.Scope,
			Content:    req.Content,
			Role:       req.Role,
			Actor:      req.Actor,
			Source:     req.Source,
			OccurredAt: occurred,
			IngestedAt: now,
			Binding:    req.Binding.Clone(),
		}
	}

	result, err := m.store.Append(ctx, AppendCommand{
		Episodes: episodes,
		Jobs:     m.jobsFor(scope),
	})
	if err != nil {
		return nil, err
	}
	return result.Episodes, nil
}

// Recall returns the episodes most worth putting in front of a model, best
// first, each with the provenance of why it was chosen.
//
// This is the read loop: embed the query, run the requested channels
// concurrently with their filters pushed into SQL, fuse the rankings, cut to the
// limit and token budget, return. No model call beyond the query embedding.
//
// A failed channel is reported in [Recollection.Channels] and the rest still
// answer. Recall fails with [ErrAllChannelsFailed] only when none succeeded, so
// a thin answer is never mistaken for an empty memory.
//
// The result is deterministic: channels are fused from fixed slots, and ties
// break towards the newer episode at every stage. The one exception is a set of
// episodes tied on score exactly at a channel's own limit, where which of them
// the database returned is arbitrary.
func (m *Memory) Recall(ctx context.Context, req RecallRequest) (Recollection, error) {
	if err := req.Validate(); err != nil {
		return Recollection{}, err
	}

	asOf := req.AsOf
	if asOf.IsZero() {
		asOf = m.clock.Now()
	}
	limit := req.Limit
	if limit == 0 {
		limit = m.defaultLimit
	}
	depth := req.CandidatesPerChannel
	if depth == 0 {
		depth = DefaultCandidatesPerChannel
	}
	// Fusion needs more to work with than it returns, so a caller who asks for
	// a deep result and a shallow search gets the deeper of the two.
	depth = max(depth, limit)

	factLimit := req.FactLimit
	if factLimit == 0 {
		factLimit = DefaultFactLimit
	}

	// The event-time instant facts must hold at. An explicit event-time window
	// means the caller is asking about its end; otherwise the cut is AsOf,
	// which is the clock unless the caller pinned it to reproduce a result.
	factsAt := req.OccurredTo
	if factsAt.IsZero() {
		factsAt = asOf
	}

	channels := m.resolveChannels(req.Channels)
	plan := recallPlan{
		text: req.Query,
		episodes: ChannelQuery{
			Scope:        req.Scope,
			AsOf:         asOf,
			OccurredFrom: req.OccurredFrom,
			OccurredTo:   req.OccurredTo,
			Binding:      req.Binding.Clone(),
			Roles:        slices.Clone(req.Roles),
			Limit:        depth,
		},
		facts: FactQuery{
			Scope:         req.Scope,
			AsOf:          asOf,
			At:            factsAt,
			Subjects:      slices.Clone(req.Subjects),
			MinConfidence: req.MinFactConfidence,
			Text:          req.Query,
			Limit:         factLimit,
		},
	}
	if m.extractor != nil {
		plan.facts.Extractor = m.extractor.Model()
	}

	results := m.runChannels(ctx, channels, plan)

	// Cancellation is not a channel failure, and reporting it as one would bury
	// the reason under a summary.
	if err := ctx.Err(); err != nil {
		return Recollection{}, fmt.Errorf("mempher: recall: %w", err)
	}
	if err := allFailed(results); err != nil {
		return Recollection{}, err
	}

	out := Recollection{
		Scope:    req.Scope,
		AsOf:     asOf,
		Channels: make([]ChannelReport, len(results)),
	}
	for i, r := range results {
		out.Channels[i] = ChannelReport{
			Channel:    r.channel,
			Candidates: len(r.candidates) + len(r.facts),
			Duration:   r.elapsed,
			Err:        r.err,
		}
	}

	// Facts take the budget first. They are the compact, current summary of
	// the scope, so trading one away for another line of transcript is a bad
	// deal at any budget.
	for _, r := range results {
		for _, fact := range r.facts {
			if len(out.Facts) >= factLimit {
				out.Truncated = true
				break
			}
			cost := m.tokens.CountTokens(fact.Statement)
			if req.MaxTokens > 0 && out.Tokens+cost > req.MaxTokens {
				out.Truncated = true
				break
			}
			out.Facts = append(out.Facts, RecalledFact{Fact: fact, Tokens: cost})
			out.Tokens += cost
		}
	}

	for _, recalled := range fuse(results, m.fusion) {
		if len(out.Episodes) >= limit {
			out.Truncated = true
			break
		}
		cost := m.tokens.CountTokens(recalled.Episode.Content)
		if req.MaxTokens > 0 && out.Tokens+cost > req.MaxTokens {
			out.Truncated = true
			break
		}
		recalled.Tokens = cost
		out.Episodes = append(out.Episodes, recalled)
		out.Tokens += cost
	}
	return out, nil
}

// runChannels runs every requested channel at once, each writing to a slot of
// its own so that the order results arrive in cannot affect the fused order.
func (m *Memory) runChannels(
	ctx context.Context,
	channels []Channel,
	plan recallPlan,
) []channelResult {
	results := make([]channelResult, len(channels))
	var wg sync.WaitGroup

	for i, channel := range channels {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Wall-clock, deliberately: this measures how long the work took,
			// which a logical test Clock would report as zero.
			started := time.Now()
			candidates, facts, err := m.runChannel(ctx, channel, plan)
			results[i] = channelResult{
				channel:    channel,
				candidates: candidates,
				facts:      facts,
				elapsed:    time.Since(started),
				err:        err,
			}
		}()
	}
	wg.Wait()
	return results
}

// recallPlan is everything the channels need, resolved once before the fan-out
// so that every channel sees exactly the same cuts. Two channels resolving "now"
// a millisecond apart would make a result that cannot be replayed.
type recallPlan struct {
	episodes ChannelQuery
	facts    FactQuery
	text     string
}

// runChannel is one channel's whole job, embedding included.
//
// The query is embedded here rather than before the fan-out, so a slow or
// unreachable embedder delays only the channel that needs it. The lexical
// channel answers at full speed either way.
func (m *Memory) runChannel(
	ctx context.Context,
	channel Channel,
	plan recallPlan,
) ([]Candidate, []Fact, error) {
	switch channel {
	case ChannelSemantic:
		vector, err := m.embedder.EmbedQuery(ctx, plan.text)
		if err != nil {
			return nil, nil, fmt.Errorf("embed query: %w", err)
		}
		candidates, err := m.store.SearchSemantic(ctx, SemanticQuery{
			ChannelQuery: plan.episodes,
			Vector:       vector,
			Model:        m.embedder.Model(),
		})
		return candidates, nil, err
	case ChannelLexical:
		candidates, err := m.store.SearchLexical(ctx,
			LexicalQuery{ChannelQuery: plan.episodes, Text: plan.text})
		return candidates, nil, err
	case ChannelFact:
		// Asked for explicitly on a Memory with no Extractor. Reported as this
		// channel failing, like any other channel that cannot answer, so the
		// rest of the result still arrives and the reason is in
		// [Recollection.Channels] rather than swallowed.
		if m.facts == nil {
			return nil, nil, fmt.Errorf(
				"the fact channel needs an Extractor and a FactStore: %w", ErrInvalidConfig)
		}
		facts, err := m.facts.Facts(ctx, plan.facts)
		return nil, facts, err
	default:
		return nil, nil, fmt.Errorf("mempher: channel %q: %w", channel, ErrInvalidChannel)
	}
}

// resolveChannels returns the channels to run, in a stable order and without
// repeats. No channels named means every channel this [Memory] can answer, which
// excludes [ChannelFact] when no [Extractor] was configured: a default that
// failed on every call would make L1 opt-out rather than opt-in.
func (m *Memory) resolveChannels(requested []Channel) []Channel {
	if len(requested) == 0 {
		if m.facts == nil {
			return []Channel{ChannelSemantic, ChannelLexical}
		}
		return []Channel{ChannelSemantic, ChannelLexical, ChannelFact}
	}
	out := make([]Channel, 0, len(requested))
	for _, channel := range requested {
		if !slices.Contains(out, channel) {
			out = append(out, channel)
		}
	}
	return out
}

// allFailed reports [ErrAllChannelsFailed] when nothing answered, so an empty
// memory is never mistaken for a broken one. Every underlying cause is joined
// in, because the first is rarely the interesting one.
func allFailed(results []channelResult) error {
	causes := make([]error, 0, len(results))
	for _, r := range results {
		if r.err == nil {
			return nil
		}
		causes = append(causes, fmt.Errorf("%s: %w", r.channel, r.err))
	}
	if len(causes) == 0 {
		return nil
	}
	return fmt.Errorf("mempher: recall: %w: %w", ErrAllChannelsFailed, errors.Join(causes...))
}

// jobsFor is the deferred work one new episode implies. A [Store] fills in the
// episode id, and the jobs commit with the episode.
func (m *Memory) jobsFor(scope ScopeID) []NewJob {
	jobs := []NewJob{{Kind: JobKindEncode, Scope: scope}}
	if m.facts != nil {
		jobs = append(jobs, NewJob{Kind: JobKindExtract, Scope: scope})
	}
	return jobs
}

// resolveFactStore picks the [FactStore] a configuration implies, and rejects
// the combinations that cannot work.
//
// L1 is opt-in through [Extractor], because it is the port that costs money.
// A FactStore without one is the configuration worth catching: nothing would
// name the facts to read or write, so recall would silently return none.
func resolveFactStore(store Store, explicit FactStore, extractor Extractor, op string) (FactStore, error) {
	if extractor == nil {
		if explicit != nil {
			return nil, fmt.Errorf(
				"mempher: %s: FactStore is set but Extractor is nil, so nothing would "+
					"produce or name the facts: %w", op, ErrInvalidConfig)
		}
		return nil, nil
	}
	if explicit != nil {
		return explicit, nil
	}
	asFactStore, ok := store.(FactStore)
	if !ok {
		return nil, fmt.Errorf(
			"mempher: %s: FactStore is required because %T does not implement mempher.FactStore: %w",
			op, store, ErrInvalidConfig)
	}
	return asFactStore, nil
}
