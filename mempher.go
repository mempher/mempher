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
// This is the write loop: validate, insert the episode, enqueue the encode job
// in the same statement, return. It makes no model call, so its latency is one
// round trip. The episode is lexically searchable the moment Append returns, and
// semantically searchable once a [Worker] drains the encode job.
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
		Episode: NewEpisode{
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
		},
		// The one thing an episode implies in this stage. It commits with the
		// episode, so it cannot be lost.
		Jobs: []NewJob{{Kind: JobKindEncode, Scope: req.Scope}},
	})
	if err != nil {
		return Episode{}, err
	}
	return result.Episode, nil
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

	channels := resolveChannels(req.Channels)
	query := ChannelQuery{
		Scope:        req.Scope,
		AsOf:         asOf,
		OccurredFrom: req.OccurredFrom,
		OccurredTo:   req.OccurredTo,
		Binding:      req.Binding.Clone(),
		Roles:        slices.Clone(req.Roles),
		Limit:        depth,
	}

	results := m.runChannels(ctx, channels, query, req.Query)

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
			Candidates: len(r.candidates),
			Duration:   r.elapsed,
			Err:        r.err,
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
	query ChannelQuery,
	text string,
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
			candidates, err := m.runChannel(ctx, channel, query, text)
			results[i] = channelResult{
				channel:    channel,
				candidates: candidates,
				elapsed:    time.Since(started),
				err:        err,
			}
		}()
	}
	wg.Wait()
	return results
}

// runChannel is one channel's whole job, embedding included.
//
// The query is embedded here rather than before the fan-out, so a slow or
// unreachable embedder delays only the channel that needs it. The lexical
// channel answers at full speed either way.
func (m *Memory) runChannel(
	ctx context.Context,
	channel Channel,
	query ChannelQuery,
	text string,
) ([]Candidate, error) {
	switch channel {
	case ChannelSemantic:
		vector, err := m.embedder.EmbedQuery(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("embed query: %w", err)
		}
		return m.store.SearchSemantic(ctx, SemanticQuery{
			ChannelQuery: query,
			Vector:       vector,
			Model:        m.embedder.Model(),
		})
	case ChannelLexical:
		return m.store.SearchLexical(ctx, LexicalQuery{ChannelQuery: query, Text: text})
	default:
		return nil, fmt.Errorf("mempher: channel %q: %w", channel, ErrInvalidChannel)
	}
}

// resolveChannels returns the channels to run, in a stable order and without
// repeats. No channels named means all of them.
func resolveChannels(requested []Channel) []Channel {
	if len(requested) == 0 {
		return []Channel{ChannelSemantic, ChannelLexical}
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
