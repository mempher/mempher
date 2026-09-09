// Memory: the public entry point, wiring the write and read loops.

package mempher

import (
	"context"
	"fmt"
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
func (m *Memory) Append(ctx context.Context, req AppendRequest) (Episode, error) { //nolint:revive // ctx is used once the write loop lands
	return Episode{}, errNotImplemented
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
func (m *Memory) Recall(ctx context.Context, req RecallRequest) (Recollection, error) { //nolint:revive // ctx is used once the read loop lands
	return Recollection{}, errNotImplemented
}
