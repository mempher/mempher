// The vector projection of L0, and the port that produces it.

package mempher

import (
	"context"
	"time"
)

// Vector is a dense embedding, stored as pgvector's vector type. Every vector
// for one [ModelID] must have that model's width; a mismatch is
// [ErrDimensionMismatch].
type Vector []float32

// Dimensions returns the length of the vector.
func (v Vector) Dimensions() int { return len(v) }

// Encoding is a vector projection of one episode, derived from L0 and therefore
// disposable: drop the row and a [Worker] rebuilds it.
//
// A [Store] fills in the encoding's scope and ingest time from the episode row
// rather than trusting a caller for them. Those copies let the vector channel
// filter by scope and as-of without a join, and cannot drift because the row
// they came from is immutable.
type Encoding struct {
	// Episode is the episode this encodes.
	Episode EpisodeID
	// Model identifies the vector space. Encodings are keyed by (episode,
	// model), so writing one twice is an upsert.
	Model ModelID
	// Vector is the embedding of the episode's content.
	Vector Vector
	// Surprise is how novel the episode was, in [0,1]. Zero means not
	// measured; nothing computes it in this stage.
	Surprise float32
	// EncodedAt is when it was produced, from the [Clock].
	EncodedAt time.Time
}

// Embedder turns text into vectors. Documents and queries go through separate
// methods because asymmetric models are common, and folding both into one call
// would silently embed queries as documents.
//
// Implementations must be safe for concurrent use and honour cancellation.
type Embedder interface {
	// EmbedDocuments embeds episode content, one vector per input, in order.
	EmbedDocuments(ctx context.Context, texts []string) ([]Vector, error)

	// EmbedQuery embeds a retrieval cue.
	EmbedQuery(ctx context.Context, text string) (Vector, error)

	// Dimensions is the width of every vector this embedder produces, and
	// must match the width the database was migrated for.
	Dimensions() int

	// Model identifies the vector space, precisely enough that changing the
	// model changes this value.
	Model() ModelID
}
