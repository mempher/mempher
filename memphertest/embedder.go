package memphertest

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"strings"
	"sync"
	"unicode"

	"github.com/mempher/mempher"
)

// Embedder is a deterministic [mempher.Embedder] that needs no model provider.
//
// It embeds a hashed bag of words: text is lowercased, split on anything that is
// not a letter or digit, each token hashed into one dimension with a sign from a
// second hash bit, then L2-normalised. That hashing trick buys the property a
// fake embedder needs to be useful: cosine similarity tracks token overlap.
//
// So a test can assert real ranking rather than identity. "hazelnut allergy"
// sits closer to "I am allergic to hazelnuts" than to "I drink dark roast
// coffee", and still will on another machine. An embedder returning a hash of
// the whole string would only match text exactly, testing nothing about order.
//
// Queries and documents embed identically, so a test that swaps them still
// passes; assert that distinction against your real embedder.
//
// Embedder is safe for concurrent use.
type Embedder struct {
	dims  int
	model mempher.ModelID

	mu      sync.RWMutex
	failure error
	calls   int
}

// NewEmbedder returns an Embedder producing vectors of dims dimensions. It
// panics if dims is not positive: that is a bug in the test, not in the code
// under test.
func NewEmbedder(dims int) *Embedder {
	if dims <= 0 {
		panic(fmt.Sprintf("memphertest: NewEmbedder(%d): dimensions must be positive", dims))
	}
	return &Embedder{
		dims:  dims,
		model: mempher.ModelID(fmt.Sprintf("memphertest/hashed-bow@%d", dims)),
	}
}

// Dimensions returns the width of every vector this embedder produces.
func (e *Embedder) Dimensions() int { return e.dims }

// Model identifies the vector space, including the width, so two differently
// sized test embedders never look interchangeable.
func (e *Embedder) Model() mempher.ModelID { return e.model }

// Fail makes every later call return err, so a test can exercise the paths that
// only run when embedding is unavailable, such as a degraded recall or a retried
// job. Pass nil to stop failing.
func (e *Embedder) Fail(err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.failure = err
}

// Calls reports how many embedding calls have been made, which is how a test
// asserts that the write path made none.
func (e *Embedder) Calls() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.calls
}

// EmbedDocuments embeds episode content, one vector per input, in order.
func (e *Embedder) EmbedDocuments(ctx context.Context, texts []string) ([]mempher.Vector, error) {
	if err := e.begin(ctx); err != nil {
		return nil, err
	}
	out := make([]mempher.Vector, len(texts))
	for i, text := range texts {
		out[i] = e.embed(text)
	}
	return out, nil
}

// EmbedQuery embeds a retrieval cue.
func (e *Embedder) EmbedQuery(ctx context.Context, text string) (mempher.Vector, error) {
	if err := e.begin(ctx); err != nil {
		return nil, err
	}
	return e.embed(text), nil
}

// begin honours cancellation, records the call, and returns any injected
// failure. Context first, so a cancelled test fails for the intended reason.
func (e *Embedder) begin(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("memphertest: embed: %w", err)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.calls++
	if e.failure != nil {
		return fmt.Errorf("memphertest: embed: %w", e.failure)
	}
	return nil
}

// embed hashes the tokens of text into a unit vector. Token-free text embeds as
// the zero vector, which pgvector accepts and sorts last under cosine distance.
func (e *Embedder) embed(text string) mempher.Vector {
	vec := make(mempher.Vector, e.dims)
	for _, token := range tokenize(text) {
		h := fnv.New64a()
		_, _ = h.Write([]byte(token))
		sum := h.Sum64()
		// Low bits choose the dimension, the top bit the sign, so colliding
		// tokens tend to cancel rather than pile up.
		idx := int(sum % uint64(e.dims))
		if sum&(1<<63) != 0 {
			vec[idx]--
		} else {
			vec[idx]++
		}
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if norm == 0 {
		return vec
	}
	norm = math.Sqrt(norm)
	for i := range vec {
		vec[i] = float32(float64(vec[i]) / norm)
	}
	return vec
}

// tokenize splits on everything that is not a letter or digit, so punctuation
// and case never change an embedding.
func tokenize(text string) []string {
	return strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

var _ mempher.Embedder = (*Embedder)(nil)
