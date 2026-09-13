// Package openai adapts any OpenAI-compatible embeddings endpoint to
// [mempher.Embedder]: OpenAI itself, Azure, Voyage's compatible route, and every
// local server that speaks the same shape -- Ollama, vLLM, text-embeddings-
// inference.
//
// Like the extraction adapters it speaks the REST API directly and adds no
// dependency: one POST, one JSON decode. Depending on mempher still pulls pgx
// and uuid and nothing else.
//
//	embedder, err := openai.New(openai.Config{
//		Model:      "text-embedding-3-small",
//		Dimensions: 1536,
//	})
//
// The width is declared rather than discovered, because the database is migrated
// for it. An embedder that reported one width and returned another would fail on
// every insert, so [New] states the contract and [Embedder.EmbedDocuments]
// enforces it on every reply.
package openai

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/mempher/mempher"
)

// Defaults, used when the corresponding [Config] field is zero.
const (
	// DefaultBaseURL is OpenAI's own endpoint.
	DefaultBaseURL = "https://api.openai.com/v1"

	// DefaultMaxBatch is how many texts go in one request. A worker hands over
	// a whole append batch at once, and providers cap how much they will take.
	DefaultMaxBatch = 64
)

// maxErrorBody is how much of a failed response is quoted back in an error.
const maxErrorBody = 2048

// Config assembles an [Embedder].
type Config struct {
	// Model is the embedding model id. Required, and deliberately without a
	// default: it names the vector space that encodings are keyed by, so a
	// default could move a deployment into a different space on upgrade and
	// leave every existing encoding unreadable.
	Model string

	// Dimensions is the width this model returns. Required, and checked
	// against every reply.
	//
	// Some providers also accept it as a request parameter and will truncate
	// to it; it is sent when [Config.RequestDimensions] is set.
	Dimensions int

	// RequestDimensions asks the provider to return vectors of exactly
	// [Config.Dimensions], for models that support shortening. Off by default
	// because a provider that does not support it rejects the request.
	RequestDimensions bool

	// QueryPrefix is prepended to the text given to [Embedder.EmbedQuery] and
	// to nothing else.
	//
	// Asymmetric models want it: bge and e5 are trained with an instruction on
	// the query side only, and omitting it measurably narrows the gap between
	// a relevant document and an irrelevant one. It is part of the model id,
	// because changing it changes where queries land in the space.
	QueryPrefix string

	// BaseURL is the endpoint root, without a trailing "/embeddings". Empty
	// means [DefaultBaseURL].
	BaseURL string

	// APIKey is sent as a bearer token. Empty falls back to OPENAI_API_KEY,
	// and a local server wanting no key is served by leaving both unset.
	APIKey string

	// MaxBatch is how many texts go in one request. Zero means
	// [DefaultMaxBatch].
	MaxBatch int

	// HTTPClient issues the request. Nil means [http.DefaultClient].
	HTTPClient *http.Client
}

// Embedder satisfies [mempher.Embedder] against an embeddings endpoint.
type Embedder struct {
	client      *http.Client
	url         string
	apiKey      string
	model       string
	id          mempher.ModelID
	dimensions  int
	sendDims    bool
	queryPrefix string
	maxBatch    int
}

var _ mempher.Embedder = (*Embedder)(nil)

// New returns an [Embedder]. It opens no connection, so it needs no context.
func New(cfg Config) (*Embedder, error) {
	switch {
	case cfg.Model == "":
		return nil, fmt.Errorf("mempher/embed/openai: new: Model is required: %w",
			mempher.ErrInvalidConfig)
	case cfg.Dimensions <= 0:
		return nil, fmt.Errorf("mempher/embed/openai: new: Dimensions is %d: %w",
			cfg.Dimensions, mempher.ErrInvalidConfig)
	case cfg.MaxBatch < 0:
		return nil, fmt.Errorf("mempher/embed/openai: new: MaxBatch is %d: %w",
			cfg.MaxBatch, mempher.ErrInvalidConfig)
	}
	if cfg.MaxBatch == 0 {
		cfg.MaxBatch = DefaultMaxBatch
	}

	base := cfg.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	key := cfg.APIKey
	if key == "" {
		key = os.Getenv("OPENAI_API_KEY")
	}
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	// A query prefix moves queries within the space, so two configurations that
	// differ only by it are not interchangeable and must not share encodings.
	id := cfg.Model
	if cfg.QueryPrefix != "" {
		sum := sha256.Sum256([]byte(cfg.QueryPrefix))
		id += "+q:" + hex.EncodeToString(sum[:])[:8]
	}

	return &Embedder{
		client:      client,
		url:         strings.TrimSuffix(base, "/") + "/embeddings",
		apiKey:      key,
		model:       cfg.Model,
		id:          mempher.ModelID(id),
		dimensions:  cfg.Dimensions,
		sendDims:    cfg.RequestDimensions,
		queryPrefix: cfg.QueryPrefix,
		maxBatch:    cfg.MaxBatch,
	}, nil
}

// Model identifies the vector space. Encodings are keyed by it, so a change is a
// backfill rather than a silent mixture of two spaces.
func (e *Embedder) Model() mempher.ModelID { return e.id }

// Dimensions is the width every vector this embedder returns.
func (e *Embedder) Dimensions() int { return e.dimensions }

// EmbedQuery embeds a retrieval cue, with [Config.QueryPrefix] if one is set.
func (e *Embedder) EmbedQuery(ctx context.Context, text string) (mempher.Vector, error) {
	vectors, err := e.post(ctx, []string{e.queryPrefix + text})
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// EmbedDocuments embeds episode content, one vector per input, in order,
// chunked into requests of at most [Config.MaxBatch].
func (e *Embedder) EmbedDocuments(
	ctx context.Context,
	texts []string,
) ([]mempher.Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([]mempher.Vector, 0, len(texts))
	for start := 0; start < len(texts); start += e.maxBatch {
		end := min(start+e.maxBatch, len(texts))
		vectors, err := e.post(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vectors...)
	}
	return out, nil
}

type embeddingsRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
}

type embeddingsResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func (e *Embedder) post(ctx context.Context, texts []string) ([]mempher.Vector, error) {
	request := embeddingsRequest{Model: e.model, Input: texts}
	if e.sendDims {
		request.Dimensions = e.dimensions
	}
	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("mempher/embed/openai: build the request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("mempher/embed/openai: build the request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	res, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("mempher/embed/openai: post %s: %w", e.url, err)
	}
	defer func() { _ = res.Body.Close() }()

	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("mempher/embed/openai: read the response: %w", err)
	}
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("mempher/embed/openai: %s returned %s: %s",
			e.url, res.Status, clip(raw))
	}

	var parsed embeddingsResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("mempher/embed/openai: decode the response: %w", err)
	}
	if parsed.Error != nil {
		return nil, fmt.Errorf("mempher/embed/openai: %s: %s",
			parsed.Error.Type, parsed.Error.Message)
	}
	if len(parsed.Data) != len(texts) {
		return nil, fmt.Errorf("mempher/embed/openai: asked for %d embeddings, got %d",
			len(texts), len(parsed.Data))
	}

	// Ordered by index rather than trusted to arrive in order: the contract is
	// one vector per input in order, and a silently transposed pair would put
	// every episode at the wrong point in the space.
	slices.SortFunc(parsed.Data, func(a, b struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	},
	) int {
		return a.Index - b.Index
	})

	out := make([]mempher.Vector, len(parsed.Data))
	for i, item := range parsed.Data {
		if len(item.Embedding) != e.dimensions {
			return nil, fmt.Errorf(
				"mempher/embed/openai: %s returned a %d-wide vector, but Dimensions is %d: %w",
				e.model, len(item.Embedding), e.dimensions, mempher.ErrDimensionMismatch)
		}
		out[i] = mempher.Vector(item.Embedding)
	}
	return out, nil
}

func clip(body []byte) string {
	if len(body) > maxErrorBody {
		return string(body[:maxErrorBody]) + "..."
	}
	return string(body)
}
