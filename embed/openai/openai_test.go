package openai

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mempher/mempher"
)

type request struct {
	body    string
	headers http.Header
}

// serve returns vectors of the configured width, one per input, so a test can
// assert the adapter's own behaviour rather than a model's.
func serve(t *testing.T, cfg Config, status int, reply func(inputs []string) string) (*Embedder, *[]request) {
	t.Helper()
	var seen []request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		seen = append(seen, request{body: string(body), headers: r.Header.Clone()})

		var parsed struct {
			Input []string `json:"input"`
		}
		_ = json.Unmarshal(body, &parsed)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, reply(parsed.Input))
	}))
	t.Cleanup(srv.Close)

	cfg.BaseURL = srv.URL
	if cfg.Model == "" {
		cfg.Model = "text-embedding-3-small"
	}
	if cfg.Dimensions == 0 {
		cfg.Dimensions = 4
	}
	if cfg.APIKey == "" {
		cfg.APIKey = "test-key"
	}
	e, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e, &seen
}

// vectors answers with one distinct vector per input, in order.
func vectors(width int) func([]string) string {
	return func(inputs []string) string {
		var b strings.Builder
		b.WriteString(`{"data":[`)
		for i := range inputs {
			if i > 0 {
				b.WriteString(",")
			}
			b.WriteString(`{"object":"embedding","index":`)
			b.WriteString(itoa(i))
			b.WriteString(`,"embedding":[`)
			for d := range width {
				if d > 0 {
					b.WriteString(",")
				}
				b.WriteString(itoa(i*100 + d))
			}
			b.WriteString(`]}`)
		}
		b.WriteString(`]}`)
		return b.String()
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digits []byte
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}

func TestNewRejectsUnusableConfig(t *testing.T) {
	cases := map[string]Config{
		"no model":       {Dimensions: 4},
		"no width":       {Model: "m"},
		"negative width": {Model: "m", Dimensions: -1},
		"negative batch": {Model: "m", Dimensions: 4, MaxBatch: -1},
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := New(cfg); !errors.Is(err, mempher.ErrInvalidConfig) {
				t.Fatalf("New: got %v, want ErrInvalidConfig", err)
			}
		})
	}
}

func TestEmbedDocumentsPreservesOrderAcrossBatches(t *testing.T) {
	// The contract is one vector per input, in order. A worker writes them
	// against episode ids positionally, so a transposition puts episodes at
	// each other's coordinates and nothing ever notices.
	e, seen := serve(t, Config{Dimensions: 4, MaxBatch: 2}, http.StatusOK, vectors(4))

	texts := []string{"a", "b", "c", "d", "e"}
	got, err := e.EmbedDocuments(t.Context(), texts)
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if len(got) != len(texts) {
		t.Fatalf("got %d vectors, want %d", len(got), len(texts))
	}
	if len(*seen) != 3 {
		t.Fatalf("made %d requests for 5 texts at MaxBatch 2, want 3", len(*seen))
	}
	// Each batch restarts the server's index at zero, so the first component
	// repeating 0,100,0,100,0 is exactly the in-order result.
	for i, want := range []float32{0, 100, 0, 100, 0} {
		if got[i][0] != want {
			t.Errorf("vector %d starts %v, want %v", i, got[i][0], want)
		}
	}
}

func TestEmbedDocumentsReordersByIndex(t *testing.T) {
	// A provider is allowed to answer out of order. Trusting arrival order
	// would silently transpose two episodes.
	e, _ := serve(t, Config{Dimensions: 2}, http.StatusOK, func([]string) string {
		return `{"data":[
			{"index":1,"embedding":[9,9]},
			{"index":0,"embedding":[1,1]}
		]}`
	})

	got, err := e.EmbedDocuments(t.Context(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if got[0][0] != 1 || got[1][0] != 9 {
		t.Fatalf("got %v, want the vectors sorted back into input order", got)
	}
}

func TestEmbedQueryAppliesThePrefixAndNothingElseDoes(t *testing.T) {
	// bge and e5 are trained with an instruction on the query side only.
	const prefix = "Represent this sentence: "
	e, seen := serve(t, Config{Dimensions: 4, QueryPrefix: prefix}, http.StatusOK, vectors(4))

	if _, err := e.EmbedQuery(t.Context(), "where do they live"); err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if !strings.Contains((*seen)[0].body, prefix+"where do they live") {
		t.Fatalf("the query went unprefixed:\n%s", (*seen)[0].body)
	}

	if _, err := e.EmbedDocuments(t.Context(), []string{"they live in Berlin"}); err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if strings.Contains((*seen)[1].body, prefix) {
		t.Fatalf("a document was prefixed:\n%s", (*seen)[1].body)
	}
}

func TestQueryPrefixIsPartOfTheModelID(t *testing.T) {
	// Encodings are keyed by the model id. A prefix moves queries within the
	// space, so two configurations differing only by it are not interchangeable
	// and must not share a set of encodings.
	plain, _ := serve(t, Config{Dimensions: 4}, http.StatusOK, vectors(4))
	prefixed, _ := serve(t, Config{Dimensions: 4, QueryPrefix: "q: "}, http.StatusOK, vectors(4))
	other, _ := serve(t, Config{Dimensions: 4, QueryPrefix: "query: "}, http.StatusOK, vectors(4))

	if plain.Model() == prefixed.Model() {
		t.Error("adding a query prefix left the model id unchanged")
	}
	if prefixed.Model() == other.Model() {
		t.Error("two different prefixes produced the same model id")
	}
	if got := string(plain.Model()); got != "text-embedding-3-small" {
		t.Errorf("Model = %q, want the bare model id when no prefix is set", got)
	}
}

func TestWrongWidthIsRefused(t *testing.T) {
	// The database is migrated for one width. Catching it here names the cause;
	// letting it through names a constraint.
	e, _ := serve(t, Config{Dimensions: 8}, http.StatusOK, vectors(4))

	_, err := e.EmbedDocuments(t.Context(), []string{"a"})
	if !errors.Is(err, mempher.ErrDimensionMismatch) {
		t.Fatalf("EmbedDocuments: got %v, want ErrDimensionMismatch", err)
	}
}

func TestShortRepliesAreRefused(t *testing.T) {
	e, _ := serve(t, Config{Dimensions: 2}, http.StatusOK, func([]string) string {
		return `{"data":[{"index":0,"embedding":[1,1]}]}`
	})
	_, err := e.EmbedDocuments(t.Context(), []string{"a", "b"})
	if err == nil || !strings.Contains(err.Error(), "asked for 2 embeddings, got 1") {
		t.Fatalf("EmbedDocuments: got %v, want a count mismatch", err)
	}
}

func TestFailuresAreReported(t *testing.T) {
	cases := map[string]struct {
		status int
		reply  string
		want   string
	}{
		"an error status": {http.StatusTooManyRequests, `{"error":{"message":"slow"}}`, "429"},
		"an error body": {
			http.StatusOK, `{"error":{"type":"invalid_request_error","message":"nope"}}`,
			"invalid_request_error",
		},
		"a reply that is not JSON": {http.StatusOK, `<html>gateway</html>`, "decode the response"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			e, _ := serve(t, Config{Dimensions: 2}, tc.status, func([]string) string { return tc.reply })
			_, err := e.EmbedQuery(t.Context(), "a")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("EmbedQuery: got %v, want an error mentioning %q", err, tc.want)
			}
		})
	}
}

func TestEmbeddingNothingCostsNoRequest(t *testing.T) {
	e, seen := serve(t, Config{Dimensions: 4}, http.StatusOK, vectors(4))
	got, err := e.EmbedDocuments(t.Context(), nil)
	if err != nil || got != nil {
		t.Fatalf("EmbedDocuments(nil) = %v, %v", got, err)
	}
	if len(*seen) != 0 {
		t.Fatalf("made %d requests for no texts", len(*seen))
	}
}
