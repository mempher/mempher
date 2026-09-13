package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/extract"
)

type request struct {
	body    string
	headers http.Header
}

func serve(t *testing.T, cfg Config, status int, reply string) (*Completer, *request) {
	t.Helper()
	var seen request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		seen = request{body: string(body), headers: r.Header.Clone()}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)

	cfg.BaseURL = srv.URL
	if cfg.Model == "" {
		cfg.Model = "gpt-5"
	}
	if cfg.APIKey == "" {
		cfg.APIKey = "test-key"
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c, &seen
}

type stub struct{}

func (stub) Model() string { return "stub" }
func (stub) Complete(context.Context, extract.Prompt) ([]byte, error) {
	return []byte(`{"assertions":[],"retractions":[]}`), nil
}

func prompt(t *testing.T) extract.Prompt {
	t.Helper()
	ex, err := extract.New(stub{}, extract.Options{
		Predicates: []mempher.Predicate{"lives_in", "allergic_to"},
	})
	if err != nil {
		t.Fatalf("extract.New: %v", err)
	}
	return extract.Prompt{System: "system text", User: "user text", Schema: ex.Schema()}
}

func reply(content string) string {
	encoded, err := json.Marshal(content)
	if err != nil {
		panic(err)
	}
	return `{"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":` +
		string(encoded) + `}}]}`
}

func TestNewRejectsUnusableConfig(t *testing.T) {
	if _, err := New(Config{}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Fatalf("New with no model: got %v, want ErrInvalidConfig", err)
	}
	if _, err := New(Config{Model: "m", MaxTokens: -1}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Fatalf("New with a negative budget: got %v, want ErrInvalidConfig", err)
	}
}

func TestCompleteSendsAStrictSchemaAndReturnsTheContent(t *testing.T) {
	c, seen := serve(t, Config{}, http.StatusOK,
		reply(`{"assertions":[],"retractions":[]}`))

	got, err := c.Complete(t.Context(), prompt(t))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if !strings.Contains(string(got), "assertions") {
		t.Fatalf("got %s, want the message content", got)
	}

	for _, want := range []string{
		`"type":"json_schema"`,
		`"strict":true`,
		`"temperature":0`,
		`"enum":["allergic_to","lives_in"]`,
		`"additionalProperties":false`,
		`"system"`,
		`"user text"`,
	} {
		if !strings.Contains(seen.body, want) {
			t.Fatalf("the request does not carry %s:\n%s", want, seen.body)
		}
	}
	if got := seen.headers.Get("Authorization"); got != "Bearer test-key" {
		t.Fatalf("Authorization = %q, want the configured key", got)
	}

	// Strict mode demands every property be required, so the one optional field
	// has to arrive as required-and-nullable rather than simply absent.
	if !strings.Contains(seen.body, `"valid_to"`) {
		t.Fatal("valid_to is missing from the strict schema")
	}
	if !strings.Contains(seen.body, `["string","null"]`) {
		t.Fatalf("valid_to was not made nullable for strict mode:\n%s", seen.body)
	}
}

func TestLenientSendsTheSchemaUntouched(t *testing.T) {
	// Some OpenAI-compatible gateways reject strict mode outright. The escape
	// hatch costs the enum that enforces the vocabulary, and nothing else.
	c, seen := serve(t, Config{Lenient: true}, http.StatusOK,
		reply(`{"assertions":[],"retractions":[]}`))

	if _, err := c.Complete(t.Context(), prompt(t)); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if !strings.Contains(seen.body, `"strict":false`) {
		t.Fatalf("strict was still requested:\n%s", seen.body)
	}
	if strings.Contains(seen.body, `["string","null"]`) {
		t.Fatal("the schema was rewritten for a mode that was not requested")
	}
}

func TestCompleteReportsWhatWentWrong(t *testing.T) {
	cases := map[string]struct {
		status int
		reply  string
		want   string
	}{
		"an error status": {
			http.StatusTooManyRequests, `{"error":{"message":"slow down"}}`, "429",
		},
		"an error body": {
			http.StatusOK,
			`{"error":{"type":"server_error","message":"try later"}}`,
			"server_error",
		},
		"a refusal": {
			http.StatusOK,
			`{"choices":[{"finish_reason":"stop","message":{"refusal":"I cannot"}}]}`,
			"refused",
		},
		"a reply cut off at the budget": {
			http.StatusOK,
			`{"choices":[{"finish_reason":"length","message":{"content":"{\"asse"}}]}`,
			"cut off at the token budget",
		},
		"no choices": {http.StatusOK, `{"choices":[]}`, "no choices"},
		"a reply that is not JSON": {
			http.StatusOK, `<html>gateway</html>`, "decode the response",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			c, _ := serve(t, Config{}, tc.status, tc.reply)
			_, err := c.Complete(t.Context(), prompt(t))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Complete: got %v, want an error mentioning %q", err, tc.want)
			}
		})
	}
}

func TestStrictifyKeepsAnOptionalFieldAnswerable(t *testing.T) {
	got, err := strictify([]byte(`{
		"type": "object",
		"properties": {
			"needed":   {"type": "string"},
			"optional": {"type": "string"},
			"nested":   {"type": "object", "properties": {"deep": {"type": "number"}}}
		},
		"required": ["needed"],
		"additionalProperties": false
	}`))
	if err != nil {
		t.Fatalf("strictify: %v", err)
	}

	var doc map[string]any
	if err := json.Unmarshal(got, &doc); err != nil {
		t.Fatalf("strictify produced invalid JSON: %v", err)
	}
	required, _ := doc["required"].([]any)
	if len(required) != 3 {
		t.Fatalf("required = %v, want every property", required)
	}

	properties := doc["properties"].(map[string]any)
	if kind := properties["needed"].(map[string]any)["type"]; kind != "string" {
		t.Fatalf("an already-required field was made nullable: %v", kind)
	}
	optional := properties["optional"].(map[string]any)["type"]
	if kinds, ok := optional.([]any); !ok || len(kinds) != 2 || kinds[1] != "null" {
		t.Fatalf("optional type = %v, want [string null]", optional)
	}
	// Nested objects are rewritten too, or the endpoint rejects the whole
	// schema for the one object that was missed.
	nested := properties["nested"].(map[string]any)
	if req, _ := nested["required"].([]any); len(req) != 1 {
		t.Fatalf("a nested object was left alone: %v", nested)
	}
}

func TestStrictifyRejectsNonsense(t *testing.T) {
	if _, err := strictify([]byte(`not a schema`)); err == nil {
		t.Fatal("strictify accepted something that is not JSON")
	}
}

func TestCompleterSatisfiesTheExtractorEndToEnd(t *testing.T) {
	c, _ := serve(t, Config{}, http.StatusOK, reply(`{
		"assertions": [{
			"subject": "the user", "predicate": "lives_in", "object": "Berlin",
			"statement": "the user lives in Berlin",
			"valid_from": "2024-06-01", "valid_to": null, "confidence": 0.9
		}],
		"retractions": []
	}`))

	ex, err := extract.New(c, extract.Options{Predicates: []mempher.Predicate{"lives_in"}})
	if err != nil {
		t.Fatalf("extract.New: %v", err)
	}
	res, err := ex.Extract(t.Context(), mempher.ExtractRequest{
		Scope:    "user:1",
		Episodes: []mempher.Episode{{Seq: 1, Content: "I moved to Berlin.", Role: mempher.RoleUser}},
		Now:      time.Date(2024, 6, 15, 12, 0, 0, 0, time.UTC),
	})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.Assert) != 1 {
		t.Fatalf("got %+v, want one assertion", res.Assert)
	}
	// A null valid_to is how strict mode says "still true", and it has to reach
	// the store as an open window rather than as a parse failure.
	if !res.Assert[0].Valid.IsOpen() {
		t.Fatalf("a null valid_to closed the window at %s", res.Assert[0].Valid.To)
	}
	if err := res.Assert[0].Validate(); err != nil {
		t.Fatalf("a returned assertion a FactStore would refuse: %v", err)
	}
}
