package anthropic

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

// request is what the endpoint saw, so a test can assert what was asked for
// rather than only what came back.
type request struct {
	body    string
	headers http.Header
}

func serve(t *testing.T, status int, reply string) (*Completer, *request) {
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

	c, err := New(Config{Model: "claude-opus-5", APIKey: "test-key", BaseURL: srv.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c, &seen
}

// stub exists only to build a real Extractor, whose schema is the thing worth
// sending. A hand-written schema here would test the test.
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

const toolUseReply = `{
	"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
	"content": [{"type": "tool_use", "id": "toolu_1", "name": "record_extraction",
		"input": {"assertions": [], "retractions": []}}],
	"stop_reason": "tool_use",
	"usage": {"input_tokens": 1, "output_tokens": 1}
}`

func TestNewRejectsUnusableConfig(t *testing.T) {
	// Model has no default: it ends up in the key of every fact this produces,
	// so a default could move a deployment onto another model on upgrade.
	if _, err := New(Config{}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Fatalf("New with no model: got %v, want ErrInvalidConfig", err)
	}
	if _, err := New(Config{Model: "m", MaxTokens: -1}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Fatalf("New with a negative budget: got %v, want ErrInvalidConfig", err)
	}
}

func TestCompleteSendsTheSchemaAndReturnsTheToolInput(t *testing.T) {
	c, seen := serve(t, http.StatusOK, toolUseReply)

	got, err := c.Complete(t.Context(), prompt(t))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(got, &body); err != nil {
		t.Fatalf("the tool input did not come back as JSON: %v", err)
	}
	if _, ok := body["assertions"]; !ok {
		t.Fatalf("got %s, want the tool input", got)
	}

	// The schema goes on the wire exactly as extract built it. A dropped enum
	// is a predicate vocabulary not enforced, and nothing downstream notices.
	for _, want := range []string{
		`"temperature":0`,
		`"tool_choice":{"type":"tool","name":"record_extraction"}`,
		`"enum":["allergic_to","lives_in"]`,
		`"additionalProperties":false`,
		`"system":"system text"`,
		`"user text"`,
	} {
		if !strings.Contains(seen.body, want) {
			t.Fatalf("the request does not carry %s:\n%s", want, seen.body)
		}
	}
	if got := seen.headers.Get("anthropic-version"); got != APIVersion {
		t.Fatalf("anthropic-version = %q, want %q", got, APIVersion)
	}
	if got := seen.headers.Get("x-api-key"); got != "test-key" {
		t.Fatalf("x-api-key = %q, want the configured key", got)
	}
}

func TestCompleteReportsWhatWentWrong(t *testing.T) {
	cases := map[string]struct {
		status int
		reply  string
		want   string
	}{
		"an error status": {
			http.StatusTooManyRequests,
			`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`,
			"429",
		},
		"an error body": {
			http.StatusOK,
			`{"type":"error","error":{"type":"overloaded_error","message":"try later"}}`,
			"overloaded_error",
		},
		"a reply cut off at the budget": {
			http.StatusOK,
			`{"content":[],"stop_reason":"max_tokens"}`,
			"cut off at the token budget",
		},
		"prose despite the forced tool": {
			http.StatusOK,
			`{"content":[{"type":"text","text":"no"}],"stop_reason":"end_turn"}`,
			"no record_extraction call",
		},
		"a reply that is not JSON": {
			http.StatusOK, `<html>gateway</html>`, "decode the response",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			c, _ := serve(t, tc.status, tc.reply)
			_, err := c.Complete(t.Context(), prompt(t))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Complete: got %v, want an error mentioning %q", err, tc.want)
			}
		})
	}
}

func TestCompleterSatisfiesTheExtractorEndToEnd(t *testing.T) {
	// The point of the adapter: what comes back is what a mempher.Worker can
	// hand to a FactStore, with no step in between.
	c, _ := serve(t, http.StatusOK, `{
		"content": [{"type": "tool_use", "name": "record_extraction", "input": {
			"assertions": [{
				"subject": "the user", "predicate": "lives_in", "object": "Berlin",
				"statement": "the user lives in Berlin",
				"valid_from": "2024-06-01", "confidence": 0.9
			}],
			"retractions": []
		}}],
		"stop_reason": "tool_use"
	}`)

	ex, err := extract.New(c, extract.Options{Predicates: []mempher.Predicate{"lives_in"}})
	if err != nil {
		t.Fatalf("extract.New: %v", err)
	}
	res, err := ex.Extract(t.Context(), mempher.ExtractRequest{
		Scope:    "user:1",
		Episodes: []mempher.Episode{{Seq: 1, Content: "I moved to Berlin.", Role: mempher.RoleUser}},
		Now:      mustTime(t),
	})
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.Assert) != 1 || res.Assert[0].Object != "Berlin" {
		t.Fatalf("got %+v, want one assertion about Berlin", res.Assert)
	}
	if err := res.Assert[0].Validate(); err != nil {
		t.Fatalf("a returned assertion a FactStore would refuse: %v", err)
	}
}

func mustTime(t *testing.T) time.Time {
	t.Helper()
	return time.Date(2024, 6, 15, 12, 0, 0, 0, time.UTC)
}
