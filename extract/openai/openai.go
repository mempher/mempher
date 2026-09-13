// Package openai adapts any OpenAI-compatible chat completions endpoint to
// [extract.Completer]: OpenAI itself, Azure, Groq, Together, OpenRouter, Ollama
// and vLLM among them.
//
// It is a separate Go module, like every adapter here, so that depending on
// mempher pulls nothing a caller did not ask for. Unlike the others it needs no
// SDK at all -- one POST and one JSON decode -- which is also why it can be
// honest about the word "compatible": the endpoint is a base URL you set, not a
// provider a vendor's client assumes.
//
//	go get github.com/mempher/mempher/extract/openai
//
// Structured output goes through response_format json_schema in strict mode,
// which is the only setting under which the predicate vocabulary is genuinely
// enforced rather than suggested.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/extract"
)

// DefaultBaseURL is OpenAI's own endpoint, used when [Config] names none.
const DefaultBaseURL = "https://api.openai.com/v1"

// maxErrorBody is how much of a failed response is quoted back in an error.
// Enough to identify the failure, not enough to put a provider's entire HTML
// error page into a log line.
const maxErrorBody = 2048

// schemaName labels the schema in the request. It is part of no persisted
// identity.
const schemaName = "extraction"

// Config assembles a [Completer].
type Config struct {
	// Model is the model id, such as "gpt-5" or whatever the gateway calls the
	// model it serves. Required, and deliberately without a default.
	//
	// It ends up inside the extractor id and therefore in the key of every fact
	// this produces, so a default here would let a library upgrade silently
	// move a deployment onto another model and orphan its facts.
	Model string

	// BaseURL is the endpoint root, without a trailing "/chat/completions".
	// Empty means [DefaultBaseURL].
	BaseURL string

	// APIKey is sent as a bearer token. Empty falls back to OPENAI_API_KEY, and
	// an endpoint that wants no key at all -- a local Ollama, say -- is served
	// by leaving both unset.
	APIKey string

	// MaxTokens is the output budget for one extraction. Zero omits the field
	// and lets the endpoint decide.
	MaxTokens int

	// Lenient asks for the schema as a hint rather than a constraint.
	//
	// Strict mode is the default because it is what actually enforces the
	// predicate vocabulary, but not every OpenAI-compatible gateway implements
	// it, and one that does not will reject the request outright. This is the
	// escape hatch, at the cost of having to catch in Go what the endpoint
	// would have refused.
	Lenient bool

	// HTTPClient issues the request. Nil means [http.DefaultClient].
	// Cancellation comes from the context either way.
	HTTPClient *http.Client
}

// Completer satisfies [extract.Completer] against a chat completions endpoint.
type Completer struct {
	client    *http.Client
	url       string
	apiKey    string
	model     string
	maxTokens int
	strict    bool
}

var _ extract.Completer = (*Completer)(nil)

// New returns a [Completer]. It opens no connection, so it needs no context.
func New(cfg Config) (*Completer, error) {
	if cfg.Model == "" {
		return nil, fmt.Errorf("mempher/extract/openai: new: Model is required: %w",
			mempher.ErrInvalidConfig)
	}
	if cfg.MaxTokens < 0 {
		return nil, fmt.Errorf("mempher/extract/openai: new: MaxTokens is %d: %w",
			cfg.MaxTokens, mempher.ErrInvalidConfig)
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
	return &Completer{
		client:    client,
		url:       strings.TrimSuffix(base, "/") + "/chat/completions",
		apiKey:    key,
		model:     cfg.Model,
		maxTokens: cfg.MaxTokens,
		strict:    !cfg.Lenient,
	}, nil
}

// Model returns the configured model id, which the extract package folds into
// the id every fact is keyed by.
func (c *Completer) Model() string { return c.model }

type chatRequest struct {
	Model          string        `json:"model"`
	Messages       []chatMessage `json:"messages"`
	Temperature    float64       `json:"temperature"`
	MaxTokens      int           `json:"max_completion_tokens,omitempty"`
	ResponseFormat any           `json:"response_format"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content string `json:"content"`
			Refusal string `json:"refusal"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// Complete sends one extraction and returns the message content, which strict
// mode guarantees is JSON matching the schema.
func (c *Completer) Complete(ctx context.Context, p extract.Prompt) ([]byte, error) {
	schema := json.RawMessage(p.Schema)
	if c.strict {
		strict, err := strictify(p.Schema)
		if err != nil {
			return nil, err
		}
		schema = strict
	}

	body, err := json.Marshal(chatRequest{
		Model: c.model,
		Messages: []chatMessage{
			{Role: "system", Content: p.System},
			{Role: "user", Content: p.User},
		},
		// Zero, and not configurable. A reclaimed lease replays a job, and an
		// extractor that answers differently the second time asserts the same
		// claim over an overlapping window -- ErrFactConflict, not a merged
		// answer.
		Temperature: 0,
		MaxTokens:   c.maxTokens,
		ResponseFormat: map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   schemaName,
				"strict": c.strict,
				"schema": schema,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("mempher/extract/openai: build the request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("mempher/extract/openai: build the request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	res, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("mempher/extract/openai: post %s: %w", c.url, err)
	}
	defer func() { _ = res.Body.Close() }()

	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("mempher/extract/openai: read the response: %w", err)
	}
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("mempher/extract/openai: %s returned %s: %s",
			c.url, res.Status, clip(raw))
	}

	var parsed chatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("mempher/extract/openai: decode the response: %w", err)
	}
	if parsed.Error != nil {
		return nil, fmt.Errorf("mempher/extract/openai: %s: %s",
			parsed.Error.Type, parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return nil, fmt.Errorf("mempher/extract/openai: the reply holds no choices: %s", clip(raw))
	}

	choice := parsed.Choices[0]
	if choice.Message.Refusal != "" {
		return nil, fmt.Errorf("mempher/extract/openai: the model refused: %s",
			choice.Message.Refusal)
	}
	if choice.FinishReason == "length" {
		// Reported here rather than left to arrive as malformed JSON, because
		// the fix is a number in Config and the symptom would not say so.
		return nil, fmt.Errorf(
			"mempher/extract/openai: the reply was cut off at the token budget; " +
				"raise Config.MaxTokens or lower extract.Options.MaxAssertions")
	}
	return []byte(choice.Message.Content), nil
}

// strictify rewrites a schema for strict json_schema mode, which demands that
// every property of every object appear in that object's required list.
//
// An optional field is expressed the only way strict mode allows: kept required
// and made nullable, so the model answers null instead of omitting it. That
// costs nothing downstream -- encoding/json leaves a string untouched when it
// decodes null, so an absent valid_to and a null one arrive identically.
func strictify(raw json.RawMessage) (json.RawMessage, error) {
	var doc any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("mempher/extract/openai: parse the response schema: %w", err)
	}
	requireEverything(doc)

	out, err := json.Marshal(doc)
	if err != nil {
		return nil, fmt.Errorf("mempher/extract/openai: rebuild the response schema: %w", err)
	}
	return out, nil
}

func requireEverything(node any) {
	switch typed := node.(type) {
	case []any:
		for _, child := range typed {
			requireEverything(child)
		}
		return
	case map[string]any:
		properties, ok := typed["properties"].(map[string]any)
		if ok {
			already := make(map[string]bool)
			if required, ok := typed["required"].([]any); ok {
				for _, name := range required {
					if s, ok := name.(string); ok {
						already[s] = true
					}
				}
			}
			names := make([]string, 0, len(properties))
			for name, sub := range properties {
				names = append(names, name)
				if !already[name] {
					allowNull(sub)
				}
			}
			// Sorted, so the request a given schema produces is byte-stable and
			// a diff between two of them means something.
			slices.Sort(names)
			typed["required"] = names
		}
		for key, child := range typed {
			if key == "required" {
				continue
			}
			requireEverything(child)
		}
	}
}

func allowNull(node any) {
	obj, ok := node.(map[string]any)
	if !ok {
		return
	}
	if name, ok := obj["type"].(string); ok && name != "null" {
		obj["type"] = []any{name, "null"}
	}
}

func clip(body []byte) string {
	if len(body) > maxErrorBody {
		return string(body[:maxErrorBody]) + "..."
	}
	return string(body)
}
