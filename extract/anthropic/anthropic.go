// Package anthropic adapts Anthropic's Messages API to [extract.Completer].
//
// It speaks the REST API directly rather than through an SDK, so it adds no
// dependency to a mempher build: the request is one POST and the reply is one
// JSON decode. That keeps the promise the rest of this library makes -- one
// PostgreSQL database and your binary -- and it means no SDK release can change
// what an extraction sends.
//
//	completer, err := anthropic.New(anthropic.Config{Model: "claude-opus-5"})
//	ex, err := extract.New(completer, extract.Options{})
//
// Structured output goes through tool use, with the tool forced. That is the
// most reliable way to hold these models to a schema: a prose answer is not
// merely discouraged, it is unavailable.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/extract"
)

// Defaults, used when the corresponding [Config] field is zero.
const (
	// DefaultBaseURL is Anthropic's own endpoint.
	DefaultBaseURL = "https://api.anthropic.com"

	// DefaultMaxTokens is the output budget for one extraction. A reply cut
	// short by it is invalid JSON, which the extract package reports as a
	// malformed response and a worker retries.
	DefaultMaxTokens = 4096

	// APIVersion is the value of the anthropic-version header. It is pinned
	// rather than tracked: the wire format this package parses is the one that
	// version promises.
	APIVersion = "2023-06-01"
)

// toolName is what the forced tool is called. It is part of no persisted
// identity, so it may change freely.
const toolName = "record_extraction"

// maxErrorBody is how much of a failed response is quoted back in an error:
// enough to identify the failure, not enough to put a provider's entire error
// page in a log line.
const maxErrorBody = 2048

// Config assembles a [Completer].
type Config struct {
	// Model is the model id, such as "claude-opus-5". Required, and
	// deliberately without a default.
	//
	// It ends up inside the extractor id and therefore in the key of every fact
	// this produces. A default here would mean a library upgrade could silently
	// move a deployment onto another model and orphan its facts, which is the
	// one thing that id exists to prevent.
	Model string

	// APIKey is sent as the x-api-key header. Empty falls back to
	// ANTHROPIC_API_KEY.
	APIKey string

	// BaseURL is the endpoint root, without a trailing "/v1/messages". Empty
	// means [DefaultBaseURL].
	BaseURL string

	// MaxTokens is the output budget for one extraction. Zero means
	// [DefaultMaxTokens].
	MaxTokens int

	// HTTPClient issues the request. Nil means [http.DefaultClient].
	// Cancellation comes from the context either way.
	HTTPClient *http.Client
}

// Completer satisfies [extract.Completer] against the Messages API.
type Completer struct {
	client    *http.Client
	url       string
	apiKey    string
	model     string
	maxTokens int
}

var _ extract.Completer = (*Completer)(nil)

// New returns a [Completer]. It opens no connection, so it needs no context.
func New(cfg Config) (*Completer, error) {
	if cfg.Model == "" {
		return nil, fmt.Errorf("mempher/extract/anthropic: new: Model is required: %w",
			mempher.ErrInvalidConfig)
	}
	if cfg.MaxTokens < 0 {
		return nil, fmt.Errorf("mempher/extract/anthropic: new: MaxTokens is %d: %w",
			cfg.MaxTokens, mempher.ErrInvalidConfig)
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = DefaultMaxTokens
	}

	base := cfg.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	key := cfg.APIKey
	if key == "" {
		key = os.Getenv("ANTHROPIC_API_KEY")
	}
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	return &Completer{
		client:    client,
		url:       strings.TrimSuffix(base, "/") + "/v1/messages",
		apiKey:    key,
		model:     cfg.Model,
		maxTokens: cfg.MaxTokens,
	}, nil
}

// Model returns the configured model id, which the extract package folds into
// the id every fact is keyed by.
func (c *Completer) Model() string { return c.model }

type messagesRequest struct {
	Model       string     `json:"model"`
	MaxTokens   int        `json:"max_tokens"`
	Temperature float64    `json:"temperature"`
	System      string     `json:"system,omitempty"`
	Messages    []message  `json:"messages"`
	Tools       []tool     `json:"tools"`
	ToolChoice  toolChoice `json:"tool_choice"`
}

type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type toolChoice struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type messagesResponse struct {
	Content []struct {
		Type  string          `json:"type"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Error      *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// Complete sends one extraction and returns the tool input verbatim, which is
// already JSON matching the schema it was given.
func (c *Completer) Complete(ctx context.Context, p extract.Prompt) ([]byte, error) {
	body, err := json.Marshal(messagesRequest{
		Model:     c.model,
		MaxTokens: c.maxTokens,
		// Zero, and not configurable. A reclaimed lease replays a job, and an
		// extractor that answers differently the second time asserts the same
		// claim over an overlapping window -- which is ErrFactConflict, not a
		// merged answer. Sampling is not a knob worth exposing here.
		Temperature: 0,
		System:      p.System,
		Messages:    []message{{Role: "user", Content: p.User}},
		// The schema goes on the wire exactly as the extract package built it,
		// so nothing it constrains can be lost in translation.
		Tools: []tool{{
			Name:        toolName,
			Description: "Record what these episodes assert and retract.",
			InputSchema: json.RawMessage(p.Schema),
		}},
		// Forced. An unforced tool leaves the model free to answer in prose,
		// which arrives as a malformed response and costs a retry to discover.
		ToolChoice: toolChoice{Type: "tool", Name: toolName},
	})
	if err != nil {
		return nil, fmt.Errorf("mempher/extract/anthropic: build the request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("mempher/extract/anthropic: build the request: %w", err)
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("anthropic-version", APIVersion)
	if c.apiKey != "" {
		req.Header.Set("x-api-key", c.apiKey)
	}

	res, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("mempher/extract/anthropic: post %s: %w", c.url, err)
	}
	defer func() { _ = res.Body.Close() }()

	raw, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, fmt.Errorf("mempher/extract/anthropic: read the response: %w", err)
	}
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("mempher/extract/anthropic: %s returned %s: %s",
			c.url, res.Status, clip(raw))
	}

	var parsed messagesResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("mempher/extract/anthropic: decode the response: %w", err)
	}
	if parsed.Error != nil {
		return nil, fmt.Errorf("mempher/extract/anthropic: %s: %s",
			parsed.Error.Type, parsed.Error.Message)
	}
	if parsed.StopReason == "max_tokens" {
		// Reported here rather than left to arrive as malformed JSON, because
		// the fix is a number in Config and the symptom would not say so.
		return nil, fmt.Errorf("mempher/extract/anthropic: the reply was cut off at the " +
			"token budget; raise Config.MaxTokens or lower extract.Options.MaxAssertions")
	}
	for _, block := range parsed.Content {
		if block.Type == "tool_use" && block.Name == toolName {
			return block.Input, nil
		}
	}
	return nil, fmt.Errorf(
		"mempher/extract/anthropic: the reply holds no %s call, stop reason %q",
		toolName, parsed.StopReason)
}

func clip(body []byte) string {
	if len(body) > maxErrorBody {
		return string(body[:maxErrorBody]) + "..."
	}
	return string(body)
}
