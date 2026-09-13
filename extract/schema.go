// The response schemas, generated from the configuration rather than fixed.

package extract

import (
	"encoding/json"
	"fmt"

	"github.com/mempher/mempher"
)

// jsonSchema is the subset of JSON Schema every provider's structured-output
// mode understands. It is deliberately small: anything cleverer is a feature one
// provider has and another silently ignores, and a constraint that is only
// sometimes enforced is worse than one asserted in Go.
type jsonSchema struct {
	Type                 string                 `json:"type"`
	Description          string                 `json:"description,omitempty"`
	Properties           map[string]*jsonSchema `json:"properties,omitempty"`
	Required             []string               `json:"required,omitempty"`
	AdditionalProperties *bool                  `json:"additionalProperties,omitempty"`
	Items                *jsonSchema            `json:"items,omitempty"`
	Enum                 []string               `json:"enum,omitempty"`
	MaxItems             *int                   `json:"maxItems,omitempty"`
	MaxLength            *int                   `json:"maxLength,omitempty"`
	Minimum              *float64               `json:"minimum,omitempty"`
	Maximum              *float64               `json:"maximum,omitempty"`
}

func ptr[T any](v T) *T { return &v }

// extractionSchema is the shape of a reply to the extraction call.
//
// The length caps mirror the ones a [mempher.FactStore] enforces, so a model
// that would have produced an unstorable fact is stopped by the provider rather
// than by a dropped assertion and a log line. The predicate enum does the same
// for [Options.Predicates], which is the whole reason this is generated from the
// configuration instead of being a constant.
func extractionSchema(opts Options) (json.RawMessage, error) {
	predicate := &jsonSchema{
		Type:      "string",
		MaxLength: ptr(mempher.MaxPredicateLen),
		Description: "The relation, as a lowercase snake_case slug with no spaces. " +
			"Reuse one that already appears in the known facts before inventing one.",
	}
	if len(opts.Predicates) > 0 {
		predicate.Enum = make([]string, len(opts.Predicates))
		for i, p := range opts.Predicates {
			predicate.Enum[i] = string(p)
		}
		predicate.Description = "The relation. Use one of the listed values, or omit the " +
			"assertion entirely if none of them fits."
	}

	assertion := &jsonSchema{
		Type:                 "object",
		AdditionalProperties: ptr(false),
		Required: []string{
			"subject", "predicate", "object", "statement", "valid_from", "confidence",
		},
		Properties: map[string]*jsonSchema{
			"subject": {
				Type:      "string",
				MaxLength: ptr(mempher.MaxSubjectLen),
				Description: "What the claim is about, spelled identically in every " +
					"assertion about the same entity.",
			},
			"predicate": predicate,
			"object": {
				Type:        "string",
				MaxLength:   ptr(mempher.MaxObjectLen),
				Description: "The value of the relation. One value per assertion.",
			},
			"statement": {
				Type:      "string",
				MaxLength: ptr(mempher.MaxStatementLen),
				Description: "The claim as one plain sentence. This is what a model is " +
					"later shown and what full-text search matches.",
			},
			"valid_from": {
				Type: "string",
				Description: "When the claim became true in the world, as YYYY-MM-DD or " +
					"RFC 3339. Not when it was mentioned.",
			},
			"valid_to": {
				Type: "string",
				Description: "When the claim stopped being true, as YYYY-MM-DD or RFC 3339. " +
					"Omit it unless one episode describes a span that has already ended.",
			},
			"confidence": {
				Type:        "number",
				Minimum:     ptr(0.0),
				Maximum:     ptr(1.0),
				Description: "How sure you are, from 0 to 1. Required.",
			},
		},
	}

	retraction := &jsonSchema{
		Type:                 "object",
		AdditionalProperties: ptr(false),
		Required:             []string{"fact_id", "at"},
		Properties: map[string]*jsonSchema{
			"fact_id": {
				Type:        "string",
				Description: "The id of a known fact, copied exactly from the list you were shown.",
			},
			"at": {
				Type: "string",
				Description: "When the fact stopped being true, as YYYY-MM-DD or RFC 3339. " +
					"The moment the world changed, not the moment it was mentioned.",
			},
		},
	}

	root := &jsonSchema{
		Type:                 "object",
		AdditionalProperties: ptr(false),
		Required:             []string{"assertions", "retractions"},
		Properties: map[string]*jsonSchema{
			"assertions": {
				Type:        "array",
				Items:       assertion,
				MaxItems:    ptr(opts.MaxAssertions),
				Description: "Durable claims to record. An empty array is a good, common answer.",
			},
			"retractions": {
				Type:     "array",
				Items:    retraction,
				MaxItems: ptr(mempher.MaxRetractionsPerExtraction),
				Description: "Known facts these episodes contradict. An empty array is a " +
					"good, common answer.",
			},
		},
	}
	return marshal(root)
}

// topicsSchema is the shape of a reply to the first pass of [Options.Reconcile]:
// a routing answer, small enough that the extra call is cheap.
func topicsSchema() (json.RawMessage, error) {
	return marshal(&jsonSchema{
		Type:                 "object",
		AdditionalProperties: ptr(false),
		Required:             []string{"subjects", "predicates"},
		Properties: map[string]*jsonSchema{
			"subjects": {
				Type:  "array",
				Items: &jsonSchema{Type: "string"},
				Description: "Every subject these episodes say anything about, whether or " +
					"not they make a new claim about it.",
			},
			"predicates": {
				Type:  "array",
				Items: &jsonSchema{Type: "string"},
				Description: "Every relation these episodes touch, as lowercase snake_case " +
					"slugs, whether or not they make a new claim.",
			},
		},
	})
}

func marshal(s *jsonSchema) (json.RawMessage, error) {
	raw, err := json.Marshal(s)
	if err != nil {
		return nil, fmt.Errorf("mempher/extract: build response schema: %w", err)
	}
	return raw, nil
}
