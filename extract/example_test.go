package extract_test

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/extract"
)

// provider stands in for the adapter you write: about twenty lines handing a
// [extract.Prompt] to a real SDK and returning the bytes it said. Everything
// that makes the answer usable happens in the package, not here.
//
// A real one sets temperature to zero and passes p.Schema to the provider's
// structured-output mechanism -- a tool definition, response_format,
// responseSchema -- rather than pasting it into the text.
type provider struct{}

func (provider) Model() string { return "example-model" }

func (provider) Complete(_ context.Context, _ extract.Prompt) ([]byte, error) {
	return []byte(`{
		"assertions": [{
			"subject": "the user", "predicate": "lives_in", "object": "Berlin",
			"statement": "the user lives in Berlin",
			"valid_from": "2024-06-01", "confidence": 0.95
		}],
		"retractions": [
			{"fact_id": "0190b1c2-1111-7000-8000-000000000001", "at": "2024-06-01"}
		]
	}`), nil
}

// This is the whole of wiring an extractor: one adapter, one constructor, and
// the id that keys everything it asserts.
func Example() {
	ex, err := extract.New(provider{}, extract.Options{
		// A closed vocabulary is the most valuable thing here. A triple is
		// identity, so a model free to write "lives_in" today and "resides_in"
		// tomorrow supersedes nothing, ever.
		Predicates:   []mempher.Predicate{"lives_in", "allergic_to", "works_at"},
		Instructions: "A support agent: care about entitlements and past incidents.",
	})
	if err != nil {
		log.Fatal(err)
	}

	// The id composes the shipped prompt revision, the provider model, and a
	// digest of the options. Editing any of them makes a different extractor,
	// whose facts are a backfill rather than a silent mixture with the old ones.
	fmt.Println(ex.Model())

	// What a Worker passes in: the new episodes, what the scope already
	// believes, and the instant that resolves every relative date in the text.
	res, err := ex.Extract(context.Background(), mempher.ExtractRequest{
		Scope: "user:8123",
		Episodes: []mempher.Episode{{
			Seq:        42,
			Content:    "I moved to Berlin at the start of June.",
			Role:       mempher.RoleUser,
			OccurredAt: time.Date(2024, 6, 15, 9, 0, 0, 0, time.UTC),
		}},
		Known: []mempher.Fact{{
			ID:        mempher.FactID(mustUUID("0190b1c2-1111-7000-8000-000000000001")),
			Subject:   "the user",
			Predicate: "lives_in",
			Object:    "Paris",
			Statement: "the user lives in Paris",
			Valid:     mempher.Validity{From: time.Date(2019, 4, 1, 0, 0, 0, 0, time.UTC)},
		}},
		Now: time.Date(2024, 6, 15, 12, 0, 0, 0, time.UTC),
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, a := range res.Assert {
		fmt.Printf("assert   %s %s %q  since %s\n",
			a.Subject, a.Predicate, a.Object, a.Valid.From.Format(time.DateOnly))
	}
	// The window closes when the world changed, not when it was mentioned --
	// which is why "where did they live last year" still has an answer.
	for _, r := range res.Retract {
		fmt.Printf("retract  %s  at %s\n", r.Fact, r.At.Format(time.DateOnly))
	}

	// Output:
	// mempher-extract/v1+example-model+657003da
	// assert   the user lives_in "Berlin"  since 2024-06-01
	// retract  0190b1c2-1111-7000-8000-000000000001  at 2024-06-01
}

func mustUUID(s string) [16]byte {
	id, err := mempher.ParseFactID(s)
	if err != nil {
		log.Fatal(err)
	}
	return id
}
