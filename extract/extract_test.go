package extract

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/mempher/mempher"
)

// fake is a [Completer] that returns canned replies and records what it was
// asked, which is the only thing worth asserting about a prompt.
type fake struct {
	model   string
	replies []string
	err     error
	prompts []Prompt
}

func (f *fake) Model() string { return f.model }

func (f *fake) Complete(_ context.Context, p Prompt) ([]byte, error) {
	f.prompts = append(f.prompts, p)
	if f.err != nil {
		return nil, f.err
	}
	i := len(f.prompts) - 1
	if i < len(f.replies) {
		return []byte(f.replies[i]), nil
	}
	return []byte(`{"assertions":[],"retractions":[]}`), nil
}

func newFake(replies ...string) *fake {
	return &fake{model: "test-model", replies: replies}
}

var (
	now       = time.Date(2024, 6, 15, 12, 0, 0, 0, time.UTC)
	parisID   = mustFactID("0190b1c2-1111-7000-8000-000000000001")
	dietID    = mustFactID("0190b1c2-2222-7000-8000-000000000002")
	unknownID = "0190b1c2-9999-7000-8000-000000000099"
)

func mustFactID(s string) mempher.FactID {
	id, err := mempher.ParseFactID(s)
	if err != nil {
		panic(err)
	}
	return id
}

func quiet() *slog.Logger { return slog.New(slog.DiscardHandler) }

func newExtractor(t *testing.T, c Completer, opts Options) *Extractor {
	t.Helper()
	if opts.Logger == nil {
		opts.Logger = quiet()
	}
	e, err := New(c, opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return e
}

func request(known ...mempher.Fact) mempher.ExtractRequest {
	return mempher.ExtractRequest{
		Scope: "user:1",
		Episodes: []mempher.Episode{{
			Seq:        1,
			Content:    "I moved to Berlin in June.",
			Role:       mempher.RoleUser,
			OccurredAt: now,
		}},
		Known: known,
		Now:   now,
	}
}

func parisFact() mempher.Fact {
	return mempher.Fact{
		ID:         parisID,
		Scope:      "user:1",
		Subject:    "the user",
		Predicate:  "lives_in",
		Object:     "Paris",
		Statement:  "the user lives in Paris",
		Valid:      mempher.Validity{From: time.Date(2019, 4, 1, 0, 0, 0, 0, time.UTC)},
		Confidence: 1,
	}
}

func dietFact() mempher.Fact {
	return mempher.Fact{
		ID:         dietID,
		Scope:      "user:1",
		Subject:    "the user",
		Predicate:  "diet",
		Object:     "vegetarian",
		Statement:  "the user is vegetarian",
		Valid:      mempher.Validity{From: time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)},
		Confidence: 1,
	}
}

func TestNewRejectsUnusableConfig(t *testing.T) {
	long := strings.Repeat("m", 200)
	cases := map[string]struct {
		completer Completer
		opts      Options
	}{
		"no completer":        {nil, Options{}},
		"no model id":         {&fake{}, Options{}},
		"negative assertions": {newFake(), Options{MaxAssertions: -1}},
		"assertions over what a FactStore takes": {
			newFake(), Options{MaxAssertions: mempher.MaxAssertionsPerExtraction + 1},
		},
		"negative prompt budget": {newFake(), Options{MaxPromptBytes: -1}},
		"predicate with a space": {
			newFake(), Options{Predicates: []mempher.Predicate{"lives in"}},
		},
		"id longer than a fact key allows": {&fake{model: long}, Options{}},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := New(tc.completer, tc.opts); !errors.Is(err, mempher.ErrInvalidConfig) {
				t.Fatalf("New: got %v, want ErrInvalidConfig", err)
			}
		})
	}
}

func TestModelIdentifiesWhatProducedTheFacts(t *testing.T) {
	base := newExtractor(t, newFake(), Options{})

	if got := string(base.Model()); !strings.HasPrefix(got, "mempher-extract/v1+test-model+") {
		t.Fatalf("Model = %q, want the prompt revision and the provider model in it", got)
	}
	if len(base.Model()) > mempher.MaxExtractorLen {
		t.Fatalf("Model is %d bytes, over the %d a fact key allows",
			len(base.Model()), mempher.MaxExtractorLen)
	}

	same := newExtractor(t, newFake(), Options{})
	if base.Model() != same.Model() {
		t.Fatal("the same configuration produced two extractor ids")
	}

	// Everything that changes what the model is asked must change the id, or
	// two extractors' opinions merge silently into one scope.
	changed := map[string]Options{
		"instructions":  {Instructions: "care about allergies"},
		"vocabulary":    {Predicates: []mempher.Predicate{"lives_in"}},
		"assertion cap": {MaxAssertions: 4},
		"reconcile":     {Reconcile: true},
	}
	for name, opts := range changed {
		t.Run(name, func(t *testing.T) {
			other := newExtractor(t, newFake(), opts)
			if other.Model() == base.Model() {
				t.Fatalf("changing the %s left the extractor id unchanged", name)
			}
		})
	}

	// A vocabulary is a set, so writing it in another order is the same
	// extractor and must not orphan a scope's facts.
	a := newExtractor(t, newFake(), Options{Predicates: []mempher.Predicate{"a", "b", "b"}})
	b := newExtractor(t, newFake(), Options{Predicates: []mempher.Predicate{"b", "a"}})
	if a.Model() != b.Model() {
		t.Fatal("reordering the predicate vocabulary changed the extractor id")
	}
}

func TestExtractReadsAssertionsAndRetractions(t *testing.T) {
	c := newFake(`{
		"assertions": [{
			"subject": "the user", "predicate": "lives_in", "object": "Berlin",
			"statement": "the user lives in Berlin",
			"valid_from": "2024-06-01", "confidence": 0.9
		}],
		"retractions": [{"fact_id": "` + parisID.String() + `", "at": "2024-06-01"}]
	}`)
	e := newExtractor(t, c, Options{})

	res, err := e.Extract(t.Context(), request(parisFact()))
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if len(res.Assert) != 1 {
		t.Fatalf("got %d assertions, want 1", len(res.Assert))
	}
	got := res.Assert[0]
	if got.Subject != "the user" || got.Predicate != "lives_in" || got.Object != "Berlin" {
		t.Fatalf("triple = %q %q %q", got.Subject, got.Predicate, got.Object)
	}
	if want := time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC); !got.Valid.From.Equal(want) {
		t.Fatalf("valid from %s, want %s", got.Valid.From, want)
	}
	if !got.Valid.IsOpen() {
		t.Fatalf("window closed at %s, want still true", got.Valid.To)
	}
	if got.Confidence != 0.9 {
		t.Fatalf("confidence = %v, want 0.9", got.Confidence)
	}
	if err := got.Validate(); err != nil {
		t.Fatalf("a returned assertion a FactStore would refuse: %v", err)
	}

	if len(res.Retract) != 1 {
		t.Fatalf("got %d retractions, want 1", len(res.Retract))
	}
	// The window closes when the world changed, not when the clock says now.
	if want := time.Date(2024, 6, 1, 0, 0, 0, 0, time.UTC); !res.Retract[0].At.Equal(want) {
		t.Fatalf("retracted at %s, want %s", res.Retract[0].At, want)
	}

	// The prompt has to carry the id a retraction names, and the episodes.
	user := c.prompts[0].User
	for _, want := range []string{parisID.String(), "I moved to Berlin in June.", now.Format(time.RFC3339)} {
		if !strings.Contains(user, want) {
			t.Fatalf("prompt does not carry %q:\n%s", want, user)
		}
	}
}

func TestExtractFindingNothingIsNotAnError(t *testing.T) {
	e := newExtractor(t, newFake(`{"assertions": [], "retractions": []}`), Options{})
	res, err := e.Extract(t.Context(), request())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.Assert) != 0 || len(res.Retract) != 0 {
		t.Fatalf("got %d assertions and %d retractions, want none",
			len(res.Assert), len(res.Retract))
	}
}

func TestExtractDropsWhatAFactStoreWouldRefuse(t *testing.T) {
	good := `{
		"subject": "the user", "predicate": "lives_in", "object": "Berlin",
		"statement": "the user lives in Berlin",
		"valid_from": "2024-06-01", "confidence": 0.9
	}`
	cases := map[string]string{
		"no confidence": `{
			"subject": "the user", "predicate": "lives_in", "object": "Berlin",
			"statement": "the user lives in Berlin", "valid_from": "2024-06-01"
		}`,
		"confidence out of range": `{
			"subject": "the user", "predicate": "lives_in", "object": "Berlin",
			"statement": "the user lives in Berlin",
			"valid_from": "2024-06-01", "confidence": 4
		}`,
		"empty statement": `{
			"subject": "the user", "predicate": "lives_in", "object": "Berlin",
			"statement": "  ", "valid_from": "2024-06-01", "confidence": 0.9
		}`,
		"unparseable valid_from": `{
			"subject": "the user", "predicate": "lives_in", "object": "Berlin",
			"statement": "the user lives in Berlin",
			"valid_from": "last June", "confidence": 0.9
		}`,
		"window ending before it began": `{
			"subject": "the user", "predicate": "lives_in", "object": "Berlin",
			"statement": "the user lives in Berlin",
			"valid_from": "2024-06-01", "valid_to": "2020-01-01", "confidence": 0.9
		}`,
		"predicate carrying a space": `{
			"subject": "the user", "predicate": "lives in", "object": "Berlin",
			"statement": "the user lives in Berlin",
			"valid_from": "2024-06-01", "confidence": 0.9
		}`,
		"statement over the length limit": `{
			"subject": "the user", "predicate": "lives_in", "object": "Berlin",
			"statement": "` + strings.Repeat("x", mempher.MaxStatementLen+1) + `",
			"valid_from": "2024-06-01", "confidence": 0.9
		}`,
	}
	for name, bad := range cases {
		t.Run(name, func(t *testing.T) {
			reply := `{"assertions": [` + bad + `,` + good + `], "retractions": []}`
			e := newExtractor(t, newFake(reply), Options{})

			res, err := e.Extract(t.Context(), request())
			if err != nil {
				t.Fatalf("Extract: %v", err)
			}
			// The bad one goes; the good one beside it survives, because one
			// bad triple must not cost a turn its real facts.
			if len(res.Assert) != 1 {
				t.Fatalf("got %d assertions, want only the valid one", len(res.Assert))
			}
			if res.Assert[0].Object != "Berlin" || res.Assert[0].Predicate != "lives_in" {
				t.Fatalf("kept the wrong assertion: %+v", res.Assert[0])
			}
		})
	}
}

func TestExtractDropsRetractionsItCannotTrust(t *testing.T) {
	cases := map[string]string{
		"a fact that was never shown": unknownID,
		"not an id at all":            "the paris one",
	}
	for name, id := range cases {
		t.Run(name, func(t *testing.T) {
			reply := `{"assertions": [], "retractions": [{"fact_id": "` + id +
				`", "at": "2024-06-01"}]}`
			e := newExtractor(t, newFake(reply), Options{})

			res, err := e.Extract(t.Context(), request(parisFact()))
			if err != nil {
				t.Fatalf("Extract: %v", err)
			}
			if len(res.Retract) != 0 {
				t.Fatalf("kept %d retractions, want none", len(res.Retract))
			}
		})
	}

	t.Run("closing before the window opened", func(t *testing.T) {
		// Clamping would invent a window the world never had, and the store
		// would accept it without complaint.
		reply := `{"assertions": [], "retractions": [{"fact_id": "` + parisID.String() +
			`", "at": "2001-01-01"}]}`
		e := newExtractor(t, newFake(reply), Options{})

		res, err := e.Extract(t.Context(), request(parisFact()))
		if err != nil {
			t.Fatalf("Extract: %v", err)
		}
		if len(res.Retract) != 0 {
			t.Fatalf("kept %d retractions, want none", len(res.Retract))
		}
	})
}

func TestVocabularyIsEnforcedInTheSchemaAndAgainInGo(t *testing.T) {
	opts := Options{Predicates: []mempher.Predicate{"lives_in", "allergic_to"}}
	reply := `{"assertions": [{
		"subject": "the user", "predicate": "enjoys", "object": "hiking",
		"statement": "the user enjoys hiking",
		"valid_from": "2024-06-01", "confidence": 0.9
	}], "retractions": []}`
	e := newExtractor(t, newFake(reply), opts)

	var schema map[string]any
	if err := json.Unmarshal(e.Schema(), &schema); err != nil {
		t.Fatalf("the schema is not JSON: %v", err)
	}
	if !strings.Contains(string(e.Schema()), `"enum":["allergic_to","lives_in"]`) {
		t.Fatalf("the vocabulary is not an enum in the schema:\n%s", e.Schema())
	}

	res, err := e.Extract(t.Context(), request())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.Assert) != 0 {
		t.Fatalf("kept an out-of-vocabulary predicate: %+v", res.Assert)
	}
}

func TestMaxAssertionsCapsWhatOneExtractionRecords(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"assertions": [`)
	for i := range 10 {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(`{"subject": "the user", "predicate": "likes", "object": "thing` +
			string(rune('a'+i)) + `", "statement": "the user likes a thing",
			"valid_from": "2024-06-01", "confidence": 0.9}`)
	}
	b.WriteString(`], "retractions": []}`)

	e := newExtractor(t, newFake(b.String()), Options{MaxAssertions: 3})
	res, err := e.Extract(t.Context(), request())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.Assert) != 3 {
		t.Fatalf("got %d assertions, want the cap of 3", len(res.Assert))
	}
	if !strings.Contains(e.prompt(), "at most 3 assertions") {
		t.Fatal("the cap is enforced but never asked for")
	}
}

func TestEpisodesOverTheBudgetFailRatherThanBeingTrimmed(t *testing.T) {
	// An extraction marks every episode the job named as read, so trimming
	// would leave those episodes marked extracted and never extracted.
	c := newFake()
	e := newExtractor(t, c, Options{MaxPromptBytes: 64})

	req := request()
	req.Episodes[0].Content = strings.Repeat("x", 100)

	if _, err := e.Extract(t.Context(), req); !errors.Is(err, ErrPromptTooLarge) {
		t.Fatalf("Extract: got %v, want ErrPromptTooLarge", err)
	}
	if len(c.prompts) != 0 {
		t.Fatal("an over-budget extraction still called the model")
	}
}

func TestMalformedResponsesAreReportedNotGuessedAt(t *testing.T) {
	e := newExtractor(t, newFake(`not json at all`), Options{})
	if _, err := e.Extract(t.Context(), request()); !errors.Is(err, ErrMalformedResponse) {
		t.Fatalf("Extract: got %v, want ErrMalformedResponse", err)
	}
}

func TestCompleterFailuresReachTheWorker(t *testing.T) {
	sentinel := errors.New("provider is down")
	e := newExtractor(t, &fake{model: "test-model", err: sentinel}, Options{})
	if _, err := e.Extract(t.Context(), request()); !errors.Is(err, sentinel) {
		t.Fatalf("Extract: got %v, want the provider's error", err)
	}
}

func TestReconcileShowsTheExtractionOnlyTheRelevantFacts(t *testing.T) {
	topics := `{"subjects": ["the user"], "predicates": ["lives_in"]}`
	extraction := `{"assertions": [], "retractions": [{"fact_id": "` +
		parisID.String() + `", "at": "2024-06-01"}]}`
	c := newFake(topics, extraction)
	e := newExtractor(t, c, Options{Reconcile: true})

	// One fact shares the subject, one shares neither.
	elsewhere := dietFact()
	elsewhere.Subject = "project:apollo"
	elsewhere.Predicate = "status"

	res, err := e.Extract(t.Context(), request(parisFact(), elsewhere))
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(c.prompts) != 2 {
		t.Fatalf("made %d model calls, want 2", len(c.prompts))
	}

	second := c.prompts[1].User
	if !strings.Contains(second, parisID.String()) {
		t.Fatal("the extraction was not shown the fact it had to supersede")
	}
	if strings.Contains(second, elsewhere.ID.String()) {
		t.Fatal("the extraction was shown a fact the episodes do not touch")
	}
	if len(res.Retract) != 1 {
		t.Fatalf("got %d retractions, want 1", len(res.Retract))
	}
}

func TestReconcileFallsBackWhenTheFirstPassNamesNothing(t *testing.T) {
	// A flaky routing call must not silently switch supersession off, which
	// would look exactly like a scope where nothing ever changes.
	c := newFake(`{"subjects": [], "predicates": []}`,
		`{"assertions": [], "retractions": []}`)
	e := newExtractor(t, c, Options{Reconcile: true})

	if _, err := e.Extract(t.Context(), request(parisFact())); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if !strings.Contains(c.prompts[1].User, parisID.String()) {
		t.Fatal("an empty first pass hid the known facts from the extraction")
	}
}

func TestReconcileSkipsTheFirstPassWithNothingToReconcile(t *testing.T) {
	c := newFake(`{"assertions": [], "retractions": []}`)
	e := newExtractor(t, c, Options{Reconcile: true})

	if _, err := e.Extract(t.Context(), request()); err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(c.prompts) != 1 {
		t.Fatalf("made %d model calls on a scope with no known facts, want 1", len(c.prompts))
	}
}

func TestDatesResolveInTheInjectedInstantsZone(t *testing.T) {
	// Reading a wall clock or assuming UTC would make a replay produce a
	// different window from the extraction it replaces.
	berlin := time.FixedZone("CEST", 2*60*60)
	reply := `{"assertions": [{
		"subject": "the user", "predicate": "lives_in", "object": "Berlin",
		"statement": "the user lives in Berlin",
		"valid_from": "2024-06-01", "confidence": 1
	}], "retractions": []}`
	e := newExtractor(t, newFake(reply), Options{})

	req := request()
	req.Now = now.In(berlin)

	res, err := e.Extract(t.Context(), req)
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	want := time.Date(2024, 6, 1, 0, 0, 0, 0, berlin)
	if !res.Assert[0].Valid.From.Equal(want) {
		t.Fatalf("valid from %s, want %s", res.Assert[0].Valid.From, want)
	}
}

func TestInstructionsReachThePrompt(t *testing.T) {
	e := newExtractor(t, newFake(), Options{Instructions: "care about entitlements"})
	if !strings.Contains(e.prompt(), "care about entitlements") {
		t.Fatal("the domain addendum never reaches the model")
	}
}

func TestPromptRevisionIsPinnedToTheTextItShips(t *testing.T) {
	// PromptVersion has to move when the prompt or the schema does, and nothing
	// else can notice that it has not. The digest inside an extractor id is
	// built from Options, which is the caller's input; the text shipped here is
	// not in it, deliberately, so that fixing a typo in a comment does not put
	// every deployment through a backfill.
	//
	// That leaves the revision resting on the discipline of whoever edits the
	// prompt, which is exactly the kind of invariant this library does not
	// leave to discipline anywhere else. So it is pinned.
	//
	// If this test fails you changed what the model is asked. That is a real
	// change -- it changes what gets asserted -- so bump PromptVersion and
	// record the new digest here in the same commit. Facts from the old
	// revision then stay the old revision's until ops.Backfill re-derives them,
	// which is the intended cost and the whole reason the revision is in the id.
	const (
		pinnedRevision = 1
		pinnedDigest   = "ea4099386a43a723"
	)

	schema, err := extractionSchema(Options{MaxAssertions: DefaultMaxAssertions})
	if err != nil {
		t.Fatalf("extractionSchema: %v", err)
	}
	topics, err := topicsSchema()
	if err != nil {
		t.Fatalf("topicsSchema: %v", err)
	}

	h := sha256.New()
	for _, part := range []string{extractionSystem, topicsPrompt, string(schema), string(topics)} {
		_, _ = fmt.Fprintf(h, "%d:%s", len(part), part)
	}
	got := hex.EncodeToString(h.Sum(nil))[:16]

	if PromptVersion != pinnedRevision {
		t.Fatalf("PromptVersion is %d but this test pins %d; update both together",
			PromptVersion, pinnedRevision)
	}
	if got != pinnedDigest {
		t.Fatalf("the shipped prompt or schema changed (digest %s, pinned %s) "+
			"but PromptVersion is still %d", got, pinnedDigest, PromptVersion)
	}
}
