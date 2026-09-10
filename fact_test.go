package mempher

import (
	"errors"
	"strings"
	"testing"
	"time"
)

var (
	jan = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	mar = time.Date(2026, 3, 1, 0, 0, 0, 0, time.UTC)
	jun = time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)
	sep = time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
)

// closed and open build the two shapes of window, so the tests below read as
// intervals rather than as struct literals.
func closed(from, to time.Time) Validity { return Validity{From: from, To: to} }
func open(from time.Time) Validity       { return Validity{From: from} }

func TestValidityContains(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		window Validity
		at     time.Time
		want   bool
	}{
		{"at the start", closed(mar, jun), mar, true},
		{"inside", closed(mar, jun), time.Date(2026, 4, 1, 0, 0, 0, 0, time.UTC), true},
		// Half-open: the instant a window ends belongs to whatever comes next.
		{"at the end", closed(mar, jun), jun, false},
		{"before", closed(mar, jun), jan, false},
		{"after", closed(mar, jun), sep, false},
		{"open window, long after", open(mar), sep, true},
		{"open window, before", open(mar), jan, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			if got := c.window.Contains(c.at); got != c.want {
				t.Errorf("Contains(%s) = %v, want %v", c.at.Format(time.DateOnly), got, c.want)
			}
		})
	}
}

func TestValidityOverlaps(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		a, b Validity
		want bool
	}{
		// The case the whole half-open convention exists for: a change of
		// state needs no gap between the old window and the new one.
		{"adjacent", closed(jan, mar), closed(mar, jun), false},
		{"overlapping", closed(jan, jun), closed(mar, sep), true},
		{"contained", closed(jan, sep), closed(mar, jun), true},
		{"disjoint", closed(jan, mar), closed(jun, sep), false},
		{"open against a later closed", open(jan), closed(jun, sep), true},
		{"closed against a later open", closed(jan, mar), open(jun), false},
		{"open against open", open(jan), open(sep), true},
		{"identical", closed(jan, mar), closed(jan, mar), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			if got := c.a.Overlaps(c.b); got != c.want {
				t.Errorf("a.Overlaps(b) = %v, want %v", got, c.want)
			}
			// Overlap is symmetric, and a bug that made it otherwise would
			// depend on the order an extractor happened to emit its claims.
			if got := c.b.Overlaps(c.a); got != c.want {
				t.Errorf("b.Overlaps(a) = %v, want %v", got, c.want)
			}
		})
	}
}

func TestValidityValidate(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		window Validity
		want   error
	}{
		{"open", open(jan), nil},
		{"closed", closed(jan, mar), nil},
		{"no start", Validity{To: mar}, ErrInvalidValidity},
		{"ends before it starts", closed(mar, jan), ErrInvalidValidity},
		// Holds at no instant at all, which is a mistake rather than a way to
		// say "never true".
		{"empty", closed(mar, mar), ErrInvalidValidity},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			err := c.window.Validate()
			if c.want == nil {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, c.want) {
				t.Fatalf("Validate() = %v, want %v", err, c.want)
			}
		})
	}
}

// validAssertion is the shape every case below starts from, so each test names
// only the one thing it breaks.
func validAssertion() Assertion {
	return Assertion{
		Subject:    "user",
		Predicate:  "allergic_to",
		Object:     "hazelnuts",
		Statement:  "the user is allergic to hazelnuts",
		Valid:      open(jan),
		Confidence: 0.9,
	}
}

func TestAssertionValidate(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		mutate func(*Assertion)
		want   error
	}{
		{"as built", func(*Assertion) {}, nil},
		{"zero confidence is taken at face value", func(a *Assertion) { a.Confidence = 0 }, nil},
		{"no subject", func(a *Assertion) { a.Subject = "" }, ErrInvalidFact},
		{"no predicate", func(a *Assertion) { a.Predicate = "" }, ErrInvalidFact},
		{"no object", func(a *Assertion) { a.Object = "" }, ErrInvalidFact},
		{"no statement", func(a *Assertion) { a.Statement = "  " }, ErrInvalidFact},
		// A predicate is the key that makes two extractions of one claim the
		// same claim, so "lives in" must not become a second "lives_in".
		{"predicate with a space", func(a *Assertion) { a.Predicate = "lives in" }, ErrInvalidFact},
		{"predicate with a newline", func(a *Assertion) { a.Predicate = "lives\nin" }, ErrInvalidFact},
		{"subject with a control character", func(a *Assertion) { a.Subject = "us\x00er" }, ErrInvalidFact},
		{"over-long object", func(a *Assertion) { a.Object = strings.Repeat("x", MaxObjectLen+1) }, ErrInvalidFact},
		{"over-long statement", func(a *Assertion) { a.Statement = strings.Repeat("x", MaxStatementLen+1) }, ErrInvalidFact},
		{"confidence above one", func(a *Assertion) { a.Confidence = 1.5 }, ErrInvalidFact},
		{"negative confidence", func(a *Assertion) { a.Confidence = -0.1 }, ErrInvalidFact},
		{"invalid window", func(a *Assertion) { a.Valid = closed(mar, jan) }, ErrInvalidValidity},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			a := validAssertion()
			c.mutate(&a)

			err := a.Validate()
			if c.want == nil {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, c.want) {
				t.Fatalf("Validate() = %v, want %v", err, c.want)
			}
		})
	}
}

func TestExtractCommandValidate(t *testing.T) {
	t.Parallel()

	episode := EpisodeID{1}
	base := func() ExtractCommand {
		return ExtractCommand{
			Scope:     "user:1",
			Extractor: "test",
			Episodes:  []EpisodeID{episode},
			Assert:    []Assertion{validAssertion()},
			At:        jan,
		}
	}

	cases := []struct {
		name   string
		mutate func(*ExtractCommand)
		want   error
	}{
		{"as built", func(*ExtractCommand) {}, nil},
		// The common and correct answer: most episodes hold no durable fact.
		{"empty result", func(c *ExtractCommand) { c.Assert = nil }, nil},
		{"no scope", func(c *ExtractCommand) { c.Scope = "" }, ErrInvalidScope},
		{"no extractor", func(c *ExtractCommand) { c.Extractor = "" }, ErrInvalidExtraction},
		{"no episodes", func(c *ExtractCommand) { c.Episodes = nil }, ErrInvalidExtraction},
		{"zero episode id", func(c *ExtractCommand) { c.Episodes = []EpisodeID{{}} }, ErrInvalidExtraction},
		{"no system time", func(c *ExtractCommand) { c.At = time.Time{} }, ErrInvalidExtraction},
		{"a bad assertion", func(c *ExtractCommand) { c.Assert[0].Subject = "" }, ErrInvalidFact},
		{"retraction naming no fact", func(c *ExtractCommand) {
			c.Retract = []Retraction{{At: mar}}
		}, ErrInvalidExtraction},
		{"retraction with no instant", func(c *ExtractCommand) {
			c.Retract = []Retraction{{Fact: FactID{2}}}
		}, ErrInvalidExtraction},
		// A runaway model must not be able to fill a scope in one job.
		{"too many assertions", func(c *ExtractCommand) {
			c.Assert = make([]Assertion, MaxAssertionsPerExtraction+1)
			for i := range c.Assert {
				c.Assert[i] = validAssertion()
			}
		}, ErrInvalidExtraction},
		{"too many retractions", func(c *ExtractCommand) {
			c.Retract = make([]Retraction, MaxRetractionsPerExtraction+1)
			for i := range c.Retract {
				c.Retract[i] = Retraction{Fact: FactID{byte(i + 1)}, At: mar}
			}
		}, ErrInvalidExtraction},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			cmd := base()
			c.mutate(&cmd)

			err := cmd.Validate()
			if c.want == nil {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, c.want) {
				t.Fatalf("Validate() = %v, want %v", err, c.want)
			}
		})
	}
}

func TestFactQueryValidate(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		query FactQuery
		want  error
	}{
		{"minimal", FactQuery{Scope: "s", Extractor: "e"}, nil},
		{"no scope", FactQuery{Extractor: "e"}, ErrInvalidScope},
		{"no extractor", FactQuery{Scope: "s"}, ErrInvalidConfig},
		{"bad subject", FactQuery{Scope: "s", Extractor: "e", Subjects: []Subject{""}}, ErrInvalidFact},
		{"bad predicate", FactQuery{Scope: "s", Extractor: "e", Predicates: []Predicate{"a b"}}, ErrInvalidFact},
		{"confidence above one", FactQuery{Scope: "s", Extractor: "e", MinConfidence: 2}, ErrInvalidConfig},
		{"negative limit", FactQuery{Scope: "s", Extractor: "e", Limit: -1}, ErrInvalidConfig},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			err := c.query.Validate()
			if c.want == nil {
				if err != nil {
					t.Fatalf("Validate() = %v, want nil", err)
				}
				return
			}
			if !errors.Is(err, c.want) {
				t.Fatalf("Validate() = %v, want %v", err, c.want)
			}
		})
	}
}

// TestJobKindExtractIsClosed guards the set the schema mirrors with a CHECK.
func TestJobKindExtractIsClosed(t *testing.T) {
	t.Parallel()

	if !JobKindExtract.Valid() {
		t.Error("JobKindExtract should be valid")
	}
	if JobKindExtract.String() != "extract" {
		t.Errorf("JobKindExtract = %q, want %q", JobKindExtract, "extract")
	}
	if !ChannelFact.Valid() {
		t.Error("ChannelFact should be valid")
	}
}
