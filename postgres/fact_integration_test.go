package postgres_test

import (
	"errors"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/ops"
	"github.com/mempher/mempher/postgres"
)

// Event-time instants the fact tests reason about. They are far from epoch, the
// system time the episodes are ingested at, so a test that confused the two axes
// fails rather than passing by coincidence.
var (
	y2019 = time.Date(2019, 1, 1, 0, 0, 0, 0, time.UTC)
	y2023 = time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
	y2024 = time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
)

const testExtractor mempher.ExtractorID = "test-extractor@1"

// seedEpisode appends one episode and returns its id, so a fact has real
// provenance and the extraction marker has a row to point at.
func seedEpisode(t *testing.T, store *postgres.Store, scope mempher.ScopeID, content string) mempher.EpisodeID {
	t.Helper()
	result, err := store.Append(t.Context(), episode(scope, content, epoch))
	if err != nil {
		t.Fatalf("Append %q: %v", content, err)
	}
	return result.Episode.ID
}

// lives builds the claim the supersession tests move through time.
func lives(object string, valid mempher.Validity) mempher.Assertion {
	return mempher.Assertion{
		Subject:    "user",
		Predicate:  "lives_in",
		Object:     object,
		Statement:  "the user lives in " + object,
		Valid:      valid,
		Confidence: 0.9,
	}
}

// currentFacts reads what the scope believes now.
func currentFacts(t *testing.T, store *postgres.Store, scope mempher.ScopeID) []mempher.Fact {
	t.Helper()
	facts, err := store.Facts(t.Context(), mempher.FactQuery{
		Scope:     scope,
		Extractor: testExtractor,
		AsOf:      epoch.Add(time.Hour),
		At:        time.Now(),
	})
	if err != nil {
		t.Fatalf("Facts: %v", err)
	}
	return facts
}

func TestApplyExtractionWritesFacts(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	id := seedEpisode(t, store, "user:1", "I live in Paris")

	facts, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope:     "user:1",
		Extractor: testExtractor,
		Episodes:  []mempher.EpisodeID{id},
		Assert:    []mempher.Assertion{lives("Paris", mempher.Validity{From: y2019})},
		At:        epoch,
	})
	if err != nil {
		t.Fatalf("ApplyExtraction: %v", err)
	}
	if len(facts) != 1 {
		t.Fatalf("got %d facts, want 1", len(facts))
	}

	got := facts[0]
	switch {
	case got.ID.IsZero():
		t.Error("the fact has no id")
	case got.Subject != "user" || got.Predicate != "lives_in" || got.Object != "Paris":
		t.Errorf("triple = (%q, %q, %q)", got.Subject, got.Predicate, got.Object)
	case !got.Valid.From.Equal(y2019):
		t.Errorf("Valid.From = %s, want %s", got.Valid.From, y2019)
	case !got.Valid.IsOpen():
		t.Errorf("Valid.To = %s, want an open window", got.Valid.To)
	case got.Extractor != testExtractor:
		t.Errorf("Extractor = %q, want %q", got.Extractor, testExtractor)
	case len(got.Episodes) != 1 || got.Episodes[0] != id:
		t.Errorf("Episodes = %v, want [%s]", got.Episodes, id)
	case !got.AssertedAt.Equal(epoch):
		t.Errorf("AssertedAt = %s, want %s", got.AssertedAt, epoch)
	}
}

// TestApplyExtractionIsIdempotent is the property at-least-once leasing needs:
// a reclaimed job must be free to replay.
func TestApplyExtractionIsIdempotent(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	id := seedEpisode(t, store, "user:1", "I live in Paris")

	cmd := mempher.ExtractCommand{
		Scope:     "user:1",
		Extractor: testExtractor,
		Episodes:  []mempher.EpisodeID{id},
		Assert:    []mempher.Assertion{lives("Paris", mempher.Validity{From: y2019})},
		At:        epoch,
	}
	first, err := store.ApplyExtraction(t.Context(), cmd)
	if err != nil {
		t.Fatalf("first ApplyExtraction: %v", err)
	}

	// A different answer from the same episodes, as a nondeterministic model
	// would give. The episodes have been read, so the first result stands and
	// the second is declined rather than fought with.
	cmd.Assert = []mempher.Assertion{lives("Berlin", mempher.Validity{From: y2023})}
	second, err := store.ApplyExtraction(t.Context(), cmd)
	if err != nil {
		t.Fatalf("second ApplyExtraction: %v", err)
	}

	if len(second) != len(first) || len(second) != 1 {
		t.Fatalf("got %d facts on replay, want the original 1", len(second))
	}
	if second[0].ID != first[0].ID || second[0].Object != "Paris" {
		t.Errorf("replay changed the fact to %q (%s), want the original Paris (%s)",
			second[0].Object, second[0].ID, first[0].ID)
	}
	if got := currentFacts(t, store, "user:1"); len(got) != 1 {
		t.Errorf("scope holds %d facts after a replay, want 1", len(got))
	}
}

// TestApplyExtractionSupersedes is what the temporal design is for: the world
// changed, and the claim that was true stays answerable.
func TestApplyExtractionSupersedes(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	first := seedEpisode(t, store, "user:1", "I live in Paris")

	paris, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope:     "user:1",
		Extractor: testExtractor,
		Episodes:  []mempher.EpisodeID{first},
		Assert:    []mempher.Assertion{lives("Paris", mempher.Validity{From: y2019})},
		At:        epoch,
	})
	if err != nil {
		t.Fatalf("assert Paris: %v", err)
	}

	// A second episode contradicts it, so the extractor closes the old window
	// and opens a new one from the same instant. Half-open bounds mean the two
	// meet exactly, with no gap and no overlap.
	second := seedEpisode(t, store, "user:1", "I moved to Berlin")
	if _, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope:     "user:1",
		Extractor: testExtractor,
		Episodes:  []mempher.EpisodeID{second},
		Assert:    []mempher.Assertion{lives("Berlin", mempher.Validity{From: y2024})},
		Retract:   []mempher.Retraction{{Fact: paris[0].ID, At: y2024}},
		At:        epoch.Add(time.Hour),
	}); err != nil {
		t.Fatalf("assert Berlin and retract Paris: %v", err)
	}

	now := currentFacts(t, store, "user:1")
	if len(now) != 1 || now[0].Object != "Berlin" {
		t.Fatalf("currently believed = %v, want only Berlin", objects(now))
	}

	// The past is still answerable, which is the whole point of not deleting.
	then, err := store.Facts(t.Context(), mempher.FactQuery{
		Scope:     "user:1",
		Extractor: testExtractor,
		AsOf:      epoch.Add(2 * time.Hour),
		At:        y2023,
	})
	if err != nil {
		t.Fatalf("Facts as of 2023: %v", err)
	}
	if len(then) != 1 || then[0].Object != "Paris" {
		t.Fatalf("believed in 2023 = %v, want only Paris", objects(then))
	}
	if !then[0].Valid.To.Equal(y2024) {
		t.Errorf("Paris window ends at %s, want %s", then[0].Valid.To, y2024)
	}

	// Both windows, from one query, when the caller asks for the whole history.
	all, err := store.Facts(t.Context(), mempher.FactQuery{
		Scope: "user:1", Extractor: testExtractor,
		AsOf: epoch.Add(2 * time.Hour), At: time.Now(), IncludeClosed: true,
	})
	if err != nil {
		t.Fatalf("Facts including closed: %v", err)
	}
	if len(all) != 2 {
		t.Errorf("history = %v, want both Berlin and Paris", objects(all))
	}
}

// TestApplyExtractionRejectsOverlap is the database refusing to believe one
// claim over two windows at once.
func TestApplyExtractionRejectsOverlap(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	first := seedEpisode(t, store, "user:1", "I live in Paris")
	second := seedEpisode(t, store, "user:1", "still in Paris")

	if _, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope: "user:1", Extractor: testExtractor,
		Episodes: []mempher.EpisodeID{first},
		Assert:   []mempher.Assertion{lives("Paris", mempher.Validity{From: y2019})},
		At:       epoch,
	}); err != nil {
		t.Fatalf("assert Paris: %v", err)
	}

	// The same claim over a window that overlaps the first, from episodes that
	// have not been read: a real contradiction, not a replay.
	_, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope: "user:1", Extractor: testExtractor,
		Episodes: []mempher.EpisodeID{second},
		Assert:   []mempher.Assertion{lives("Paris", mempher.Validity{From: y2023})},
		At:       epoch.Add(time.Hour),
	})
	if !errors.Is(err, mempher.ErrFactConflict) {
		t.Fatalf("err = %v, want ErrFactConflict", err)
	}
}

// TestApplyExtractionRejectsSelfContradiction catches the model that disagrees
// with itself inside one answer, before the statement that would insert both.
func TestApplyExtractionRejectsSelfContradiction(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	id := seedEpisode(t, store, "user:1", "I live in Paris")

	_, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope: "user:1", Extractor: testExtractor,
		Episodes: []mempher.EpisodeID{id},
		Assert: []mempher.Assertion{
			lives("Paris", mempher.Validity{From: y2019}),
			lives("Paris", mempher.Validity{From: y2023}),
		},
		At: epoch,
	})
	if !errors.Is(err, mempher.ErrFactConflict) {
		t.Fatalf("err = %v, want ErrFactConflict", err)
	}
}

// TestApplyExtractionDropsExactRepeats: a model repeating itself is a nuisance,
// not a contradiction.
func TestApplyExtractionDropsExactRepeats(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	id := seedEpisode(t, store, "user:1", "I live in Paris")

	facts, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope: "user:1", Extractor: testExtractor,
		Episodes: []mempher.EpisodeID{id},
		Assert: []mempher.Assertion{
			lives("Paris", mempher.Validity{From: y2019}),
			lives("Paris", mempher.Validity{From: y2019}),
		},
		At: epoch,
	})
	if err != nil {
		t.Fatalf("ApplyExtraction: %v", err)
	}
	if len(facts) != 1 {
		t.Fatalf("got %d facts, want the repeat collapsed into 1", len(facts))
	}
}

// TestApplyExtractionKeepsTwoObjectsOfOnePredicate: you can be allergic to more
// than one thing, and only the extractor knows when a claim replaces another.
func TestApplyExtractionKeepsTwoObjectsOfOnePredicate(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	id := seedEpisode(t, store, "user:1", "allergies")

	allergy := func(object string) mempher.Assertion {
		return mempher.Assertion{
			Subject: "user", Predicate: "allergic_to", Object: object,
			Statement: "the user is allergic to " + object,
			Valid:     mempher.Validity{From: y2019}, Confidence: 1,
		}
	}
	facts, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope: "user:1", Extractor: testExtractor,
		Episodes: []mempher.EpisodeID{id},
		Assert:   []mempher.Assertion{allergy("hazelnuts"), allergy("shellfish")},
		At:       epoch,
	})
	if err != nil {
		t.Fatalf("ApplyExtraction: %v", err)
	}
	if len(facts) != 2 {
		t.Fatalf("got %v, want both allergies", objects(facts))
	}
}

func TestApplyExtractionRejectsBadCommands(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	id := seedEpisode(t, store, "user:1", "I live in Paris")

	cases := []struct {
		name string
		cmd  mempher.ExtractCommand
		want error
	}{
		{
			name: "an episode from another scope",
			cmd: mempher.ExtractCommand{
				Scope: "user:2", Extractor: testExtractor,
				Episodes: []mempher.EpisodeID{id}, At: epoch,
			},
			want: mempher.ErrNotFound,
		},
		{
			name: "an episode that does not exist",
			cmd: mempher.ExtractCommand{
				Scope: "user:1", Extractor: testExtractor,
				Episodes: []mempher.EpisodeID{{9}}, At: epoch,
			},
			want: mempher.ErrNotFound,
		},
		{
			name: "no extractor",
			cmd: mempher.ExtractCommand{
				Scope: "user:1", Episodes: []mempher.EpisodeID{id}, At: epoch,
			},
			want: mempher.ErrInvalidExtraction,
		},
		{
			name: "no system time",
			cmd: mempher.ExtractCommand{
				Scope: "user:1", Extractor: testExtractor, Episodes: []mempher.EpisodeID{id},
			},
			want: mempher.ErrInvalidExtraction,
		},
		{
			name: "retracting a fact that is not there",
			cmd: mempher.ExtractCommand{
				Scope: "user:1", Extractor: testExtractor,
				Episodes: []mempher.EpisodeID{id}, At: epoch,
				Retract: []mempher.Retraction{{Fact: mempher.FactID{7}, At: y2024}},
			},
			want: mempher.ErrNotFound,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			if _, err := store.ApplyExtraction(t.Context(), c.cmd); !errors.Is(err, c.want) {
				t.Fatalf("err = %v, want %v", err, c.want)
			}
		})
	}
}

// TestRetractBeforeAFactBegan: a window cannot end before it starts, and saying
// so beats writing a row that holds at no instant.
func TestRetractBeforeAFactBegan(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	first := seedEpisode(t, store, "user:1", "I live in Paris")
	second := seedEpisode(t, store, "user:1", "and again")

	facts, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope: "user:1", Extractor: testExtractor,
		Episodes: []mempher.EpisodeID{first},
		Assert:   []mempher.Assertion{lives("Paris", mempher.Validity{From: y2023})},
		At:       epoch,
	})
	if err != nil {
		t.Fatalf("assert Paris: %v", err)
	}

	_, err = store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope: "user:1", Extractor: testExtractor,
		Episodes: []mempher.EpisodeID{second}, At: epoch.Add(time.Hour),
		Retract: []mempher.Retraction{{Fact: facts[0].ID, At: y2019}},
	})
	if !errors.Is(err, mempher.ErrInvalidValidity) {
		t.Fatalf("err = %v, want ErrInvalidValidity", err)
	}
}

func TestFactsFilters(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	id := seedEpisode(t, store, "user:1", "about the user")

	assertions := []mempher.Assertion{
		{
			Subject: "user", Predicate: "lives_in", Object: "Paris",
			Statement: "the user lives in Paris",
			Valid:     mempher.Validity{From: y2019}, Confidence: 0.9,
		},
		{
			Subject: "user", Predicate: "drinks", Object: "espresso",
			Statement: "the user drinks espresso",
			Valid:     mempher.Validity{From: y2019}, Confidence: 0.4,
		},
		{
			Subject: "project:apollo", Predicate: "status", Object: "shipped",
			Statement: "project apollo has shipped",
			Valid:     mempher.Validity{From: y2019}, Confidence: 1,
		},
	}
	if _, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope: "user:1", Extractor: testExtractor,
		Episodes: []mempher.EpisodeID{id}, Assert: assertions, At: epoch,
	}); err != nil {
		t.Fatalf("ApplyExtraction: %v", err)
	}

	base := mempher.FactQuery{
		Scope: "user:1", Extractor: testExtractor,
		AsOf: epoch.Add(time.Hour), At: time.Now(),
	}
	cases := []struct {
		name  string
		query func(*mempher.FactQuery)
		want  []string
	}{
		// Ordered by subject, then predicate, then object, so the same query
		// twice returns the same list.
		{"everything", func(*mempher.FactQuery) {}, []string{"shipped", "espresso", "Paris"}},
		{"by subject", func(q *mempher.FactQuery) {
			q.Subjects = []mempher.Subject{"project:apollo"}
		}, []string{"shipped"}},
		{"by predicate", func(q *mempher.FactQuery) {
			q.Predicates = []mempher.Predicate{"lives_in", "drinks"}
		}, []string{"espresso", "Paris"}},
		{"by confidence", func(q *mempher.FactQuery) { q.MinConfidence = 0.5 },
			[]string{"shipped", "Paris"}},
		// A query text orders, it never excludes: what is true about a scope
		// is true whether or not the question mentioned it. Only one of the
		// two words matches anything, which must still be enough to rank --
		// AND semantics here would score every fact zero.
		{"ranked by text but not filtered", func(q *mempher.FactQuery) { q.Text = "coffee espresso" },
			[]string{"espresso", "shipped", "Paris"}},
		{"text of nothing but stop words", func(q *mempher.FactQuery) { q.Text = "the and of" },
			[]string{"shipped", "espresso", "Paris"}},
		{"an unrelated extractor sees nothing", func(q *mempher.FactQuery) {
			q.Extractor = "someone-else"
		}, nil},
		{"an as-of before the facts were learned", func(q *mempher.FactQuery) {
			q.AsOf = epoch.Add(-time.Hour)
		}, nil},
		{"an instant before the facts were true", func(q *mempher.FactQuery) {
			q.At = time.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC)
		}, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			q := base
			c.query(&q)

			facts, err := store.Facts(t.Context(), q)
			if err != nil {
				t.Fatalf("Facts: %v", err)
			}
			got := objects(facts)
			if len(got) != len(c.want) {
				t.Fatalf("got %v, want %v", got, c.want)
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Fatalf("got %v, want %v", got, c.want)
				}
			}
		})
	}
}

// TestFactIsScopedLikeAnEpisode: isolation is reported as absence, never as a
// permission error.
func TestFactIsScopedLikeAnEpisode(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	id := seedEpisode(t, store, "user:1", "I live in Paris")

	facts, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
		Scope: "user:1", Extractor: testExtractor,
		Episodes: []mempher.EpisodeID{id},
		Assert:   []mempher.Assertion{lives("Paris", mempher.Validity{From: y2019})},
		At:       epoch,
	})
	if err != nil {
		t.Fatalf("ApplyExtraction: %v", err)
	}

	got, err := store.Fact(t.Context(), "user:1", facts[0].ID)
	if err != nil {
		t.Fatalf("Fact: %v", err)
	}
	if got.ID != facts[0].ID {
		t.Errorf("Fact returned %s, want %s", got.ID, facts[0].ID)
	}

	if _, err := store.Fact(t.Context(), "user:2", facts[0].ID); !errors.Is(err, mempher.ErrNotFound) {
		t.Errorf("cross-scope Fact = %v, want ErrNotFound", err)
	}
	if _, err := store.Fact(t.Context(), "user:1", mempher.FactID{4}); !errors.Is(err, mempher.ErrNotFound) {
		t.Errorf("unknown Fact = %v, want ErrNotFound", err)
	}
}

// TestPendingExtractions is the safety net behind the queue: it must find work
// the queue lost, and must stop reporting it once an extractor has read it --
// including when that yielded no facts at all.
func TestPendingExtractions(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	first := seedEpisode(t, store, "user:1", "I live in Paris")
	second := seedEpisode(t, store, "user:1", "ok, thanks")

	pending, err := store.PendingExtractions(t.Context(),
		ops.PendingExtractions{Extractor: testExtractor})
	if err != nil {
		t.Fatalf("PendingExtractions: %v", err)
	}
	if len(pending) != 2 {
		t.Fatalf("got %d pending, want both episodes", len(pending))
	}

	// The second episode yields nothing, which is the common case. Without the
	// marker it would be re-read, at model prices, on every pass.
	for _, id := range []mempher.EpisodeID{first, second} {
		var assertions []mempher.Assertion
		if id == first {
			assertions = []mempher.Assertion{lives("Paris", mempher.Validity{From: y2019})}
		}
		if _, err := store.ApplyExtraction(t.Context(), mempher.ExtractCommand{
			Scope: "user:1", Extractor: testExtractor,
			Episodes: []mempher.EpisodeID{id}, Assert: assertions, At: epoch,
		}); err != nil {
			t.Fatalf("ApplyExtraction: %v", err)
		}
	}

	pending, err = store.PendingExtractions(t.Context(),
		ops.PendingExtractions{Extractor: testExtractor})
	if err != nil {
		t.Fatalf("PendingExtractions: %v", err)
	}
	if len(pending) != 0 {
		t.Errorf("got %d pending after extracting both, want none", len(pending))
	}

	// A different extractor has read nothing, so changing extractor is a
	// backfill rather than a migration.
	pending, err = store.PendingExtractions(t.Context(),
		ops.PendingExtractions{Extractor: "test-extractor@2"})
	if err != nil {
		t.Fatalf("PendingExtractions: %v", err)
	}
	if len(pending) != 2 {
		t.Errorf("a new extractor sees %d pending, want both episodes", len(pending))
	}
}

// objects renders a fact list for an assertion message.
func objects(facts []mempher.Fact) []string {
	out := make([]string, len(facts))
	for i, f := range facts {
		out[i] = f.Object
	}
	return out
}
