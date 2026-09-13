package postgres_test

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/postgres"
)

// seed appends an episode and encodes it with v, which is what a worker would do.
func seed(
	t *testing.T,
	store *postgres.Store,
	scope mempher.ScopeID,
	content string,
	at time.Time,
	v mempher.Vector,
) mempher.EpisodeID {
	t.Helper()
	got, err := store.Append(t.Context(), episode(scope, content, at))
	if err != nil {
		t.Fatalf("Append %q: %v", content, err)
	}
	if v != nil {
		if err := store.PutEncoding(t.Context(), encoding(got.Episodes[0].ID, v)); err != nil {
			t.Fatalf("PutEncoding %q: %v", content, err)
		}
	}
	return got.Episodes[0].ID
}

// unit builds a unit vector pointing along one axis, so cosine distances between
// the fixtures are obvious by construction.
func unit(axis int) mempher.Vector {
	v := make(mempher.Vector, testDimensions)
	v[axis%testDimensions] = 1
	return v
}

func semantic(scope mempher.ScopeID, v mempher.Vector, at time.Time) mempher.SemanticQuery {
	return mempher.SemanticQuery{
		ChannelQuery: mempher.ChannelQuery{Scope: scope, AsOf: at, Limit: 10},
		Vector:       v,
		Model:        testModel,
	}
}

func lexical(scope mempher.ScopeID, text string, at time.Time) mempher.LexicalQuery {
	return mempher.LexicalQuery{
		ChannelQuery: mempher.ChannelQuery{Scope: scope, AsOf: at, Limit: 10},
		Text:         text,
	}
}

func contents(candidates []mempher.Candidate) []string {
	out := make([]string, len(candidates))
	for i, c := range candidates {
		out[i] = c.Episode.Content
	}
	return out
}

func TestSearchSemantic(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	later := epoch.Add(time.Hour)

	seed(t, store, "user:1", "axis zero", epoch, unit(0))
	seed(t, store, "user:1", "axis one", epoch, unit(1))
	seed(t, store, "user:1", "axis two", epoch, unit(2))

	got, err := store.SearchSemantic(ctx, semantic("user:1", unit(1), later))
	if err != nil {
		t.Fatalf("SearchSemantic: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d candidates, want 3: %v", len(got), contents(got))
	}

	t.Run("the nearest vector ranks first", func(t *testing.T) {
		if got[0].Episode.Content != "axis one" {
			t.Errorf("best match is %q, want %q", got[0].Episode.Content, "axis one")
		}
	})

	t.Run("ranks are dense and start at one", func(t *testing.T) {
		for i, c := range got {
			if c.Rank != i+1 {
				t.Errorf("candidate %d has Rank %d, want %d", i, c.Rank, i+1)
			}
			if c.Channel != mempher.ChannelSemantic {
				t.Errorf("candidate %d reports channel %q", i, c.Channel)
			}
		}
	})

	t.Run("the score is cosine similarity, descending", func(t *testing.T) {
		// The query is a unit axis, so the identical vector scores 1 and the
		// orthogonal ones score 0.
		if diff := got[0].Score - 1; diff > 1e-6 || diff < -1e-6 {
			t.Errorf("best score = %v, want 1", got[0].Score)
		}
		for i := 1; i < len(got); i++ {
			if got[i].Score > got[i-1].Score {
				t.Errorf("scores are not descending: %v", got[i-1].Score)
			}
		}
		if got[1].Score > 1e-6 {
			t.Errorf("orthogonal score = %v, want 0", got[1].Score)
		}
	})

	t.Run("candidates carry the whole episode, so fusion needs no second trip",
		func(t *testing.T) {
			ep := got[0].Episode
			if ep.ID.IsZero() || ep.Seq == 0 || ep.Role != mempher.RoleUser ||
				ep.Source != "test" || ep.Binding["session"] != "s-42" {
				t.Errorf("candidate episode is incomplete: %+v", ep)
			}
		})
}

// TestSearchSemanticFindsSparseScopes is the regression test for the hazard that
// shapes this query.
//
// pgvector's HNSW scan yields at most hnsw.ef_search candidates before the scope
// filter is applied. With many other scopes crowding the query direction, a
// naive filtered search returns nothing at all while the scope's own episodes sit
// in the index unvisited. Iterative scans are what make this pass.
func TestSearchSemanticFindsSparseScopes(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	later := epoch.Add(time.Hour)

	// Crowd the query direction with other scopes: comfortably more than
	// ef_search, which defaults to 40.
	near := unit(0)
	for i := range 200 {
		seed(t, store, mempher.ScopeID(fmt.Sprintf("other:%d", i%20)),
			fmt.Sprintf("crowd %d", i), epoch, near)
	}
	// Our scope points elsewhere, so it is nowhere near the query.
	mine := []mempher.EpisodeID{
		seed(t, store, "user:1", "mine one", epoch, unit(1)),
		seed(t, store, "user:1", "mine two", epoch, unit(2)),
		seed(t, store, "user:1", "mine three", epoch, unit(3)),
	}

	got, err := store.SearchSemantic(ctx, semantic("user:1", near, later))
	if err != nil {
		t.Fatalf("SearchSemantic: %v", err)
	}
	if len(got) != len(mine) {
		t.Fatalf("got %d candidates, want all %d of the scope's episodes; "+
			"a filtered HNSW scan without iterative scans returns fewer",
			len(got), len(mine))
	}
	for _, c := range got {
		if c.Episode.Scope != "user:1" {
			t.Errorf("candidate leaked from scope %q", c.Episode.Scope)
		}
	}
}

func TestSearchSemanticRespectsFilters(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	later := epoch.Add(24 * time.Hour)

	seed(t, store, "user:1", "in scope", epoch, unit(0))
	seed(t, store, "user:2", "other scope", epoch, unit(0))

	// An episode ingested after the as-of cut.
	afterCut := epoch.Add(48 * time.Hour)
	cmd := episode("user:1", "from the future", afterCut)
	future, err := store.Append(ctx, cmd)
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := store.PutEncoding(ctx, encoding(future.Episodes[0].ID, unit(0))); err != nil {
		t.Fatalf("PutEncoding: %v", err)
	}

	t.Run("scope", func(t *testing.T) {
		got, err := store.SearchSemantic(ctx, semantic("user:1", unit(0), later))
		if err != nil {
			t.Fatalf("SearchSemantic: %v", err)
		}
		if len(got) != 1 || got[0].Episode.Content != "in scope" {
			t.Errorf("got %v, want only the in-scope episode", contents(got))
		}
	})

	t.Run("as-of hides what was not yet known", func(t *testing.T) {
		got, err := store.SearchSemantic(ctx, semantic("user:1", unit(0), afterCut))
		if err != nil {
			t.Fatalf("SearchSemantic: %v", err)
		}
		if len(got) != 2 {
			t.Errorf("got %v, want both episodes once the cut moves past them", contents(got))
		}
	})

	t.Run("an unencoded episode is invisible to this channel", func(t *testing.T) {
		seed(t, store, "user:1", "not encoded yet", epoch, nil)
		got, err := store.SearchSemantic(ctx, semantic("user:1", unit(0), later))
		if err != nil {
			t.Fatalf("SearchSemantic: %v", err)
		}
		for _, c := range got {
			if c.Episode.Content == "not encoded yet" {
				t.Error("an episode with no encoding was returned")
			}
		}
	})

	t.Run("another model's encodings are not searched", func(t *testing.T) {
		q := semantic("user:1", unit(0), later)
		q.Model = "a-different-model@8"
		got, err := store.SearchSemantic(ctx, q)
		if err != nil {
			t.Fatalf("SearchSemantic: %v", err)
		}
		if len(got) != 0 {
			t.Errorf("got %v, want nothing for an unused model", contents(got))
		}
	})
}

func TestSearchLexical(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	later := epoch.Add(time.Hour)

	seed(t, store, "user:1", "I am allergic to hazelnuts and hazelnut syrup", epoch, nil)
	seed(t, store, "user:1", "A passing mention of hazelnuts", epoch, nil)
	seed(t, store, "user:1", "I drink dark roast coffee with no sugar", epoch, nil)
	seed(t, store, "user:2", "hazelnuts in another scope", epoch, nil)

	t.Run("finds matches with no worker having run", func(t *testing.T) {
		got, err := store.SearchLexical(ctx, lexical("user:1", "hazelnut", later))
		if err != nil {
			t.Fatalf("SearchLexical: %v", err)
		}
		if len(got) != 2 {
			t.Fatalf("got %v, want the two hazelnut episodes", contents(got))
		}
		// Cover density ranks the denser mention first.
		if got[0].Episode.Content != "I am allergic to hazelnuts and hazelnut syrup" {
			t.Errorf("best match is %q", got[0].Episode.Content)
		}
		for i, c := range got {
			if c.Rank != i+1 || c.Channel != mempher.ChannelLexical {
				t.Errorf("candidate %d = rank %d channel %q", i, c.Rank, c.Channel)
			}
			if c.Score <= 0 {
				t.Errorf("candidate %d has score %v, want positive", i, c.Score)
			}
		}
	})

	t.Run("stemming works, so an inflected query finds the root", func(t *testing.T) {
		// "roasted" appears nowhere in the content; it stems to "roast",
		// which does. ("allergies" would not match "allergic": they stem to
		// allergi and allerg respectively.)
		got, err := store.SearchLexical(ctx, lexical("user:1", "roasted", later))
		if err != nil {
			t.Fatalf("SearchLexical: %v", err)
		}
		if len(got) != 1 || got[0].Episode.Content != "I drink dark roast coffee with no sugar" {
			t.Errorf("got %v, want the coffee episode via stemming", contents(got))
		}
	})

	t.Run("a query of only stop words is empty, not an error", func(t *testing.T) {
		got, err := store.SearchLexical(ctx, lexical("user:1", "the and of", later))
		if err != nil {
			t.Fatalf("SearchLexical: %v", err)
		}
		if len(got) != 0 {
			t.Errorf("got %v, want nothing", contents(got))
		}
	})

	t.Run("punctuation and operators cannot be a syntax error", func(t *testing.T) {
		for _, text := range []string{`hazelnut & | ! ( )`, `"unclosed quote`, `<->`, `a:*`} {
			if _, err := store.SearchLexical(ctx, lexical("user:1", text, later)); err != nil {
				t.Errorf("SearchLexical(%q): %v", text, err)
			}
		}
	})

	t.Run("scope isolation", func(t *testing.T) {
		got, err := store.SearchLexical(ctx, lexical("user:2", "hazelnut", later))
		if err != nil {
			t.Fatalf("SearchLexical: %v", err)
		}
		if len(got) != 1 || got[0].Episode.Scope != "user:2" {
			t.Errorf("got %v, want only scope user:2", contents(got))
		}
	})
}

func TestSearchSharedFilters(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	later := epoch.Add(72 * time.Hour)

	// Two episodes differing in event time, role and binding.
	early := episode("user:1", "hazelnut early", epoch)
	early.Episodes[0].OccurredAt = epoch
	early.Episodes[0].Role = mempher.RoleUser
	early.Episodes[0].Binding = mempher.Binding{"session": "a"}
	first, err := store.Append(ctx, early)
	if err != nil {
		t.Fatalf("Append: %v", err)
	}

	late := episode("user:1", "hazelnut late", epoch)
	late.Episodes[0].OccurredAt = epoch.Add(48 * time.Hour)
	late.Episodes[0].Role = mempher.RoleAssistant
	late.Episodes[0].Binding = mempher.Binding{"session": "b"}
	second, err := store.Append(ctx, late)
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	for _, id := range []mempher.EpisodeID{first.Episodes[0].ID, second.Episodes[0].ID} {
		if err := store.PutEncoding(ctx, encoding(id, unit(0))); err != nil {
			t.Fatalf("PutEncoding: %v", err)
		}
	}

	tests := []struct {
		name   string
		mutate func(*mempher.ChannelQuery)
		want   string
	}{
		{
			name:   "event-time lower bound",
			mutate: func(q *mempher.ChannelQuery) { q.OccurredFrom = epoch.Add(24 * time.Hour) },
			want:   "hazelnut late",
		},
		{
			name:   "event-time upper bound",
			mutate: func(q *mempher.ChannelQuery) { q.OccurredTo = epoch.Add(24 * time.Hour) },
			want:   "hazelnut early",
		},
		{
			name:   "role",
			mutate: func(q *mempher.ChannelQuery) { q.Roles = []mempher.Role{mempher.RoleAssistant} },
			want:   "hazelnut late",
		},
		{
			name:   "binding containment",
			mutate: func(q *mempher.ChannelQuery) { q.Binding = mempher.Binding{"session": "a"} },
			want:   "hazelnut early",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// The same filter must behave identically on both channels.
			sem := semantic("user:1", unit(0), later)
			tc.mutate(&sem.ChannelQuery)
			gotSem, err := store.SearchSemantic(t.Context(), sem)
			if err != nil {
				t.Fatalf("SearchSemantic: %v", err)
			}
			if len(gotSem) != 1 || gotSem[0].Episode.Content != tc.want {
				t.Errorf("semantic got %v, want [%s]", contents(gotSem), tc.want)
			}

			lex := lexical("user:1", "hazelnut", later)
			tc.mutate(&lex.ChannelQuery)
			gotLex, err := store.SearchLexical(t.Context(), lex)
			if err != nil {
				t.Fatalf("SearchLexical: %v", err)
			}
			if len(gotLex) != 1 || gotLex[0].Episode.Content != tc.want {
				t.Errorf("lexical got %v, want [%s]", contents(gotLex), tc.want)
			}
		})
	}
}

func TestSearchRejectsBadQueries(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	t.Run("semantic", func(t *testing.T) {
		tests := []struct {
			name    string
			q       mempher.SemanticQuery
			wantErr error
		}{
			{
				name:    "no scope",
				q:       semantic("", unit(0), epoch),
				wantErr: mempher.ErrInvalidScope,
			},
			{
				name:    "unresolved as-of",
				q:       semantic("user:1", unit(0), time.Time{}),
				wantErr: mempher.ErrInvalidConfig,
			},
			{
				name: "no model",
				q: func() mempher.SemanticQuery {
					q := semantic("user:1", unit(0), epoch)
					q.Model = ""
					return q
				}(),
				wantErr: mempher.ErrInvalidConfig,
			},
			{
				name:    "wrong vector width",
				q:       semantic("user:1", unit(0)[:2], epoch),
				wantErr: mempher.ErrDimensionMismatch,
			},
			{
				name: "inverted event-time window",
				q: func() mempher.SemanticQuery {
					q := semantic("user:1", unit(0), epoch)
					q.OccurredFrom, q.OccurredTo = epoch.Add(time.Hour), epoch
					return q
				}(),
				wantErr: mempher.ErrInvalidConfig,
			},
			{
				name: "unknown role filter",
				q: func() mempher.SemanticQuery {
					q := semantic("user:1", unit(0), epoch)
					q.Roles = []mempher.Role{mempher.Role("wizard")}
					return q
				}(),
				wantErr: mempher.ErrInvalidRole,
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				if _, err := store.SearchSemantic(ctx, tc.q); !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
			})
		}
	})

	t.Run("lexical", func(t *testing.T) {
		if _, err := store.SearchLexical(ctx, lexical("", "x", epoch)); !errors.Is(err, mempher.ErrInvalidScope) {
			t.Errorf("no scope err = %v", err)
		}
		if _, err := store.SearchLexical(ctx, lexical("user:1", "  ", epoch)); !errors.Is(err, mempher.ErrInvalidContent) {
			t.Errorf("blank text err = %v", err)
		}
		if _, err := store.SearchLexical(ctx, lexical("user:1", "x", time.Time{})); !errors.Is(err, mempher.ErrInvalidConfig) {
			t.Errorf("unresolved as-of err = %v", err)
		}
	})
}

// TestSearchLimit checks a channel returns no more than it was asked for.
func TestSearchLimit(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	later := epoch.Add(time.Hour)

	for i := range 12 {
		seed(t, store, "user:1", fmt.Sprintf("hazelnut %d", i), epoch, unit(i))
	}

	sem := semantic("user:1", unit(0), later)
	sem.Limit = 5
	got, err := store.SearchSemantic(ctx, sem)
	if err != nil {
		t.Fatalf("SearchSemantic: %v", err)
	}
	if len(got) != 5 {
		t.Errorf("semantic returned %d, want 5", len(got))
	}

	lex := lexical("user:1", "hazelnut", later)
	lex.Limit = 4
	gotLex, err := store.SearchLexical(ctx, lex)
	if err != nil {
		t.Fatalf("SearchLexical: %v", err)
	}
	if len(gotLex) != 4 {
		t.Errorf("lexical returned %d, want 4", len(gotLex))
	}
}

// TestSearchOrderIsStableUnderTies pins the determinism Recall promises.
//
// Episodes that mention a term equally often tie exactly, and a vector index has
// no reason to return tied rows in a consistent order between scans. Without a
// tie-break the same request would give different answers on different runs.
func TestSearchOrderIsStableUnderTies(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	later := epoch.Add(time.Hour)

	// Identical vectors and identical text: every ordering signal is a tie.
	tied := unit(0)
	for i := range 6 {
		seed(t, store, "user:1", fmt.Sprintf("hazelnut %d", i), epoch, tied)
	}

	var first []string
	for attempt := range 5 {
		sem, err := store.SearchSemantic(ctx, semantic("user:1", unit(0), later))
		if err != nil {
			t.Fatalf("SearchSemantic: %v", err)
		}
		lex, err := store.SearchLexical(ctx, lexical("user:1", "hazelnut", later))
		if err != nil {
			t.Fatalf("SearchLexical: %v", err)
		}
		order := append(contents(sem), contents(lex)...)

		if attempt == 0 {
			first = order
			if len(sem) != 6 || len(lex) != 6 {
				t.Fatalf("got %d semantic and %d lexical candidates, want 6 each",
					len(sem), len(lex))
			}
			continue
		}
		if fmt.Sprint(order) != fmt.Sprint(first) {
			t.Fatalf("attempt %d ordered tied episodes as %v, want %v", attempt, order, first)
		}
	}

	// Ranks must still be dense and start at one after the re-sort.
	sem, err := store.SearchSemantic(ctx, semantic("user:1", unit(0), later))
	if err != nil {
		t.Fatalf("SearchSemantic: %v", err)
	}
	for i, c := range sem {
		if c.Rank != i+1 {
			t.Errorf("candidate %d has Rank %d, want %d", i, c.Rank, i+1)
		}
	}
}

// TestSearchLexicalMatchesAnyTerm pins the property that decides whether the
// lexical channel answers a real question at all.
//
// It shipped the other way: websearch_to_tsquery puts AND between terms, so an
// episode had to carry every word of the query. Against LongMemEval that left
// 83% of questions matching nothing, and every existing test still passed --
// because every one of them queried a single word. This is the test whose
// absence let that through.
func TestSearchLexicalMatchesAnyTerm(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	later := epoch.Add(time.Hour)

	seed(t, store, "user:1", "I moved to Berlin in June for a new job", epoch, nil)
	seed(t, store, "user:1", "Berlin is cold in winter", epoch, nil)
	seed(t, store, "user:1", "My new job starts on Monday", epoch, nil)
	seed(t, store, "user:1", "I had pasta for dinner", epoch, nil)

	// The shape of a real question: no single episode holds all of it.
	const question = "why did I move to Berlin for a new job in June"

	t.Run("a conversational query matches partial overlaps", func(t *testing.T) {
		got, err := store.SearchLexical(ctx, lexical("user:1", question, later))
		if err != nil {
			t.Fatalf("SearchLexical: %v", err)
		}
		if len(got) < 3 {
			t.Fatalf("got %v, want every episode sharing any term", contents(got))
		}
	})

	t.Run("more overlap still ranks higher", func(t *testing.T) {
		// This is why dropping AND costs nothing: ts_rank_cd already expresses
		// it. The episode carrying the most of the query comes back first, so
		// what AND would have selected is what OR puts at the top.
		got, err := store.SearchLexical(ctx, lexical("user:1", question, later))
		if err != nil {
			t.Fatalf("SearchLexical: %v", err)
		}
		if got[0].Episode.Content != "I moved to Berlin in June for a new job" {
			t.Errorf("best match is %q, want the episode sharing the most terms",
				got[0].Episode.Content)
		}
		for i := 1; i < len(got); i++ {
			if got[i].Score > got[i-1].Score {
				t.Errorf("candidate %d scores above the one before it", i)
			}
		}
	})

	t.Run("an unrelated episode is still excluded", func(t *testing.T) {
		// Permissive is not indiscriminate: sharing no lexeme still means no
		// match, or the channel would return the whole scope every time.
		got, err := store.SearchLexical(ctx, lexical("user:1", "Berlin winter", later))
		if err != nil {
			t.Fatalf("SearchLexical: %v", err)
		}
		for _, candidate := range got {
			if candidate.Episode.Content == "I had pasta for dinner" {
				t.Errorf("got %v, want the pasta episode excluded", contents(got))
			}
		}
	})

	t.Run("a query of nothing but stop words matches nothing", func(t *testing.T) {
		got, err := store.SearchLexical(ctx, lexical("user:1", "the and of", later))
		if err != nil {
			t.Fatalf("SearchLexical: %v", err)
		}
		if len(got) != 0 {
			t.Errorf("got %v, want no matches from an empty tsquery", contents(got))
		}
	})
}
