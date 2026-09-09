package memphertest_test

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/memphertest"
)

func cosine(t *testing.T, a, b mempher.Vector) float64 {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("dimension mismatch: %d vs %d", len(a), len(b))
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func TestEmbedderIsDeterministic(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	e := memphertest.NewEmbedder(64)

	first, err := e.EmbedQuery(ctx, "I am allergic to hazelnuts")
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	second, err := e.EmbedQuery(ctx, "I am allergic to hazelnuts")
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if got := cosine(t, first, second); math.Abs(got-1) > 1e-9 {
		t.Errorf("identical text embedded to cosine %v, want 1", got)
	}

	// A separate embedder of the same width must agree, or a test that
	// rebuilds its fixtures would drift.
	other := memphertest.NewEmbedder(64)
	again, err := other.EmbedQuery(ctx, "I am allergic to hazelnuts")
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if got := cosine(t, first, again); math.Abs(got-1) > 1e-9 {
		t.Errorf("a fresh embedder disagreed: cosine %v, want 1", got)
	}
}

// TestEmbedderSimilarityTracksOverlap is the property that makes this fake worth
// having: a semantic-channel test can assert ranking, not just identity.
func TestEmbedderSimilarityTracksOverlap(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	e := memphertest.NewEmbedder(256)

	embed := func(s string) mempher.Vector {
		v, err := e.EmbedDocuments(ctx, []string{s})
		if err != nil {
			t.Fatalf("EmbedDocuments(%q): %v", s, err)
		}
		return v[0]
	}

	query := embed("hazelnut allergy")
	related := embed("I am allergic to hazelnuts, so no hazelnut syrup")
	unrelated := embed("I drink dark roast coffee with no sugar")

	near, far := cosine(t, query, related), cosine(t, query, unrelated)
	if near <= far {
		t.Errorf("related cosine %v should exceed unrelated %v", near, far)
	}
	if near <= 0 {
		t.Errorf("related cosine %v should be positive", near)
	}
}

func TestEmbedderNormalisationAndEdgeCases(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	e := memphertest.NewEmbedder(32)

	tests := []struct {
		name     string
		text     string
		wantZero bool
	}{
		{name: "prose", text: "the quick brown fox"},
		{name: "punctuation only", text: "!!! ??? ...", wantZero: true},
		{name: "empty", text: "", wantZero: true},
		{name: "single token", text: "hazelnut"},
		{name: "digits count as tokens", text: "user 8123"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			v, err := e.EmbedQuery(ctx, tc.text)
			if err != nil {
				t.Fatalf("EmbedQuery: %v", err)
			}
			if v.Dimensions() != 32 {
				t.Fatalf("dimensions = %d, want 32", v.Dimensions())
			}
			var norm float64
			for _, f := range v {
				norm += float64(f) * float64(f)
			}
			norm = math.Sqrt(norm)
			if tc.wantZero {
				if norm != 0 {
					t.Errorf("norm = %v, want 0 for token-free text", norm)
				}
				return
			}
			if math.Abs(norm-1) > 1e-6 {
				t.Errorf("norm = %v, want 1", norm)
			}
		})
	}
}

// TestEmbedderIgnoresCaseAndPunctuation keeps fixtures from being brittle about
// how a sentence was typed.
func TestEmbedderIgnoresCaseAndPunctuation(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	e := memphertest.NewEmbedder(64)

	a, err := e.EmbedQuery(ctx, "Dark Roast, no sugar!")
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	b, err := e.EmbedQuery(ctx, "dark   roast no sugar")
	if err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if got := cosine(t, a, b); math.Abs(got-1) > 1e-9 {
		t.Errorf("cosine = %v, want 1", got)
	}
}

func TestEmbedderFailAndCalls(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	e := memphertest.NewEmbedder(16)

	if got := e.Calls(); got != 0 {
		t.Fatalf("Calls() = %d before use, want 0", got)
	}
	if _, err := e.EmbedQuery(ctx, "hello"); err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if got := e.Calls(); got != 1 {
		t.Errorf("Calls() = %d, want 1", got)
	}

	sentinel := errors.New("provider is down")
	e.Fail(sentinel)
	if _, err := e.EmbedQuery(ctx, "hello"); !errors.Is(err, sentinel) {
		t.Errorf("EmbedQuery err = %v, want %v", err, sentinel)
	}
	if _, err := e.EmbedDocuments(ctx, []string{"hello"}); !errors.Is(err, sentinel) {
		t.Errorf("EmbedDocuments err = %v, want %v", err, sentinel)
	}
	if got := e.Calls(); got != 3 {
		t.Errorf("Calls() = %d, want 3 (failures count)", got)
	}

	e.Fail(nil)
	if _, err := e.EmbedQuery(ctx, "hello"); err != nil {
		t.Errorf("EmbedQuery after clearing the failure: %v", err)
	}
}

func TestEmbedderHonoursCancellation(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	e := memphertest.NewEmbedder(16)
	if _, err := e.EmbedQuery(ctx, "hello"); !errors.Is(err, context.Canceled) {
		t.Errorf("EmbedQuery err = %v, want context.Canceled", err)
	}
	if _, err := e.EmbedDocuments(ctx, []string{"hello"}); !errors.Is(err, context.Canceled) {
		t.Errorf("EmbedDocuments err = %v, want context.Canceled", err)
	}
}

func TestEmbedderDocumentsPreserveOrder(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	e := memphertest.NewEmbedder(64)
	texts := []string{"alpha", "beta", "gamma"}

	batch, err := e.EmbedDocuments(ctx, texts)
	if err != nil {
		t.Fatalf("EmbedDocuments: %v", err)
	}
	if len(batch) != len(texts) {
		t.Fatalf("got %d vectors, want %d", len(batch), len(texts))
	}
	for i, text := range texts {
		one, err := e.EmbedQuery(ctx, text)
		if err != nil {
			t.Fatalf("EmbedQuery(%q): %v", text, err)
		}
		if got := cosine(t, batch[i], one); math.Abs(got-1) > 1e-9 {
			t.Errorf("vector %d does not match %q: cosine %v", i, text, got)
		}
	}
}

func TestEmbedderModelIdentifiesWidth(t *testing.T) {
	t.Parallel()

	small, large := memphertest.NewEmbedder(8), memphertest.NewEmbedder(16)
	if small.Model() == large.Model() {
		t.Errorf("embedders of different widths share model id %q", small.Model())
	}
	if small.Dimensions() != 8 || large.Dimensions() != 16 {
		t.Errorf("dimensions = %d, %d; want 8, 16", small.Dimensions(), large.Dimensions())
	}
}

func TestNewEmbedderRejectsBadWidth(t *testing.T) {
	t.Parallel()

	for _, dims := range []int{0, -1} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("NewEmbedder(%d) should panic", dims)
				}
			}()
			memphertest.NewEmbedder(dims)
		}()
	}
}

func TestClock(t *testing.T) {
	t.Parallel()

	start := time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)
	c := memphertest.NewClock(start)

	if got := c.Now(); !got.Equal(start) {
		t.Errorf("Now() = %s, want %s", got, start)
	}
	if got := c.Now(); !got.Equal(start) {
		t.Errorf("Now() moved on its own to %s", got)
	}

	want := start.Add(90 * time.Minute)
	if got := c.Advance(90 * time.Minute); !got.Equal(want) {
		t.Errorf("Advance returned %s, want %s", got, want)
	}
	if got := c.Now(); !got.Equal(want) {
		t.Errorf("Now() = %s after Advance, want %s", got, want)
	}

	if got := c.Advance(-time.Hour); !got.Equal(start.Add(30 * time.Minute)) {
		t.Errorf("Advance backwards returned %s", got)
	}

	c.Set(start)
	if got := c.Now(); !got.Equal(start) {
		t.Errorf("Now() = %s after Set, want %s", got, start)
	}
}

// TestClockIsConcurrencySafe matters because a worker under test advances the
// clock from one goroutine while reading it on another; run with -race.
func TestClockIsConcurrencySafe(t *testing.T) {
	t.Parallel()

	c := memphertest.NewClock(time.Unix(0, 0).UTC())
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 1000 {
			c.Advance(time.Millisecond)
		}
	}()
	for range 1000 {
		_ = c.Now()
	}
	<-done

	if got := c.Now(); !got.Equal(time.Unix(0, 0).UTC().Add(time.Second)) {
		t.Errorf("Now() = %s, want one second past the epoch", got)
	}
}
