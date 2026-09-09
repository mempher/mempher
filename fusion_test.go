package mempher

import (
	"errors"
	"math"
	"testing"
)

// ep builds an episode whose id encodes n, so tests can name episodes by number
// and still get the id ordering that uuidv7 gives in production: a larger n is a
// "newer" episode.
func ep(n byte) Episode {
	var id EpisodeID
	id[15] = n
	return Episode{ID: id, Content: string(rune('a' + n))}
}

// ids renders the fused order as the numbers ep produced, so a failure message
// reads as an order rather than as sixteen bytes of UUID.
func ids(fused []RecalledEpisode) []byte {
	out := make([]byte, len(fused))
	for i, r := range fused {
		out[i] = r.Episode.ID[15]
	}
	return out
}

func rrf(k float64, ranks ...int) float64 {
	var sum float64
	for _, r := range ranks {
		sum += 1 / (k + float64(r))
	}
	return sum
}

func TestFuse(t *testing.T) {
	t.Parallel()

	boom := errors.New("channel exploded")

	tests := []struct {
		name    string
		results []channelResult
		opts    FusionOptions
		want    []byte    // fused order, by episode number
		scores  []float64 // expected score per position; nil to skip
		hits    map[byte]int
	}{
		{
			name:    "no channels yields nothing",
			results: nil,
			want:    []byte{},
		},
		{
			name: "single channel preserves its own order",
			results: []channelResult{{
				channel: ChannelLexical,
				candidates: []Candidate{
					{Episode: ep(1), Channel: ChannelLexical, Rank: 1, Score: 0.9},
					{Episode: ep(2), Channel: ChannelLexical, Rank: 2, Score: 0.5},
					{Episode: ep(3), Channel: ChannelLexical, Rank: 3, Score: 0.1},
				},
			}},
			want:   []byte{1, 2, 3},
			scores: []float64{rrf(60, 1), rrf(60, 2), rrf(60, 3)},
			hits:   map[byte]int{1: 1, 2: 1, 3: 1},
		},
		{
			name: "agreement beats a single top hit",
			// Episode 2 is second in both channels; episode 1 is first in
			// one and absent from the other. Fusing must prefer the
			// episode both channels found -- this is the whole point.
			results: []channelResult{
				{channel: ChannelSemantic, candidates: []Candidate{
					{Episode: ep(1), Channel: ChannelSemantic, Rank: 1},
					{Episode: ep(2), Channel: ChannelSemantic, Rank: 2},
				}},
				{channel: ChannelLexical, candidates: []Candidate{
					{Episode: ep(3), Channel: ChannelLexical, Rank: 1},
					{Episode: ep(2), Channel: ChannelLexical, Rank: 2},
				}},
			},
			want:   []byte{2, 3, 1},
			scores: []float64{rrf(60, 2, 2), rrf(60, 1), rrf(60, 1)},
			hits:   map[byte]int{2: 2, 3: 1, 1: 1},
		},
		{
			name: "a failed channel contributes nothing, partial results included",
			results: []channelResult{
				{channel: ChannelSemantic, err: boom, candidates: []Candidate{
					{Episode: ep(9), Channel: ChannelSemantic, Rank: 1},
				}},
				{channel: ChannelLexical, candidates: []Candidate{
					{Episode: ep(1), Channel: ChannelLexical, Rank: 1},
				}},
			},
			want: []byte{1},
			hits: map[byte]int{1: 1},
		},
		{
			name: "a repeat within one channel counts once at its best rank",
			results: []channelResult{{
				channel: ChannelLexical,
				candidates: []Candidate{
					{Episode: ep(1), Channel: ChannelLexical, Rank: 1},
					{Episode: ep(1), Channel: ChannelLexical, Rank: 2},
					{Episode: ep(2), Channel: ChannelLexical, Rank: 3},
				},
			}},
			want:   []byte{1, 2},
			scores: []float64{rrf(60, 1), rrf(60, 3)},
			hits:   map[byte]int{1: 1, 2: 1},
		},
		{
			name: "ties break towards the newer episode",
			// Same rank in the same channel means identical scores, so the
			// uuidv7 ordering decides: higher id first.
			results: []channelResult{
				{channel: ChannelSemantic, candidates: []Candidate{
					{Episode: ep(1), Channel: ChannelSemantic, Rank: 1},
				}},
				{channel: ChannelLexical, candidates: []Candidate{
					{Episode: ep(7), Channel: ChannelLexical, Rank: 1},
				}},
			},
			want: []byte{7, 1},
		},
		{
			name: "weights reorder without changing what was found",
			results: []channelResult{
				{channel: ChannelSemantic, candidates: []Candidate{
					{Episode: ep(1), Channel: ChannelSemantic, Rank: 1},
				}},
				{channel: ChannelLexical, candidates: []Candidate{
					{Episode: ep(2), Channel: ChannelLexical, Rank: 5},
				}},
			},
			opts: FusionOptions{K: 1, Weights: map[Channel]float64{ChannelLexical: 100}},
			want: []byte{2, 1},
			hits: map[byte]int{1: 1, 2: 1},
		},
		{
			name: "a zero weight scores nothing but is still reported",
			results: []channelResult{
				{channel: ChannelSemantic, candidates: []Candidate{
					{Episode: ep(1), Channel: ChannelSemantic, Rank: 1},
				}},
				{channel: ChannelLexical, candidates: []Candidate{
					{Episode: ep(2), Channel: ChannelLexical, Rank: 1},
				}},
			},
			opts:   FusionOptions{Weights: map[Channel]float64{ChannelLexical: 0}},
			want:   []byte{1, 2},
			scores: []float64{rrf(60, 1), 0},
			hits:   map[byte]int{1: 1, 2: 1},
		},
		{
			name: "a smaller K sharpens the advantage of rank 1",
			results: []channelResult{{
				channel: ChannelLexical,
				candidates: []Candidate{
					{Episode: ep(1), Channel: ChannelLexical, Rank: 1},
					{Episode: ep(2), Channel: ChannelLexical, Rank: 2},
				},
			}},
			opts:   FusionOptions{K: 1},
			want:   []byte{1, 2},
			scores: []float64{rrf(1, 1), rrf(1, 2)},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			opts, err := tc.opts.resolve()
			if err != nil {
				t.Fatalf("resolve options: %v", err)
			}
			got := fuse(tc.results, opts)

			if gotIDs := ids(got); string(gotIDs) != string(tc.want) {
				t.Errorf("fused order = %v, want %v", gotIDs, tc.want)
			}
			for i, want := range tc.scores {
				if i >= len(got) {
					break
				}
				if math.Abs(got[i].Score-want) > 1e-12 {
					t.Errorf("score[%d] = %v, want %v", i, got[i].Score, want)
				}
			}
			for _, r := range got {
				if want, ok := tc.hits[r.Episode.ID[15]]; ok && len(r.Hits) != want {
					t.Errorf("episode %d has %d hits, want %d",
						r.Episode.ID[15], len(r.Hits), want)
				}
			}
		})
	}
}

// TestFuseHitsAreSorted pins the provenance order, so a caller can diff two
// recollections without sorting them first.
func TestFuseHitsAreSorted(t *testing.T) {
	t.Parallel()

	opts, err := FusionOptions{}.resolve()
	if err != nil {
		t.Fatalf("resolve options: %v", err)
	}
	// Deliberately supplied lexical-first, which is not alphabetical order.
	fused := fuse([]channelResult{
		{channel: ChannelLexical, candidates: []Candidate{
			{Episode: ep(1), Channel: ChannelLexical, Rank: 2},
		}},
		{channel: ChannelSemantic, candidates: []Candidate{
			{Episode: ep(1), Channel: ChannelSemantic, Rank: 1},
		}},
	}, opts)

	if len(fused) != 1 {
		t.Fatalf("got %d episodes, want 1", len(fused))
	}
	got := fused[0].Hits
	if len(got) != 2 {
		t.Fatalf("got %d hits, want 2", len(got))
	}
	if got[0].Channel != ChannelLexical || got[1].Channel != ChannelSemantic {
		t.Errorf("hits = [%s %s], want [%s %s]",
			got[0].Channel, got[1].Channel, ChannelLexical, ChannelSemantic)
	}
	if got[0].Rank != 2 || got[1].Rank != 1 {
		t.Errorf("ranks = [%d %d], want [2 1]", got[0].Rank, got[1].Rank)
	}
}

func TestFusionOptionsResolve(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		opts    FusionOptions
		wantK   float64
		wantErr error
	}{
		{name: "zero K takes the default", opts: FusionOptions{}, wantK: DefaultFusionK},
		{name: "explicit K is kept", opts: FusionOptions{K: 1}, wantK: 1},
		{name: "negative K is rejected", opts: FusionOptions{K: -1}, wantErr: ErrInvalidConfig},
		{
			name:    "negative weight is rejected",
			opts:    FusionOptions{Weights: map[Channel]float64{ChannelLexical: -0.5}},
			wantErr: ErrInvalidConfig,
		},
		{
			name:    "unknown channel is rejected",
			opts:    FusionOptions{Weights: map[Channel]float64{Channel("telepathy"): 1}},
			wantErr: ErrInvalidChannel,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := tc.opts.resolve()
			if tc.wantErr != nil {
				if !errors.Is(err, tc.wantErr) {
					t.Fatalf("err = %v, want %v", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.K != tc.wantK {
				t.Errorf("K = %v, want %v", got.K, tc.wantK)
			}
		})
	}
}

// TestFusionOptionsResolveDetachesWeights guards against a caller mutating a
// live Memory by holding on to the map it passed to New.
func TestFusionOptionsResolveDetachesWeights(t *testing.T) {
	t.Parallel()

	weights := map[Channel]float64{ChannelLexical: 2}
	resolved, err := FusionOptions{Weights: weights}.resolve()
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	weights[ChannelLexical] = 999

	if got := resolved.weight(ChannelLexical); got != 2 {
		t.Errorf("weight = %v after mutating the caller's map, want 2", got)
	}
	if got := resolved.weight(ChannelSemantic); got != 1 {
		t.Errorf("unweighted channel = %v, want 1", got)
	}
}
