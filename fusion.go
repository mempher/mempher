// Reciprocal rank fusion: how independent channel rankings combine.

package mempher

import (
	"bytes"
	"cmp"
	"fmt"
	"maps"
	"slices"
	"time"
)

// DefaultFusionK is the reciprocal rank fusion smoothing constant: the k in
// 1/(k+rank) from Cormack, Clarke and Buettcher. Larger values flatten the
// advantage of the top ranks.
//
// 60 is the value from the paper, and it survived being measured against
// alternatives rather than being kept because it was published. Swept against
// [DefaultLexicalWeight] over 479 LongMemEval questions, hit@10:
//
//	 K    w=0.1   w=0.25    w=0.5      w=1
//	 1    0.846    0.837    0.820    0.787
//	 5    0.850    0.843    0.837    0.800
//	20    0.852    0.854    0.843    0.810
//	60    0.854    0.871    0.843    0.800
//
// Two things come out of that. K is the second-order knob: moving it spans at
// most 0.034 at a fixed weight, where moving the weight spans up to 0.071 at a
// fixed K. And a smaller K is worse at every weight, not better.
//
// Which is the opposite of what the shape of the arithmetic suggests, so it is
// worth saying why. A small K raises what a rank-1 is worth in absolute terms
// for every channel at once: at K=1 a lexical rank-1 scores 1/2 while a semantic
// rank-2 scores 1/3, so one noisy top hit leapfrogs almost the whole of the
// better channel's ranking. At K=60 the same lexical rank-1 scores 1/61 against
// a semantic rank-1's 1/61 -- near enough equal that what separates episodes is
// being found by both channels, which is the more robust signal. Flattening the
// ranks is what makes fusion about agreement instead of about whichever channel
// shouted loudest.
//
// So this is calibrated for channels that disagree in quality. Re-measure it
// with the eval subpackage before changing it.
const DefaultFusionK = 60.0

// DefaultLexicalWeight is how much a lexical rank counts against a semantic one
// when [FusionOptions.Weights] does not say.
//
// It is not 1, and the reason is that reciprocal rank fusion assumes its inputs
// are comparably reliable. These two are not. The lexical channel matches an
// episode sharing any one lexeme with the query, so it always returns a full
// candidate list whose tail is noise; the semantic channel's ordering is
// meaningful the whole way down. At equal weight a lexical rank-1 contributes
// exactly what a semantic rank-1 does, and on a conversational query that is
// usually the wrong episode.
//
// Measured on LongMemEval over 479 questions, fusing at equal weight scored
// below the semantic channel used alone -- hit@10 0.800 against 0.850, and MRR
// 0.317 against 0.495. Quartering the lexical contribution reverses it:
//
//	lexical alone              hit@10 0.384   MRR 0.123
//	semantic alone             hit@10 0.850   MRR 0.495
//	fused, lexical weight 1     hit@10 0.800   MRR 0.317
//	fused, lexical weight 0.25  hit@10 0.871   MRR 0.425
//
// So fusion is worth having, and it is worth having asymmetrically. Set
// Weights explicitly to override this; a corpus of keyword-like queries rather
// than conversational ones would want it higher.
const DefaultLexicalWeight = 0.25

// FusionOptions tunes how channel rankings combine.
//
// Fusion works on ranks, not scores: a cosine similarity and a ts_rank live on
// incomparable scales, and normalising them would invent a relationship that is
// not there. Each episode scores the sum, over the channels that found it, of
// weight/(K+rank), so an episode found halfway down two channels can outrank one
// at the top of a single channel.
type FusionOptions struct {
	// K is the smoothing constant. Zero means [DefaultFusionK]. Must not be
	// negative.
	K float64
	// Weights scales individual channels. A weight of 0 excludes a channel
	// from the score while still reporting what it found. Must not be
	// negative.
	//
	// A channel with no entry weighs 1, except [ChannelLexical], which weighs
	// [DefaultLexicalWeight] for the reason given there. Naming it in this map
	// -- at 1, or at anything else -- overrides that.
	Weights map[Channel]float64
}

// resolve returns a validated copy with defaults applied and the weight map
// detached, so editing the map passed to [New] cannot reweight a live [Memory].
func (o FusionOptions) resolve() (FusionOptions, error) {
	out := FusionOptions{K: o.K}
	switch {
	case out.K < 0:
		return FusionOptions{}, fmt.Errorf("fusion K is %v: %w", out.K, ErrInvalidConfig)
	case out.K == 0:
		out.K = DefaultFusionK
	}
	for _, ch := range slices.Sorted(maps.Keys(o.Weights)) {
		w := o.Weights[ch]
		if !ch.Valid() {
			return FusionOptions{}, fmt.Errorf("fusion weight for %q: %w", ch, ErrInvalidChannel)
		}
		if w < 0 {
			return FusionOptions{}, fmt.Errorf("fusion weight for %q is %v: %w", ch, w, ErrInvalidConfig)
		}
	}
	out.Weights = maps.Clone(o.Weights)
	if out.Weights == nil {
		out.Weights = make(map[Channel]float64, 1)
	}
	// Applied here rather than in weight() so that a resolved FusionOptions
	// states every weight in force, and a caller reading one back is not left
	// to know which channels have a default of their own.
	if _, named := out.Weights[ChannelLexical]; !named {
		out.Weights[ChannelLexical] = DefaultLexicalWeight
	}
	return out, nil
}

// weight returns the multiplier for one channel.
func (o FusionOptions) weight(c Channel) float64 {
	if w, ok := o.Weights[c]; ok {
		return w
	}
	return 1
}

// channelResult is what one channel contributed. Recall fills one per requested
// channel in a fixed slot, so concurrent completion cannot make the fused order
// depend on timing.
type channelResult struct {
	channel    Channel
	candidates []Candidate
	// facts is what [ChannelFact] contributed. It is carried alongside
	// candidates rather than converted into them because fusion ranks
	// episodes, and a fact is not one.
	facts   []Fact
	elapsed time.Duration
	err     error
}

// fuse combines per-channel rankings into one list, best first.
//
// A failed channel contributes nothing, even partial results: a truncated
// ranking has misleading ranks, and half a channel is worse input than none.
// Within one channel a repeated episode counts once, at its best rank.
//
// Ties break towards the more recent episode. Ids are uuidv7, so the larger id
// is the newer one, which makes the order meaningful and fully deterministic.
func fuse(results []channelResult, opts FusionOptions) []RecalledEpisode {
	type entry struct {
		episode Episode
		score   float64
		hits    []ChannelHit
	}

	order := make([]EpisodeID, 0, len(results)*DefaultCandidatesPerChannel)
	byID := make(map[EpisodeID]*entry)

	for _, res := range results {
		if res.err != nil {
			continue
		}
		weight := opts.weight(res.channel)
		seen := make(map[EpisodeID]struct{}, len(res.candidates))
		for _, cand := range res.candidates {
			id := cand.Episode.ID
			if _, dup := seen[id]; dup {
				continue
			}
			seen[id] = struct{}{}

			acc, ok := byID[id]
			if !ok {
				acc = &entry{episode: cand.Episode}
				byID[id] = acc
				order = append(order, id)
			}
			acc.score += weight / (opts.K + float64(cand.Rank))
			acc.hits = append(acc.hits, ChannelHit{
				Channel: res.channel,
				Rank:    cand.Rank,
				Score:   cand.Score,
			})
		}
	}

	fused := make([]RecalledEpisode, 0, len(order))
	for _, id := range order {
		acc := byID[id]
		slices.SortFunc(acc.hits, func(a, b ChannelHit) int {
			return cmp.Compare(a.Channel, b.Channel)
		})
		fused = append(fused, RecalledEpisode{
			Episode: acc.episode,
			Score:   acc.score,
			Hits:    acc.hits,
		})
	}

	slices.SortStableFunc(fused, func(a, b RecalledEpisode) int {
		if c := cmp.Compare(b.Score, a.Score); c != 0 { // score, descending
			return c
		}
		return bytes.Compare(b.Episode.ID[:], a.Episode.ID[:]) // newer id first
	})
	return fused
}
