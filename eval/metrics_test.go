package eval

import (
	"math"
	"testing"

	"github.com/mempher/mempher"
)

// recollection builds a result whose episodes carry the given sequence
// numbers, best first, which is all scoring looks at.
func recollection(seqs ...int64) mempher.Recollection {
	var rec mempher.Recollection
	for _, seq := range seqs {
		rec.Episodes = append(rec.Episodes,
			mempher.RecalledEpisode{Episode: mempher.Episode{Seq: seq}})
	}
	return rec
}

func TestScore(t *testing.T) {
	// Sequence numbers are allocated densely from 1, so seq 3 is haystack
	// index 2. Getting that off by one would silently score the wrong episode.
	cases := map[string]struct {
		relevant []int
		returned []int64
		hit      bool
		recall   float64
		rr       float64
	}{
		"the only relevant episode, ranked first": {
			relevant: []int{0}, returned: []int64{1, 5, 9},
			hit: true, recall: 1, rr: 1,
		},
		"the only relevant episode, ranked third": {
			relevant: []int{4}, returned: []int64{1, 2, 5},
			hit: true, recall: 1, rr: 1.0 / 3,
		},
		"one of two found": {
			relevant: []int{4, 7}, returned: []int64{5, 1, 2},
			hit: true, recall: 0.5, rr: 1,
		},
		"nothing found": {
			relevant: []int{4, 7}, returned: []int64{1, 2, 3},
			hit: false, recall: 0, rr: 0,
		},
		"nothing returned at all": {
			relevant: []int{4}, returned: nil,
			hit: false, recall: 0, rr: 0,
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := score(
				Question{ID: "q", Relevant: tc.relevant},
				recollection(tc.returned...), 10)

			if got.Hit != tc.hit {
				t.Errorf("Hit = %v, want %v", got.Hit, tc.hit)
			}
			if math.Abs(got.Recall-tc.recall) > 1e-9 {
				t.Errorf("Recall = %v, want %v", got.Recall, tc.recall)
			}
			if math.Abs(got.ReciprocalRank-tc.rr) > 1e-9 {
				t.Errorf("ReciprocalRank = %v, want %v", got.ReciprocalRank, tc.rr)
			}
			if got.NDCG < 0 || got.NDCG > 1 {
				t.Errorf("NDCG = %v, want it inside [0,1]", got.NDCG)
			}
		})
	}
}

func TestNDCGRewardsTheBetterRanking(t *testing.T) {
	q := Question{Relevant: []int{0, 1}}
	perfect := score(q, recollection(1, 2, 9, 8), 10)
	worse := score(q, recollection(9, 8, 1, 2), 10)

	if perfect.NDCG != 1 {
		t.Errorf("a perfect ranking scored %v, want 1", perfect.NDCG)
	}
	// Recall cannot tell these apart; that is what nDCG is here for.
	if perfect.Recall != worse.Recall {
		t.Fatalf("recall differs (%v, %v), so this tests the wrong thing",
			perfect.Recall, worse.Recall)
	}
	if worse.NDCG >= perfect.NDCG {
		t.Errorf("the worse ranking scored %v, not below %v", worse.NDCG, perfect.NDCG)
	}
}

func TestSummaryAggregates(t *testing.T) {
	s := Summary{ByType: map[string]*TypeSummary{}}
	s.add(Result{Type: "a", Hit: true, Recall: 1, ReciprocalRank: 1, NDCG: 1})
	s.add(Result{Type: "a", Hit: false})
	s.add(Result{Type: "b", Hit: true, Recall: 0.5, ReciprocalRank: 0.5, NDCG: 0.5})
	s.finish()

	if s.Questions != 3 || s.Hits != 2 {
		t.Fatalf("Questions/Hits = %d/%d, want 3/2", s.Questions, s.Hits)
	}
	if math.Abs(s.HitRate-2.0/3) > 1e-9 {
		t.Errorf("HitRate = %v, want 2/3", s.HitRate)
	}
	// The per-category breakdown is the point: an average of 0.5 here would
	// hide that one category scored 0.5 and the other 0.
	if got := s.ByType["a"].HitRate; got != 0.5 {
		t.Errorf("category a hit rate = %v, want 0.5", got)
	}
	if got := s.ByType["b"].HitRate; got != 1 {
		t.Errorf("category b hit rate = %v, want 1", got)
	}
}
