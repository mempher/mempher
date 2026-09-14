// Scoring one recollection, and aggregating many.

package eval

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"time"

	"github.com/mempher/mempher"
)

// score grades one recollection against a question's labels.
//
// Relevance is binary: an episode either carries the answer or it does not. The
// benchmark labels it that way, and inventing grades it does not have would make
// the number less comparable, not more precise.
func score(q Question, rec mempher.Recollection, limit int) Result {
	relevant := make(map[int]bool, len(q.Relevant))
	for _, index := range q.Relevant {
		relevant[index] = true
	}
	result := Result{
		ID:       q.ID,
		Type:     q.Type,
		Relevant: len(relevant),
		Returned: len(rec.Episodes),
	}

	var gain float64
	found := 0
	for rank, recalled := range rec.Episodes {
		// Sequence numbers are allocated densely from 1 in append order, so
		// this is the index the label refers to.
		if !relevant[int(recalled.Episode.Seq)-1] {
			continue
		}
		found++
		if !result.Hit {
			result.Hit = true
			result.ReciprocalRank = 1 / float64(rank+1)
		}
		gain += 1 / math.Log2(float64(rank+2))
	}

	result.Recall = float64(found) / float64(len(relevant))
	if ideal := idcg(min(len(relevant), limit)); ideal > 0 {
		result.NDCG = gain / ideal
	}
	return result
}

// idcg is the gain of a perfect ranking of n relevant episodes, which is what
// normalises the score into [0,1].
func idcg(n int) float64 {
	var ideal float64
	for i := range n {
		ideal += 1 / math.Log2(float64(i+2))
	}
	return ideal
}

// TypeSummary aggregates one question category. It exists because a number that
// is respectable overall and poor on one category is the interesting case, and
// an average is exactly what hides it.
type TypeSummary struct {
	Type      string
	Questions int
	Hits      int
	HitRate   float64
	Recall    float64
	MRR       float64
	NDCG      float64

	recall, rr, ndcg float64
}

// Summary is what a run measured.
type Summary struct {
	// Dataset names the corpus, Variant the configuration that was measured,
	// Channels the channels it ran, and Limit is the k every metric is at.
	Dataset  string
	Variant  string
	Channels []mempher.Channel
	Limit    int
	// Questions is how many were scored, and Skipped how many were passed
	// over for labelling no relevant episode at all.
	Questions int
	Skipped   int
	// Hits is how many questions returned at least one relevant episode, and
	// Empty how many returned no episode at all.
	//
	// Empty separates two very different failures. A question that returned ten
	// episodes and none of them right is a ranking problem. A question that
	// returned nothing is a matching problem, and no amount of reranking will
	// touch it.
	Hits  int
	Empty int
	// HitRate is the fraction of questions whose answer was in the result at
	// all. For a memory layer it is the number that matters most: an agent
	// needs the evidence in its context, not all of it.
	HitRate float64
	// EmptyRate is the fraction of questions recall answered with nothing.
	EmptyRate float64
	// Recall is the mean fraction of relevant episodes returned, MRR the mean
	// reciprocal rank of the first one, and NDCG the mean normalised gain.
	Recall float64
	MRR    float64
	NDCG   float64
	// ByType breaks the same metrics down by question category.
	ByType map[string]*TypeSummary
	// P50 and P95 are recall latency, and Ingested is the total time spent
	// writing haystacks, which is not part of what recall costs.
	P50, P95 time.Duration
	Ingested time.Duration

	recall, rr, ndcg float64
	latencies        []time.Duration
}

func (s *Summary) add(r Result) {
	s.Questions++
	if r.Hit {
		s.Hits++
	}
	if r.Returned == 0 {
		s.Empty++
	}
	s.recall += r.Recall
	s.rr += r.ReciprocalRank
	s.ndcg += r.NDCG
	s.latencies = append(s.latencies, r.Latency)
	s.Ingested += r.Ingest

	byType := s.ByType[r.Type]
	if byType == nil {
		byType = &TypeSummary{Type: r.Type}
		s.ByType[r.Type] = byType
	}
	byType.Questions++
	if r.Hit {
		byType.Hits++
	}
	byType.recall += r.Recall
	byType.rr += r.ReciprocalRank
	byType.ndcg += r.NDCG
}

func (s *Summary) finish() {
	if s.Questions > 0 {
		n := float64(s.Questions)
		s.HitRate = float64(s.Hits) / n
		s.EmptyRate = float64(s.Empty) / n
		s.Recall = s.recall / n
		s.MRR = s.rr / n
		s.NDCG = s.ndcg / n
	}
	for _, byType := range s.ByType {
		if byType.Questions == 0 {
			continue
		}
		n := float64(byType.Questions)
		byType.HitRate = float64(byType.Hits) / n
		byType.Recall = byType.recall / n
		byType.MRR = byType.rr / n
		byType.NDCG = byType.ndcg / n
	}
	slices.Sort(s.latencies)
	s.P50 = percentile(s.latencies, 0.50)
	s.P95 = percentile(s.latencies, 0.95)
}

func percentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	index := int(p * float64(len(sorted)-1))
	return sorted[index]
}

// String renders the summary as the table a run is worth reporting as.
func (s Summary) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s -- %s -- %d scored, %d skipped, k=%d\n\n",
		s.Dataset, s.Variant, s.Questions, s.Skipped, s.Limit)
	fmt.Fprintf(&b, "  %-26s %5s  %8s %10s %8s %7s\n",
		"category", "n", "hit@k", "recall@k", "nDCG@k", "MRR")
	fmt.Fprintf(&b, "  %s\n", strings.Repeat("-", 70))
	fmt.Fprintf(&b, "  %-26s %5d  %8.3f %10.3f %8.3f %7.3f\n",
		"overall", s.Questions, s.HitRate, s.Recall, s.NDCG, s.MRR)

	types := make([]string, 0, len(s.ByType))
	for name := range s.ByType {
		types = append(types, name)
	}
	slices.Sort(types)
	for _, name := range types {
		t := s.ByType[name]
		fmt.Fprintf(&b, "  %-26s %5d  %8.3f %10.3f %8.3f %7.3f\n",
			name, t.Questions, t.HitRate, t.Recall, t.NDCG, t.MRR)
	}
	fmt.Fprintf(&b, "\n  %d of %d questions (%.1f%%) matched nothing at all\n",
		s.Empty, s.Questions, 100*s.EmptyRate)
	fmt.Fprintf(&b, "  recall latency: p50 %s, p95 %s (ingest %s, not part of recall)\n",
		s.P50.Round(time.Microsecond), s.P95.Round(time.Microsecond),
		s.Ingested.Round(time.Millisecond))
	return b.String()
}
