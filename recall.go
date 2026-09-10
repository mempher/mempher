// The read loop: per-channel queries, and the result with its provenance.

package mempher

import "time"

// Retrieval defaults, used when the corresponding request field is left zero.
const (
	// DefaultRecallLimit is the number of episodes [Memory.Recall] returns.
	DefaultRecallLimit = 10
	// DefaultCandidatesPerChannel is how deep each channel searches before
	// fusion, which needs more candidates than it returns.
	DefaultCandidatesPerChannel = 50
	// DefaultFactLimit is how many facts [Memory.Recall] returns when a
	// request does not say.
	DefaultFactLimit = 32
)

// Channel names one independent retrieval path. Channels run concurrently, each
// producing its own ranking, and [Memory.Recall] fuses them.
type Channel string

// The retrieval channels available in this stage.
const (
	// ChannelSemantic ranks by cosine distance between the query embedding
	// and the episode's encoding. It sees only encoded episodes.
	ChannelSemantic Channel = "semantic"
	// ChannelLexical ranks by full-text search over episode content. It sees
	// an episode the instant it is appended, because the tsvector is a
	// generated column.
	ChannelLexical Channel = "lexical"
	// ChannelFact reads L1: the facts currently believed about the scope.
	//
	// It is a channel for the two things a channel is for -- it can be
	// switched off in [RecallRequest.Channels], and its failure is reported in
	// [Recollection.Channels] rather than swallowed -- but it does not join
	// the fusion. It contributes to [Recollection.Facts], not
	// [Recollection.Episodes], because a fact and an episode are different
	// kinds of evidence and ranking them against each other would assert an
	// ordering that does not exist. The layers are not peers.
	ChannelFact Channel = "fact"
)

// Valid reports whether c is one of the defined channels.
func (c Channel) Valid() bool {
	switch c {
	case ChannelSemantic, ChannelLexical, ChannelFact:
		return true
	default:
		return false
	}
}

// String returns the channel name.
func (c Channel) String() string { return string(c) }

// ChannelQuery is the filter every channel pushes down into SQL. Pushing down
// matters: filtering after a channel's LIMIT would drop matches that were merely
// ranked below an excluded row.
type ChannelQuery struct {
	// Scope restricts the search to one partition. Required.
	Scope ScopeID
	// AsOf is the system-time cut: only episodes ingested at or before it are
	// visible. It answers what the agent could have recalled at that moment.
	AsOf time.Time
	// OccurredFrom and OccurredTo optionally bound event time; zero means
	// unbounded. Unlike AsOf this filters the world, not knowledge.
	OccurredFrom time.Time
	OccurredTo   time.Time
	// Binding, when non-empty, keeps only episodes whose binding contains
	// every one of these pairs.
	Binding Binding
	// Roles, when non-empty, keeps only episodes carrying one of them.
	Roles []Role
	// Limit is how many candidates this channel returns.
	Limit int
}

// SemanticQuery asks the vector channel for the nearest encodings.
type SemanticQuery struct {
	ChannelQuery
	// Vector is the embedded query.
	Vector Vector
	// Model selects which encodings to search.
	Model ModelID
}

// LexicalQuery asks the full-text channel for the best matches.
type LexicalQuery struct {
	ChannelQuery
	// Text is the raw query string, parsed with the text search configuration
	// the schema was migrated with.
	Text string
}

// Candidate is one episode as a single channel saw it, before fusion.
type Candidate struct {
	// Episode is the full episode, so fusion needs no second round trip.
	Episode Episode
	// Channel is the channel that produced it.
	Channel Channel
	// Rank is its 1-based position within that channel's results, and is what
	// fusion consumes.
	Rank int
	// Score is the channel's native score, kept for explanation only. Scores
	// from different channels are not comparable, which is why fusion uses
	// ranks.
	Score float64
}

// RecallRequest asks for the episodes most worth putting in front of a model.
type RecallRequest struct {
	// Scope restricts the search to one partition. Required.
	Scope ScopeID
	// Query is the natural-language cue. Required.
	Query string
	// AsOf is the system-time cut. Zero means now.
	AsOf time.Time
	// OccurredFrom and OccurredTo optionally bound event time.
	OccurredFrom time.Time
	OccurredTo   time.Time
	// Binding, when non-empty, keeps only episodes whose binding contains
	// every one of these pairs.
	Binding Binding
	// Roles, when non-empty, keeps only episodes carrying one of them.
	Roles []Role
	// Channels selects which channels to run. Empty means all of them.
	Channels []Channel
	// CandidatesPerChannel is the per-channel search depth. Zero means
	// [DefaultCandidatesPerChannel].
	CandidatesPerChannel int
	// Limit caps how many episodes are returned. Zero means
	// [DefaultRecallLimit].
	Limit int
	// FactLimit caps how many facts are returned. Zero means
	// [DefaultFactLimit].
	FactLimit int
	// MinFactConfidence drops facts the extractor was less sure of than this.
	// Zero keeps everything it asserted.
	MinFactConfidence float32
	// Subjects, when non-empty, keeps only facts about one of them. It does
	// not filter episodes, which have no subject.
	Subjects []Subject
	// MaxTokens caps the total token cost of the returned content, facts and
	// episodes together. Zero means no budget.
	//
	// Facts are budgeted first. They are the compact, current summary of the
	// scope, so dropping them to fit one more transcript line is a bad trade
	// at any budget; [Recollection.Truncated] says when it happened.
	MaxTokens int
}

// ChannelHit records that one channel found a given episode, and where.
type ChannelHit struct {
	// Channel is the channel that found it.
	Channel Channel
	// Rank is its 1-based position in that channel's ranking.
	Rank int
	// Score is that channel's native score.
	Score float64
}

// ChannelReport says what one channel did, so a thin result can be told from a
// broken one.
type ChannelReport struct {
	// Channel is the channel described.
	Channel Channel
	// Candidates is how many results it contributed.
	Candidates int
	// Duration is how long it took.
	Duration time.Duration
	// Err is why it contributed nothing, if it failed. Recall degrades rather
	// than failing while one channel succeeds, so this is the only place that
	// degradation is visible.
	Err error
}

// RecalledEpisode is one episode in a [Recollection], with the evidence for why
// it is there.
type RecalledEpisode struct {
	// Episode is the episode itself.
	Episode Episode
	// Score is the fused reciprocal rank fusion score.
	Score float64
	// Hits are the per-channel findings behind that score: the provenance of
	// the result.
	Hits []ChannelHit
	// Tokens is the counted cost of the episode's content.
	Tokens int
}

// RecalledFact is one fact in a [Recollection].
//
// It carries no fused score. Facts are not ranked against each other by
// relevance to the query: what is true about a scope is true whether or not the
// question mentioned it, and the query only decides which facts survive
// [RecallRequest.FactLimit] when there are more than fit.
type RecalledFact struct {
	// Fact is the fact itself, with its validity window and provenance.
	Fact Fact
	// Tokens is the counted cost of the fact's statement.
	Tokens int
}

// Recollection is what [Memory.Recall] returns.
type Recollection struct {
	// Scope is the partition searched.
	Scope ScopeID
	// AsOf is the system-time cut actually applied.
	//
	// Replaying the same request with this value reproduces the episode
	// results exactly. It bounds the facts too, to those asserted at or
	// before it, but L1 records only what is currently believed about the
	// past and not what was believed at a past instant: a fact retracted
	// since will be absent from a replay that once returned it. L0 is what is
	// reproducible; a projection is what is current.
	AsOf time.Time
	// Facts are the facts believed about the scope, ordered by subject and
	// predicate. They are evidence of a different kind from Episodes, so they
	// are returned beside them rather than mixed in.
	Facts []RecalledFact
	// Episodes are the fused results, best first.
	Episodes []RecalledEpisode
	// Channels reports what each channel contributed, including failures.
	Channels []ChannelReport
	// Tokens is the total counted cost of the returned content, facts and
	// episodes together.
	Tokens int
	// Truncated reports that a limit or the token budget cut results.
	Truncated bool
}
