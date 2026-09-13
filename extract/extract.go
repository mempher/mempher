// The port a caller supplies, the configuration, and the extraction loop.

package extract

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"

	"github.com/mempher/mempher"
)

// Defaults and the prompt revision.
const (
	// PromptVersion is the revision of the prompt and response schema this
	// package ships.
	//
	// It is part of the extractor id, so a release that moves it makes every
	// deployment's existing facts the previous revision's facts until a
	// backfill re-derives them. It therefore moves on a minor version and is
	// called out in the changelog, never on a patch.
	PromptVersion = 1

	// DefaultMaxAssertions is how many claims one extraction may assert when
	// [Options.MaxAssertions] does not say.
	//
	// It is far below [mempher.MaxAssertionsPerExtraction], which bounds what
	// a FactStore will accept rather than what a model can emit: sixty-four
	// statements at their permitted length is more output than a provider will
	// return in one call. The useful direction for this number is down.
	DefaultMaxAssertions = 16

	// DefaultMaxPromptBytes is how much episode content one extraction will
	// send when [Options.MaxPromptBytes] does not say.
	//
	// A job may name up to [mempher.MaxAppendBatch] episodes and an episode may
	// hold [mempher.MaxContentLen] bytes, so the largest thing a worker can
	// hand an extractor is sixty-four megabytes. Something has to refuse it,
	// and refusing here gives a legible error rather than a provider's.
	DefaultMaxPromptBytes = 256 << 10
)

// Prompt is one extraction, ready to send.
type Prompt struct {
	// System is the standing instruction: the prompt shipped here, plus
	// [Options.Instructions].
	System string
	// User is this extraction's episodes, known facts and current instant.
	User string
	// Schema is the JSON Schema the response must conform to.
	//
	// Pass it to the provider's structured-output mechanism -- a tool
	// definition, response_format, responseSchema -- rather than pasting it
	// into the text. That is what makes [ErrMalformedResponse] rare instead of
	// routine, and it is where a closed predicate vocabulary is enforced.
	Schema json.RawMessage
}

// Completer is one structured model call: the whole of what a caller supplies.
//
// It is narrow on purpose. Everything that makes extraction hard -- the prompt,
// the schema, resolving dates against the injected instant, matching retractions
// to known facts, and rejecting what a FactStore would refuse -- lives in this
// package. An adapter is the twenty lines that hand a [Prompt] to a provider and
// give back what it said.
//
// Implementations must be safe for concurrent use and must honour cancellation.
// They must also be as close to deterministic as the provider allows:
// temperature zero, no nucleus sampling. A reclaimed lease replays a job, and a
// model that answers differently the second time asserts the same claim over an
// overlapping window, which is [mempher.ErrFactConflict] rather than a merged
// answer.
type Completer interface {
	// Complete returns the model's response to p, which must be JSON
	// conforming to p.Schema. An adapter returns those bytes rather than
	// interpreting them: reading the answer is this package's job.
	Complete(ctx context.Context, p Prompt) ([]byte, error)

	// Model identifies the provider model, such as "claude-opus-5". It is one
	// component of [Extractor.Model], which facts are keyed by, so it must
	// change when the model does.
	Model() string
}

// Options tunes the shipped extractor. The zero value works.
//
// Every field here except Logger and MaxPromptBytes changes what the model is
// asked and therefore what it asserts, so every one of them is part of the
// extractor's identity. See [Extractor.Model].
type Options struct {
	// Predicates, when non-empty, is the closed vocabulary an extraction may
	// use. It is the most valuable field in this struct.
	//
	// A triple is identity: re-asserting the same claim is an upsert only if
	// the claim renders as the same triple twice. An unconstrained model writes
	// "lives_in" on Monday and "resides_in" on Tuesday, the two never collide,
	// so nothing dedupes, nothing supersedes, and the temporal key never gets
	// the chance to reject a contradiction.
	//
	// It is emitted into the schema as an enum, so the provider's own
	// constraint does the work and the check here is only a backstop. Order and
	// repetition do not matter: it is treated as a set.
	Predicates []mempher.Predicate

	// Instructions is a domain addendum appended to the shipped prompt: what
	// this deployment considers worth remembering, in its own words. It never
	// replaces the shipped prompt, which carries the rules a FactStore would
	// enforce anyway.
	Instructions string

	// MaxAssertions caps one extraction. Zero means [DefaultMaxAssertions].
	// It is a salience lever as much as a limit: a model told it may return
	// four facts chooses more carefully than one told it may return sixty.
	MaxAssertions int

	// Reconcile spends a first, cheap model call naming the subjects and
	// relations the episodes touch, so that the extraction itself is shown only
	// the known facts that could possibly be contradicted.
	//
	// It buys precision, not reach. A model given a hundred known facts and
	// told to retract what is contradicted misses; the same model given the
	// three that share a subject does not. What it cannot do is show the
	// extraction a fact that was never handed to it -- that set is
	// [mempher.WorkerConfig.KnownFactLimit], chosen by the worker, and no
	// extractor may go looking for more without giving up replay.
	//
	// Off by default, because it doubles the model calls and a scope whose
	// facts already fit in one prompt gains little.
	Reconcile bool

	// MaxPromptBytes caps the episode content one extraction will send. Zero
	// means [DefaultMaxPromptBytes]. Exceeding it is [ErrPromptTooLarge], which
	// fails the job rather than shortening the prompt.
	MaxPromptBytes int

	// Logger receives the assertions and retractions dropped for failing
	// validation. Nil means [slog.Default].
	//
	// They are dropped rather than fatal, because one hallucinated predicate
	// should not cost a turn its good facts. They are logged rather than
	// swallowed, because a model that has quietly started answering badly would
	// otherwise be indistinguishable from a scope with nothing in it.
	Logger *slog.Logger
}

// Extractor is a [mempher.Extractor] that asks a model. It is safe for
// concurrent use and holds no mutable state.
type Extractor struct {
	completer    Completer
	id           mempher.ExtractorID
	opts         Options
	predicates   map[mempher.Predicate]struct{}
	system       string
	topics       string
	schema       json.RawMessage
	topicsSchema json.RawMessage
	log          *slog.Logger
}

// Assert at compile time what this package is for.
var _ mempher.Extractor = (*Extractor)(nil)

// New returns an [Extractor] built on c.
//
// It performs no I/O, so it needs no context, and it rejects a configuration
// that cannot work rather than failing on the first job.
func New(c Completer, opts Options) (*Extractor, error) {
	if c == nil {
		return nil, fmt.Errorf("mempher/extract: new: Completer is required: %w",
			mempher.ErrInvalidConfig)
	}
	model := c.Model()
	if model == "" {
		return nil, fmt.Errorf("mempher/extract: new: completer reports no model id: %w",
			mempher.ErrInvalidConfig)
	}

	switch {
	case opts.MaxAssertions < 0:
		return nil, fmt.Errorf("mempher/extract: new: MaxAssertions is %d: %w",
			opts.MaxAssertions, mempher.ErrInvalidConfig)
	case opts.MaxAssertions > mempher.MaxAssertionsPerExtraction:
		return nil, fmt.Errorf(
			"mempher/extract: new: MaxAssertions is %d, but a FactStore accepts at most %d: %w",
			opts.MaxAssertions, mempher.MaxAssertionsPerExtraction, mempher.ErrInvalidConfig)
	case opts.MaxPromptBytes < 0:
		return nil, fmt.Errorf("mempher/extract: new: MaxPromptBytes is %d: %w",
			opts.MaxPromptBytes, mempher.ErrInvalidConfig)
	}
	if opts.MaxAssertions == 0 {
		opts.MaxAssertions = DefaultMaxAssertions
	}
	if opts.MaxPromptBytes == 0 {
		opts.MaxPromptBytes = DefaultMaxPromptBytes
	}

	// A set, so that two spellings of the same vocabulary are one extractor.
	predicates := slices.Clone(opts.Predicates)
	slices.Sort(predicates)
	predicates = slices.Compact(predicates)
	for _, p := range predicates {
		if err := p.Validate(); err != nil {
			// Both sentinels: the value is one a fact could not carry, and
			// finding that out at construction makes it a configuration error,
			// which is what New documents and what a caller checks for.
			return nil, fmt.Errorf("mempher/extract: new: predicate vocabulary: %w: %w",
				err, mempher.ErrInvalidConfig)
		}
	}
	opts.Predicates = predicates

	id := mempher.ExtractorID(fmt.Sprintf("mempher-extract/v%d+%s+%s",
		PromptVersion, model, digest(opts)))
	if len(id) > mempher.MaxExtractorLen {
		return nil, fmt.Errorf(
			"mempher/extract: new: extractor id %q is %d bytes, limit is %d: %w",
			id, len(id), mempher.MaxExtractorLen, mempher.ErrInvalidConfig)
	}

	schema, err := extractionSchema(opts)
	if err != nil {
		return nil, err
	}
	topicSchema, err := topicsSchema()
	if err != nil {
		return nil, err
	}

	e := &Extractor{
		completer:    c,
		id:           id,
		opts:         opts,
		system:       extractionPrompt(opts),
		topics:       topicsPrompt,
		schema:       schema,
		topicsSchema: topicSchema,
		log:          opts.Logger,
	}
	if len(predicates) > 0 {
		e.predicates = make(map[mempher.Predicate]struct{}, len(predicates))
		for _, p := range predicates {
			e.predicates[p] = struct{}{}
		}
	}
	if e.log == nil {
		e.log = slog.Default()
	}
	return e, nil
}

// Model returns the extractor id: this package's prompt revision, the provider
// model, and a digest of [Options].
//
//	mempher-extract/v1+claude-opus-5+a3f9c1e2
//
// All three belong in it. Facts are keyed by extractor precisely so that two
// extractors' opinions never merge into one scope, and changing the
// instructions or the predicate vocabulary makes a different extractor in the
// only sense that matters: it changes what gets asserted.
//
// So changing [Options] is a backfill, exactly as changing embedding model is.
// The facts already asserted are not destroyed -- they are still there, under
// the id that asserted them -- but recall will not return them until the scope
// has been re-derived under the new id, because a fact query names one
// extractor. That is the projection contract, not an accident.
func (e *Extractor) Model() mempher.ExtractorID { return e.id }

// Schema returns the response schema this extractor asks for. It is the first
// thing to read when a model keeps returning something unexpected.
func (e *Extractor) Schema() json.RawMessage { return slices.Clone(e.schema) }

// Extract reads the episodes and returns what to assert and retract.
//
// An empty result is a valid and common answer: most episodes carry no durable
// fact. Anything the model returns that a [mempher.FactStore] would refuse is
// dropped and logged rather than failing the extraction, so one bad triple does
// not cost a turn its good ones.
func (e *Extractor) Extract(
	ctx context.Context,
	req mempher.ExtractRequest,
) (mempher.ExtractResult, error) {
	if len(req.Episodes) == 0 {
		return mempher.ExtractResult{}, nil
	}
	if err := e.budget(req.Episodes); err != nil {
		return mempher.ExtractResult{}, err
	}

	known := req.Known
	if e.opts.Reconcile && len(known) > 0 {
		narrowed, err := e.narrow(ctx, req)
		if err != nil {
			return mempher.ExtractResult{}, err
		}
		known = narrowed
	}
	return e.extract(ctx, req, known)
}

// extract is the call that produces facts: episodes, the known facts chosen for
// it, and the instant that resolves every relative date in the text.
func (e *Extractor) extract(
	ctx context.Context,
	req mempher.ExtractRequest,
	known []mempher.Fact,
) (mempher.ExtractResult, error) {
	var user strings.Builder
	renderNow(&user, req.Now)
	renderKnown(&user, known)
	renderEpisodes(&user, req.Episodes)

	raw, err := e.completer.Complete(ctx, Prompt{
		System: e.system,
		User:   user.String(),
		Schema: e.schema,
	})
	if err != nil {
		return mempher.ExtractResult{}, fmt.Errorf("mempher/extract: complete: %w", err)
	}
	body, err := decode[extraction](raw)
	if err != nil {
		return mempher.ExtractResult{}, err
	}
	// Retractions are checked against everything the worker supplied rather
	// than the narrowed set: a fact id the model returns is either real or
	// invented, and which facts it was shown does not change that.
	return e.validate(ctx, body, req), nil
}

// narrow runs the first pass of [Options.Reconcile]: a cheap call naming the
// subjects and relations the episodes touch, used to choose which known facts
// the extraction itself will see.
//
// A first pass that names nothing falls back to the full set. The failure worth
// guarding is a flaky small call silently switching supersession off, which
// would look exactly like a scope where nothing ever changes.
func (e *Extractor) narrow(
	ctx context.Context,
	req mempher.ExtractRequest,
) ([]mempher.Fact, error) {
	var user strings.Builder
	renderNow(&user, req.Now)
	renderEpisodes(&user, req.Episodes)

	raw, err := e.completer.Complete(ctx, Prompt{
		System: e.topics,
		User:   user.String(),
		Schema: e.topicsSchema,
	})
	if err != nil {
		return nil, fmt.Errorf("mempher/extract: complete topics: %w", err)
	}
	body, err := decode[topics](raw)
	if err != nil {
		return nil, err
	}
	if len(body.Subjects) == 0 && len(body.Predicates) == 0 {
		return req.Known, nil
	}

	subjects := lowerSet(body.Subjects)
	predicates := lowerSet(body.Predicates)
	kept := make([]mempher.Fact, 0, len(req.Known))
	for _, f := range req.Known {
		_, bySubject := subjects[strings.ToLower(string(f.Subject))]
		_, byPredicate := predicates[strings.ToLower(string(f.Predicate))]
		if bySubject || byPredicate {
			kept = append(kept, f)
		}
	}
	e.log.DebugContext(ctx, "mempher/extract: narrowed the known facts",
		"from", len(req.Known), "to", len(kept))
	return kept, nil
}

// budget refuses an extraction whose episodes hold more content than one prompt
// should carry. See [ErrPromptTooLarge] for why this fails rather than trims.
func (e *Extractor) budget(episodes []mempher.Episode) error {
	total := 0
	for _, ep := range episodes {
		total += len(ep.Content)
	}
	if total > e.opts.MaxPromptBytes {
		return fmt.Errorf(
			"mempher/extract: %d episodes hold %d bytes of content, limit is %d: %w",
			len(episodes), total, e.opts.MaxPromptBytes, ErrPromptTooLarge)
	}
	return nil
}

// digest fingerprints everything in opts that changes what the model is asked.
// Values are length-prefixed so that instructions containing the separator
// cannot forge a different configuration's digest.
func digest(opts Options) string {
	h := sha256.New()
	// The error is discarded because hash.Hash.Write is documented never to
	// return one.
	write := func(key, value string) {
		_, _ = fmt.Fprintf(h, "%s:%d:%s\n", key, len(value), value)
	}
	write("prompt", strconv.Itoa(PromptVersion))
	write("instructions", opts.Instructions)
	write("assertions", strconv.Itoa(opts.MaxAssertions))
	write("reconcile", strconv.FormatBool(opts.Reconcile))
	for _, p := range opts.Predicates {
		write("predicate", string(p))
	}
	return hex.EncodeToString(h.Sum(nil))[:8]
}

// lowerSet indexes strings case-insensitively, because a model that answers
// "The User" once and "the user" the next time is describing one subject.
func lowerSet(values []string) map[string]struct{} {
	out := make(map[string]struct{}, len(values))
	for _, v := range values {
		out[strings.ToLower(strings.TrimSpace(v))] = struct{}{}
	}
	return out
}

// prompt returns the standing instruction this extractor sends, which is what a
// test asserts about and what an operator reads when a model starts answering
// oddly.
func (e *Extractor) prompt() string { return e.system }
