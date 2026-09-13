// The shipped prompts, and how one extraction is rendered for a model.

package extract

import (
	"fmt"
	"strings"
	"time"

	"github.com/mempher/mempher"
)

// extractionSystem is the standing instruction for the call that produces facts.
//
// Most of it defends one property: the same claim, extracted twice, must render
// as the same triple. Everything a FactStore does -- recognising a re-assertion,
// rejecting a contradiction, closing a window instead of overwriting a row --
// depends on that, and nothing downstream can repair a model that calls the same
// relation two different things.
const extractionSystem = `You read a fragment of an agent's conversation log and decide what, if
anything, about it is worth remembering durably.

Return JSON conforming to the schema you were given, and nothing else.

WHAT TO RECORD

Most fragments yield nothing. Returning no assertions and no retractions is a
good answer and the common one. An extractor that finds something durable in
every exchange fills a memory with noise, and that noise is then read back on
every single recall -- which is worse than finding nothing at all.

Record a claim only if all of these hold:
  - it would still matter in a month, to someone who was not present;
  - it is about the world, not about this conversation. "The user asked about
    refunds" is a transcript, not a fact;
  - it was stated or clearly implied, not inferred from one hedged remark.

Do not record pleasantries, acknowledgements, questions, the agent's own output
unless it records a commitment it has made, or anything the speaker framed as
uncertain, hypothetical or someone else's opinion.

SHAPE

Each assertion is a triple with a sentence beside it:

  subject    what the claim is about
  predicate  the relation, as a lowercase snake_case slug containing no spaces
  object     the value of that relation, one value per assertion
  statement  the claim as one plain sentence -- this is what a model will later
             be shown, and what full-text search will match against

The triple is identity. The same claim extracted twice must produce the same
three values or the memory will hold it twice, so:

  - Keep subjects stable. The person the agent serves is "the user" in every
    assertion, never "he" in one and their name in the next. Reuse a subject
    that already appears in the known facts before writing a new one.
  - Keep predicates stable. Reuse a predicate that already appears in the known
    facts before inventing one. "lives_in" and "resides_in" are one relation and
    must not both exist.
  - Put one thing in each object. Being allergic to two things is two
    assertions, not one object holding both.

TIME

valid_from is when the claim became true in the world, which is usually not when
it was mentioned. Resolve relative expressions -- "last spring", "since
Tuesday" -- against the current instant given below. When nothing in the text
dates the claim, use the occurrence time of the episode that stated it.

valid_to is when the claim stopped being true. It is usually absent, because
most claims are still true. Set it only when a single episode describes a span
that has already ended.

confidence is required, between 0 and 1. Use it honestly: 1.0 for something
stated plainly, lower for something merely implied.

WHAT CHANGED

You are shown the facts already believed, each with an id.

  - If an episode contradicts one, retract it -- at the instant it stopped being
    true, not now. "I moved to Berlin in June" closes the Paris fact on the
    first of June, whatever today's date is. This is the whole difference
    between a memory that can answer "where did they live last year" and one
    that answers it with where they live today.
  - If an episode merely restates a known fact, do nothing. It is recorded.
  - If an episode adds a claim that stands alongside a known one rather than
    replacing it, assert it and retract nothing. A second allergy does not
    cancel the first.
  - Never retract a fact the episodes say nothing about.`

// topicsPrompt is the standing instruction for the first pass of
// [Options.Reconcile]. It asks for a routing decision, not for facts: what the
// fragment is about, so that the extraction which follows sees the handful of
// known facts that could be contradicted rather than a hundred that could not.
const topicsPrompt = `You read a fragment of an agent's conversation log and name what it is about, so
that only the relevant part of a large memory has to be consulted.

Return JSON conforming to the schema you were given, and nothing else.

List every subject the fragment says anything about, and every relation it
touches, whether or not it makes a new claim about them. "I'm not vegetarian any
more" names the subject "the user" and the relation "diet" while asserting
nothing at all.

Use the vocabulary an extraction would: the person the agent serves is "the
user", and relations are lowercase snake_case slugs.

Be generous. Naming a subject that turns out to be irrelevant costs one line of
a later prompt. Missing one hides a fact that should have been superseded.`

// extractionPrompt assembles the standing instruction: the shipped prompt, the
// caller's domain addendum, and the limits this configuration imposes.
//
// The addendum goes last so that a deployment can add to the shipped rules but
// reads as adding to them; the assertion cap goes last of all because it is the
// instruction a model is most likely to drop.
func extractionPrompt(opts Options) string {
	var b strings.Builder
	b.WriteString(extractionSystem)
	if strings.TrimSpace(opts.Instructions) != "" {
		b.WriteString("\n\nWHAT THIS DEPLOYMENT CARES ABOUT\n\n")
		b.WriteString(strings.TrimSpace(opts.Instructions))
	}
	if len(opts.Predicates) > 0 {
		b.WriteString("\n\nVOCABULARY\n\nUse only these predicates, and no others:\n")
		for _, p := range opts.Predicates {
			fmt.Fprintf(&b, "  %s\n", p)
		}
		b.WriteString("\nA claim that fits none of them is a claim not to record.")
	}
	fmt.Fprintf(&b, "\n\nReturn at most %d assertions. If more seem to qualify, return the %d that\n"+
		"would matter most in a month.", opts.MaxAssertions, opts.MaxAssertions)
	return b.String()
}

// renderNow states the instant every relative date in the episodes resolves
// against. It comes from [mempher.ExtractRequest.Now] rather than a clock read
// here, which is what lets a backfill reproduce an extraction instead of
// re-deciding it against today.
func renderNow(b *strings.Builder, now time.Time) {
	fmt.Fprintf(b, "The current instant is %s.\n", now.Format(time.RFC3339))
}

// renderKnown lists the facts currently believed, with the ids a retraction
// names. An empty set is stated rather than omitted: a model shown no heading at
// all tends to invent one.
func renderKnown(b *strings.Builder, known []mempher.Fact) {
	b.WriteString("\nKNOWN FACTS\n\n")
	if len(known) == 0 {
		b.WriteString("Nothing is believed about this subject yet.\n")
		return
	}
	for _, f := range known {
		window := "still true"
		if !f.Valid.IsOpen() {
			window = "ended " + f.Valid.To.Format(time.RFC3339)
		}
		fmt.Fprintf(b, "[%s] %s\n    %s %s %q | since %s, %s | confidence %.2f\n",
			f.ID, f.Statement,
			f.Subject, f.Predicate, f.Object,
			f.Valid.From.Format(time.RFC3339), window, f.Confidence)
	}
}

// renderEpisodes lists the new episodes in Seq order, which the worker has
// already sorted them into: an extractor reading a conversation out of order
// resolves "then" and "that" against the wrong episode.
func renderEpisodes(b *strings.Builder, episodes []mempher.Episode) {
	b.WriteString("\nNEW EPISODES\n\n")
	for _, ep := range episodes {
		speaker := string(ep.Role)
		if ep.Actor != "" {
			speaker = fmt.Sprintf("%s (%s)", ep.Role, ep.Actor)
		}
		fmt.Fprintf(b, "#%d %s %s:\n%s\n\n",
			ep.Seq, ep.OccurredAt.Format(time.RFC3339), speaker, ep.Content)
	}
}
