// Package extract is a reference [mempher.Extractor] built on one structured
// model call.
//
// The root package declares [mempher.Extractor] and deliberately never
// implements it, because what is worth remembering is a judgement about a domain
// rather than a property of storage. This package is the judgement most
// deployments would otherwise write from scratch: a prompt, a response schema,
// and the validation that decides what a [mempher.FactStore] will accept.
//
// # What a caller supplies
//
// One method. A [Completer] hands a [Prompt] to a provider and returns what it
// said, which is about twenty lines against any SDK. Everything else -- what to
// ask, how to ask for it, how to read the answer, and what to do when the answer
// is wrong -- is here, and needs no dependency beyond the standard library.
//
// # Identity
//
// Facts are keyed by extractor so that two extractors' opinions never merge
// silently into one scope, which makes the id this package reports the most
// consequential thing in it. It composes the three things that decide what gets
// asserted: the prompt revision shipped here, the provider model, and a digest
// of [Options].
//
//	mempher-extract/v1+claude-opus-5+a3f9c1e2
//
// So editing [Options.Instructions] makes a different extractor, and the facts
// the old one asserted stay under the old id until a backfill re-derives the
// scope. That is the same contract as changing embedding model: the facts are
// not lost, they are simply not this extractor's. Read [Extractor.Model] before
// deciding that is surprising.
//
// # What this package cannot do
//
// Extraction is a pure function of [mempher.ExtractRequest]. An extractor that
// queried a [mempher.FactStore] could not be replayed, and replay is the only
// repair L1 has -- so the facts an extraction may retract are exactly the facts
// it was handed, and how many of those there are is
// [mempher.WorkerConfig.KnownFactLimit], decided by the worker rather than here.
// [Options.Reconcile] spends a second model call to use that set better. It
// cannot enlarge it.
package extract
