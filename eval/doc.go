// Package eval measures what [mempher.Memory.Recall] actually puts in front of
// a model.
//
// Latency is already answered by the benchmarks in the root package. This is
// the other question, and the harder one: when the answer is somewhere in a
// scope, does recall return it?
//
// # What is measured, and what is not
//
// Retrieval, not question answering. A labelled question names the episodes
// that carry its answer, so the metrics here are computable without a model
// judging anything -- which is what makes them reproducible, cheap, and free of
// an opinion about which generator you would have used.
//
// That is deliberate rather than a shortcut. mempher is a memory substrate, not
// an agent: an end-to-end accuracy number would mostly measure the model
// reading the context, and would move when that model changed. What this
// library is responsible for is whether the evidence was in the context at all.
//
// # Running one
//
// A [Dataset] is labelled questions. [Run] ingests each question's haystack into
// a scope of its own, recalls against it, and scores what came back.
//
// Nothing here needs a model. With no [Worker] configured, no episode is ever
// encoded and the semantic channel has nothing to return, so a run over the
// lexical channel alone needs no API key and no budget -- and is a floor rather
// than a result. Configure a real [mempher.Embedder] and a worker to measure the
// whole thing.
package eval
