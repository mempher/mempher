// Package memphertest provides deterministic test doubles for mempher's injected
// ports, so code using mempher can be tested without a model provider and
// without waiting for wall-clock time.
//
// Determinism in mempher comes from these seams -- a [Clock], an [Embedder] and
// an [Extractor] -- and never from mocking the database: retrieval depends on
// real pgvector distance ordering, real PostgreSQL text search ranking and real
// temporal constraints, all of which a mock would reimplement wrongly.
package memphertest
