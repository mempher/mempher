// Package memphertest provides deterministic test doubles for mempher's injected
// ports, so code using mempher can be tested without a model provider and
// without waiting for wall-clock time.
//
// Determinism in mempher comes from these two seams, a [Clock] and an
// [Embedder], and never from mocking the database: retrieval depends on real
// pgvector distance ordering and real PostgreSQL text search ranking, which a
// mock would reimplement wrongly.
package memphertest
