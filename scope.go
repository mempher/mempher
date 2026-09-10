// The catalogue of L0 partitions.

package mempher

import "time"

// Scope is one partition of L0 as the catalogue holds it: its identity, how far
// its log has run, and when it was first written to.
//
// A scope row exists from the moment its first episode is appended and is never
// removed, because the episodes referring to it are never removed either. It is
// the only place the length of a log can be read without counting it.
type Scope struct {
	// ID is the partition key every episode, job and fact carries.
	ID ScopeID
	// LastSeq is the Seq of the newest episode in the scope, and therefore
	// how many episodes it holds: Seq starts at 1 and is dense.
	LastSeq int64
	// CreatedAt is when the scope's first episode was appended, from the
	// [Clock] of whoever appended it.
	CreatedAt time.Time
}

// ScopeQuery lists the partitions of L0, in id order.
//
// It exists for operating the library rather than for serving a request: a
// backfill has to walk every scope, and until something can enumerate them there
// is no way to reach a scope whose id nobody wrote down.
type ScopeQuery struct {
	// Prefix, when non-empty, keeps only scopes whose id starts with it,
	// which is how a multi-tenant deployment lists one tenant's scopes. It
	// is a filter and not a promise about indexes: whether it can seek the
	// primary key depends on the database's collation.
	Prefix ScopeID
	// After pages the listing: only scopes ordered after it are returned.
	// Empty starts at the beginning.
	After ScopeID
	// Limit is the most scopes to return. Zero means a store-chosen default.
	Limit int
}
