// The only source of time in this package.

package mempher

import "time"

// Clock is the only source of time in this package. Injecting it is what makes
// as-of behaviour testable: a test clock can place episodes years apart without
// a test sleeping, and no [Store] method reads a clock of its own.
//
// Implementations must be safe for concurrent use.
type Clock interface {
	// Now returns the current time.
	Now() time.Time
}

// SystemClock reads the operating system clock, and is the default when a
// [Config] leaves Clock nil. Its zero value is ready to use.
type SystemClock struct{}

// Now returns the current system time.
func (SystemClock) Now() time.Time { return time.Now() }
