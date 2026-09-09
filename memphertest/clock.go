package memphertest

import (
	"sync"
	"time"

	"github.com/mempher/mempher"
)

// Clock is a [mempher.Clock] that only moves when you move it, so episodes can
// sit years apart in system time without a test sleeping.
//
// It is safe for concurrent use, which matters because a worker under test runs
// on its own goroutine.
type Clock struct {
	mu  sync.Mutex
	now time.Time
}

// NewClock returns a Clock reading start.
func NewClock(start time.Time) *Clock {
	return &Clock{now: start}
}

// Now returns the current time. Repeated calls return the same instant, which is
// what makes an append's timestamps predictable.
func (c *Clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

// Advance moves the clock forward by d and returns the new time. A negative d
// moves it backwards.
func (c *Clock) Advance(d time.Duration) time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
	return c.now
}

// Set moves the clock to t.
func (c *Clock) Set(t time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = t
}

var _ mempher.Clock = (*Clock)(nil)
