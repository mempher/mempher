// Expected conditions, reported as wrapped sentinels.

package extract

import "errors"

// Test these with [errors.Is]. A configuration this package refuses wraps
// [mempher.ErrInvalidConfig] instead, so that a caller assembling a Memory, a
// Worker and an Extractor has one sentinel to check rather than three.
var (
	// ErrMalformedResponse means the model returned something that is not
	// JSON, or is JSON of the wrong shape.
	//
	// It is returned plainly rather than retried here. A [mempher.Worker]
	// already backs off and retries, and a provider that keeps answering this
	// way should dead-letter where an operator can see it rather than spin
	// inside one job.
	ErrMalformedResponse = errors.New("malformed extraction response")

	// ErrPromptTooLarge means the episodes of one job hold more content than
	// [Options.MaxPromptBytes] allows.
	//
	// It fails the job rather than sending a shortened prompt, and that is the
	// important half. An extraction marks every episode the job named as read,
	// whether or not it yielded anything, so an extractor that quietly dropped
	// the tail would leave those episodes marked extracted and never
	// extracted. Failing costs nothing that was not already lost, and is
	// visible in the dead letters rather than only in a token bill.
	ErrPromptTooLarge = errors.New("extraction prompt is too large")
)
