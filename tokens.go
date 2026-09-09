// Token accounting for the read loop budget.

package mempher

// TokenCounter estimates the token cost of text, so [Memory.Recall] can cut
// results to a budget the caller's model actually has. Optional: without one,
// recall uses [ApproxTokenCounter].
//
// Implementations must be safe for concurrent use.
type TokenCounter interface {
	// CountTokens returns the estimated token cost of text.
	CountTokens(text string) int
}

// DefaultBytesPerToken is the divisor [ApproxTokenCounter] uses when its own is
// unset. Four bytes per token is the usual rule of thumb for English prose.
const DefaultBytesPerToken = 4

// ApproxTokenCounter estimates token cost by dividing byte length, without
// running a tokenizer. Its zero value is ready to use.
//
// It exists so the token budget has a sane default, not so you can rely on it:
// the estimate over-counts text with long words and under-counts scripts whose
// characters cost three UTF-8 bytes but one token. If the budget must be exact,
// inject your model's real tokenizer.
type ApproxTokenCounter struct {
	// BytesPerToken is the divisor. Zero means [DefaultBytesPerToken].
	BytesPerToken int
}

// CountTokens returns the estimated token cost of text, rounded up. Non-empty
// text always costs at least one token.
func (c ApproxTokenCounter) CountTokens(text string) int {
	if text == "" {
		return 0
	}
	per := c.BytesPerToken
	if per <= 0 {
		per = DefaultBytesPerToken
	}
	return (len(text) + per - 1) / per
}
