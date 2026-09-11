package ops

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/mempher/mempher"
)

func TestJobQueryValidate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		query   JobQuery
		wantErr error
	}{
		{name: "the zero query lists everything"},
		{
			name: "every filter set",
			query: JobQuery{
				Scope:  "user:1",
				Kinds:  []mempher.JobKind{mempher.JobKindEncode, mempher.JobKindExtract},
				States: []mempher.JobState{mempher.JobStateDead},
				Limit:  10,
			},
		},
		{
			name:    "unknown kind",
			query:   JobQuery{Kinds: []mempher.JobKind{"consolidate"}},
			wantErr: mempher.ErrInvalidJobKind,
		},
		{
			name:    "unknown state",
			query:   JobQuery{States: []mempher.JobState{"stuck"}},
			wantErr: mempher.ErrInvalidJobState,
		},
		{
			name:    "over-long scope",
			query:   JobQuery{Scope: mempher.ScopeID(strings.Repeat("s", 300))},
			wantErr: mempher.ErrInvalidScope,
		},
		{name: "negative limit", query: JobQuery{Limit: -1}, wantErr: mempher.ErrInvalidConfig},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assertValidation(t, tc.query.Validate(), tc.wantErr)
		})
	}
}

// TestPurgeRequestValidate is the guard on the one DELETE in this library: it
// may take history, never outstanding work.
func TestPurgeRequestValidate(t *testing.T) {
	t.Parallel()

	horizon := time.Unix(1_700_000_000, 0).UTC()
	tests := []struct {
		name    string
		req     PurgeRequest
		wantErr error
	}{
		{name: "a horizon and nothing else", req: PurgeRequest{Before: horizon}},
		{
			name: "both finished states named",
			req:  PurgeRequest{Before: horizon, States: []mempher.JobState{mempher.JobStateDone, mempher.JobStateDead}},
		},
		{name: "no horizon", req: PurgeRequest{}, wantErr: mempher.ErrInvalidConfig},
		{
			name:    "pending work is not history",
			req:     PurgeRequest{Before: horizon, States: []mempher.JobState{mempher.JobStatePending}},
			wantErr: mempher.ErrInvalidJobState,
		},
		{
			name:    "running work is not history",
			req:     PurgeRequest{Before: horizon, States: []mempher.JobState{mempher.JobStateRunning}},
			wantErr: mempher.ErrInvalidJobState,
		},
		{
			name:    "unknown state",
			req:     PurgeRequest{Before: horizon, States: []mempher.JobState{"archived"}},
			wantErr: mempher.ErrInvalidJobState,
		},
		{
			name:    "negative limit",
			req:     PurgeRequest{Before: horizon, Limit: -1},
			wantErr: mempher.ErrInvalidConfig,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assertValidation(t, tc.req.Validate(), tc.wantErr)
		})
	}
}

func TestScopeQueryValidate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		query   ScopeQuery
		wantErr error
	}{
		{name: "the zero query lists every scope"},
		{name: "a prefix and a cursor", query: ScopeQuery{Prefix: "user:", After: "user:1", Limit: 50}},
		{
			name:    "over-long prefix",
			query:   ScopeQuery{Prefix: mempher.ScopeID(strings.Repeat("p", 300))},
			wantErr: mempher.ErrInvalidScope,
		},
		{
			name:    "over-long cursor",
			query:   ScopeQuery{After: mempher.ScopeID(strings.Repeat("a", 300))},
			wantErr: mempher.ErrInvalidScope,
		},
		{name: "negative limit", query: ScopeQuery{Limit: -1}, wantErr: mempher.ErrInvalidConfig},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assertValidation(t, tc.query.Validate(), tc.wantErr)
		})
	}
}

func TestBackfillRequestValidate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		req     BackfillRequest
		wantErr error
	}{
		{name: "the zero request backfills everything"},
		{
			name: "resuming one scope",
			req: BackfillRequest{
				Scope:     "user:1",
				Kinds:     []mempher.JobKind{mempher.JobKindEncode},
				AfterSeq:  120,
				BatchSize: 8,
				Limit:     500,
			},
		},
		{
			name:    "unknown kind",
			req:     BackfillRequest{Kinds: []mempher.JobKind{"consolidate"}},
			wantErr: mempher.ErrInvalidJobKind,
		},
		{
			// Seq is only ordered within a scope, so a cursor without one
			// would resume in a place that does not exist.
			name:    "a cursor with no scope",
			req:     BackfillRequest{AfterSeq: 4},
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "negative cursor",
			req:     BackfillRequest{Scope: "user:1", AfterSeq: -1},
			wantErr: mempher.ErrInvalidConfig,
		},
		{name: "negative batch size", req: BackfillRequest{BatchSize: -1}, wantErr: mempher.ErrInvalidConfig},
		{name: "negative limit", req: BackfillRequest{Limit: -1}, wantErr: mempher.ErrInvalidConfig},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assertValidation(t, tc.req.Validate(), tc.wantErr)
		})
	}
}

// assertValidation is the check every table above ends with.
func assertValidation(t *testing.T, err, want error) {
	t.Helper()
	switch {
	case want == nil && err != nil:
		t.Fatalf("unexpected error: %v", err)
	case want != nil && !errors.Is(err, want):
		t.Fatalf("err = %v, want %v", err, want)
	}
}
