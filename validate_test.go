package mempher

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestAppendRequestValidate(t *testing.T) {
	t.Parallel()

	valid := func(mutate func(*AppendRequest)) AppendRequest {
		r := AppendRequest{
			Scope:   "user:1",
			Content: "I am allergic to hazelnuts.",
			Role:    RoleUser,
			Source:  "cli",
		}
		if mutate != nil {
			mutate(&r)
		}
		return r
	}

	tests := []struct {
		name    string
		req     AppendRequest
		wantErr error
	}{
		{name: "minimal valid request", req: valid(nil)},
		{
			name: "every optional field set",
			req: valid(func(r *AppendRequest) {
				r.Actor = "user:8123"
				r.OccurredAt = time.Unix(0, 0).UTC()
				r.Binding = Binding{"session": "s-42", "task": "shopping"}
			}),
		},
		{name: "empty scope", req: valid(func(r *AppendRequest) { r.Scope = "" }), wantErr: ErrInvalidScope},
		{
			name:    "over-long scope",
			req:     valid(func(r *AppendRequest) { r.Scope = ScopeID(strings.Repeat("s", MaxScopeIDLen+1)) }),
			wantErr: ErrInvalidScope,
		},
		{
			name:    "scope at the limit is fine",
			req:     valid(func(r *AppendRequest) { r.Scope = ScopeID(strings.Repeat("s", MaxScopeIDLen)) }),
			wantErr: nil,
		},
		{
			name:    "scope with a newline",
			req:     valid(func(r *AppendRequest) { r.Scope = "user:1\nadmin" }),
			wantErr: ErrInvalidScope,
		},
		{
			name:    "scope that is not UTF-8",
			req:     valid(func(r *AppendRequest) { r.Scope = ScopeID([]byte{0xff, 0xfe}) }),
			wantErr: ErrInvalidScope,
		},
		{name: "empty content", req: valid(func(r *AppendRequest) { r.Content = "" }), wantErr: ErrInvalidContent},
		{
			name:    "over-long content",
			req:     valid(func(r *AppendRequest) { r.Content = strings.Repeat("x", MaxContentLen+1) }),
			wantErr: ErrInvalidContent,
		},
		{
			name:    "content that is not UTF-8",
			req:     valid(func(r *AppendRequest) { r.Content = string([]byte{0xff}) }),
			wantErr: ErrInvalidContent,
		},
		{name: "unset role", req: valid(func(r *AppendRequest) { r.Role = "" }), wantErr: ErrInvalidRole},
		{
			name:    "invented role",
			req:     valid(func(r *AppendRequest) { r.Role = Role("wizard") }),
			wantErr: ErrInvalidRole,
		},
		{name: "empty source", req: valid(func(r *AppendRequest) { r.Source = "" }), wantErr: ErrInvalidSource},
		{
			name:    "over-long source",
			req:     valid(func(r *AppendRequest) { r.Source = Source(strings.Repeat("s", MaxSourceLen+1)) }),
			wantErr: ErrInvalidSource,
		},
		{
			name:    "over-long actor",
			req:     valid(func(r *AppendRequest) { r.Actor = Actor(strings.Repeat("a", MaxActorLen+1)) }),
			wantErr: ErrInvalidActor,
		},
		{name: "empty actor is allowed", req: valid(func(r *AppendRequest) { r.Actor = "" })},
		{
			name: "too many binding keys",
			req: valid(func(r *AppendRequest) {
				r.Binding = Binding{}
				for i := range MaxBindingKeys + 1 {
					r.Binding[string(rune('a'+i%26))+string(rune('0'+i/26))] = "v"
				}
			}),
			wantErr: ErrInvalidBinding,
		},
		{
			name:    "empty binding key",
			req:     valid(func(r *AppendRequest) { r.Binding = Binding{"": "v"} }),
			wantErr: ErrInvalidBinding,
		},
		{
			name: "over-long binding key",
			req: valid(func(r *AppendRequest) {
				r.Binding = Binding{strings.Repeat("k", MaxBindingKeyLen+1): "v"}
			}),
			wantErr: ErrInvalidBinding,
		},
		{
			name: "over-long binding value",
			req: valid(func(r *AppendRequest) {
				r.Binding = Binding{"k": strings.Repeat("v", MaxBindingValueLen+1)}
			}),
			wantErr: ErrInvalidBinding,
		},
		{name: "nil binding is allowed", req: valid(func(r *AppendRequest) { r.Binding = nil })},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := tc.req.Validate()
			switch {
			case tc.wantErr == nil && err != nil:
				t.Fatalf("unexpected error: %v", err)
			case tc.wantErr != nil && !errors.Is(err, tc.wantErr):
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestRecallRequestValidate(t *testing.T) {
	t.Parallel()

	epoch := time.Unix(0, 0).UTC()
	valid := func(mutate func(*RecallRequest)) RecallRequest {
		r := RecallRequest{Scope: "user:1", Query: "food allergies"}
		if mutate != nil {
			mutate(&r)
		}
		return r
	}

	tests := []struct {
		name    string
		req     RecallRequest
		wantErr error
	}{
		{name: "minimal valid request", req: valid(nil)},
		{
			name: "every optional field set",
			req: valid(func(r *RecallRequest) {
				r.AsOf = epoch
				r.OccurredFrom, r.OccurredTo = epoch, epoch.Add(time.Hour)
				r.Binding = Binding{"session": "s-42"}
				r.Roles = []Role{RoleUser, RoleAssistant}
				r.Channels = []Channel{ChannelSemantic, ChannelLexical}
				r.CandidatesPerChannel, r.Limit, r.MaxTokens = 100, 5, 2000
			}),
		},
		{name: "empty scope", req: valid(func(r *RecallRequest) { r.Scope = "" }), wantErr: ErrInvalidScope},
		{name: "empty query", req: valid(func(r *RecallRequest) { r.Query = "" }), wantErr: ErrInvalidContent},
		{
			name:    "whitespace-only query",
			req:     valid(func(r *RecallRequest) { r.Query = "  \t\n " }),
			wantErr: ErrInvalidContent,
		},
		{
			name:    "unknown channel",
			req:     valid(func(r *RecallRequest) { r.Channels = []Channel{Channel("telepathy")} }),
			wantErr: ErrInvalidChannel,
		},
		{
			name:    "unknown role filter",
			req:     valid(func(r *RecallRequest) { r.Roles = []Role{Role("wizard")} }),
			wantErr: ErrInvalidRole,
		},
		{name: "negative limit", req: valid(func(r *RecallRequest) { r.Limit = -1 }), wantErr: ErrInvalidConfig},
		{
			name:    "negative candidates per channel",
			req:     valid(func(r *RecallRequest) { r.CandidatesPerChannel = -1 }),
			wantErr: ErrInvalidConfig,
		},
		{
			name:    "negative token budget",
			req:     valid(func(r *RecallRequest) { r.MaxTokens = -1 }),
			wantErr: ErrInvalidConfig,
		},
		{
			name: "inverted event-time window",
			req: valid(func(r *RecallRequest) {
				r.OccurredFrom, r.OccurredTo = epoch.Add(time.Hour), epoch
			}),
			wantErr: ErrInvalidConfig,
		},
		{
			name: "half-open window is fine",
			req:  valid(func(r *RecallRequest) { r.OccurredFrom = epoch }),
		},
		{
			name: "an instantaneous window is fine",
			req: valid(func(r *RecallRequest) {
				r.OccurredFrom, r.OccurredTo = epoch, epoch
			}),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := tc.req.Validate()
			switch {
			case tc.wantErr == nil && err != nil:
				t.Fatalf("unexpected error: %v", err)
			case tc.wantErr != nil && !errors.Is(err, tc.wantErr):
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestEnumsRejectUnknownValues(t *testing.T) {
	t.Parallel()

	t.Run("role", func(t *testing.T) {
		t.Parallel()
		for _, r := range []Role{RoleUser, RoleAssistant, RoleSystem, RoleTool, RoleObservation} {
			if !r.Valid() {
				t.Errorf("%q should be valid", r)
			}
			var got Role
			if err := got.UnmarshalText([]byte(r)); err != nil || got != r {
				t.Errorf("round trip %q: got %q, err %v", r, got, err)
			}
		}
		for _, bad := range []string{"", "wizard", "User", "USER", " user"} {
			var got Role
			if err := got.UnmarshalText([]byte(bad)); !errors.Is(err, ErrInvalidRole) {
				t.Errorf("UnmarshalText(%q) err = %v, want ErrInvalidRole", bad, err)
			}
		}
		if _, err := Role("wizard").MarshalText(); !errors.Is(err, ErrInvalidRole) {
			t.Errorf("MarshalText of an invalid role should fail, got %v", err)
		}
	})

	t.Run("channel and job enums", func(t *testing.T) {
		t.Parallel()
		if Channel("telepathy").Valid() || Channel("").Valid() {
			t.Error("unknown channels should be invalid")
		}
		if JobKind("nonesuch").Valid() || JobKind("").Valid() {
			t.Error("unknown job kinds should be invalid")
		}
		if JobState("zombie").Valid() || JobState("").Valid() {
			t.Error("unknown job states should be invalid")
		}
	})
}

func TestEpisodeIDRoundTrip(t *testing.T) {
	t.Parallel()

	var zero EpisodeID
	if !zero.IsZero() {
		t.Error("the zero EpisodeID should report IsZero")
	}

	const text = "018f3a2b-6c1d-7e2f-8a9b-0c1d2e3f4a5b"
	id, err := ParseEpisodeID(text)
	switch {
	case err != nil:
		t.Fatalf("ParseEpisodeID: %v", err)
	case id.IsZero():
		t.Fatal("a parsed id should not be zero")
	case id.String() != text:
		t.Errorf("String() = %q, want %q", id.String(), text)
	}

	var round EpisodeID
	if err := round.UnmarshalText([]byte(text)); err != nil || round != id {
		t.Errorf("text round trip: got %v, err %v", round, err)
	}
	if _, err := ParseEpisodeID("not-a-uuid"); err == nil {
		t.Error("ParseEpisodeID should reject a non-UUID")
	}
}

func TestApproxTokenCounter(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		per  int
		text string
		want int
	}{
		{name: "empty text costs nothing", text: "", want: 0},
		{name: "one byte still costs a token", text: "a", want: 1},
		{name: "rounds up", text: "abcde", want: 2},
		{name: "exact multiple", text: "abcdefgh", want: 2},
		{name: "custom divisor", per: 2, text: "abcde", want: 3},
		{name: "non-positive divisor falls back", per: -3, text: "abcd", want: 1},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := ApproxTokenCounter{BytesPerToken: tc.per}.CountTokens(tc.text)
			if got != tc.want {
				t.Errorf("CountTokens(%q) = %d, want %d", tc.text, got, tc.want)
			}
		})
	}
}

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
				Kinds:  []JobKind{JobKindEncode, JobKindExtract},
				States: []JobState{JobStateDead},
				Limit:  10,
			},
		},
		{
			name:    "unknown kind",
			query:   JobQuery{Kinds: []JobKind{"consolidate"}},
			wantErr: ErrInvalidJobKind,
		},
		{
			name:    "unknown state",
			query:   JobQuery{States: []JobState{"stuck"}},
			wantErr: ErrInvalidJobState,
		},
		{
			name:    "over-long scope",
			query:   JobQuery{Scope: ScopeID(strings.Repeat("s", 300))},
			wantErr: ErrInvalidScope,
		},
		{name: "negative limit", query: JobQuery{Limit: -1}, wantErr: ErrInvalidConfig},
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
			req:  PurgeRequest{Before: horizon, States: []JobState{JobStateDone, JobStateDead}},
		},
		{name: "no horizon", req: PurgeRequest{}, wantErr: ErrInvalidConfig},
		{
			name:    "pending work is not history",
			req:     PurgeRequest{Before: horizon, States: []JobState{JobStatePending}},
			wantErr: ErrInvalidJobState,
		},
		{
			name:    "running work is not history",
			req:     PurgeRequest{Before: horizon, States: []JobState{JobStateRunning}},
			wantErr: ErrInvalidJobState,
		},
		{
			name:    "unknown state",
			req:     PurgeRequest{Before: horizon, States: []JobState{"archived"}},
			wantErr: ErrInvalidJobState,
		},
		{
			name:    "negative limit",
			req:     PurgeRequest{Before: horizon, Limit: -1},
			wantErr: ErrInvalidConfig,
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
			query:   ScopeQuery{Prefix: ScopeID(strings.Repeat("p", 300))},
			wantErr: ErrInvalidScope,
		},
		{
			name:    "over-long cursor",
			query:   ScopeQuery{After: ScopeID(strings.Repeat("a", 300))},
			wantErr: ErrInvalidScope,
		},
		{name: "negative limit", query: ScopeQuery{Limit: -1}, wantErr: ErrInvalidConfig},
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
				Kinds:     []JobKind{JobKindEncode},
				AfterSeq:  120,
				BatchSize: 8,
				Limit:     500,
			},
		},
		{
			name:    "unknown kind",
			req:     BackfillRequest{Kinds: []JobKind{"consolidate"}},
			wantErr: ErrInvalidJobKind,
		},
		{
			// Seq is only ordered within a scope, so a cursor without one
			// would resume in a place that does not exist.
			name:    "a cursor with no scope",
			req:     BackfillRequest{AfterSeq: 4},
			wantErr: ErrInvalidConfig,
		},
		{
			name:    "negative cursor",
			req:     BackfillRequest{Scope: "user:1", AfterSeq: -1},
			wantErr: ErrInvalidConfig,
		},
		{name: "negative batch size", req: BackfillRequest{BatchSize: -1}, wantErr: ErrInvalidConfig},
		{name: "negative limit", req: BackfillRequest{Limit: -1}, wantErr: ErrInvalidConfig},
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
