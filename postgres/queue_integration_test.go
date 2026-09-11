package postgres_test

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mempher/mempher"
	"github.com/mempher/mempher/ops"
	"github.com/mempher/mempher/postgres"
)

// seedScope appends one episode so a scope exists to enqueue against, and returns
// its id.
func seedScope(t *testing.T, store *postgres.Store, scope mempher.ScopeID) mempher.EpisodeID {
	t.Helper()
	got, err := store.Append(t.Context(), episode(scope, "an episode", epoch))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	return got.Episodes[0].ID
}

func newJob(scope mempher.ScopeID, id mempher.EpisodeID) mempher.NewJob {
	return mempher.NewJob{
		Kind:     mempher.JobKindEncode,
		Scope:    scope,
		Episodes: []mempher.EpisodeID{id},
	}
}

func lease(worker mempher.WorkerID, at time.Time, limit int) mempher.LeaseRequest {
	return mempher.LeaseRequest{
		Worker:   worker,
		Now:      at,
		Limit:    limit,
		Duration: time.Minute,
	}
}

func TestEnqueue(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	// Append already enqueued one encode job for this episode, so a fresh
	// episode is needed to test Enqueue on its own.
	seedScope(t, store, "user:1")
	second, err := store.Append(ctx, func() mempher.AppendCommand {
		cmd := episode("user:1", "no job implied", epoch)
		cmd.Jobs = nil
		return cmd
	}())
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	id := second.Episodes[0].ID

	job, err := store.Enqueue(ctx, newJob("user:1", id), epoch)
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	switch {
	case job.ID.IsZero():
		t.Error("no job id was returned")
	case job.State != mempher.JobStatePending:
		t.Errorf("State = %q, want pending", job.State)
	case job.Attempts != 0:
		t.Errorf("Attempts = %d, want 0", job.Attempts)
	case job.MaxAttempts != mempher.DefaultMaxAttempts:
		t.Errorf("MaxAttempts = %d, want %d", job.MaxAttempts, mempher.DefaultMaxAttempts)
	case !job.RunAfter.Equal(epoch):
		t.Errorf("RunAfter = %s, want %s (zero means now)", job.RunAfter, epoch)
	case len(job.Episodes) != 1 || job.Episodes[0] != id:
		t.Errorf("Episodes = %v, want [%s]", job.Episodes, id)
	case job.LeasedBy != "":
		t.Errorf("LeasedBy = %q on a pending job", job.LeasedBy)
	case !job.LeasedUntil.IsZero():
		t.Errorf("LeasedUntil = %s on a pending job", job.LeasedUntil)
	}

	t.Run("outstanding work is not queued twice", func(t *testing.T) {
		again, err := store.Enqueue(ctx, newJob("user:1", id), epoch.Add(time.Minute))
		if err != nil {
			t.Fatalf("second Enqueue: %v", err)
		}
		if again.ID != job.ID {
			t.Errorf("got a new job %s, want the outstanding %s", again.ID, job.ID)
		}
	})

	t.Run("but the same work can be queued again once it is done", func(t *testing.T) {
		leased, err := store.Lease(ctx, lease("w1", epoch, 10))
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		for _, l := range leased {
			if err := store.Succeed(ctx, l.ID, "w1", epoch); err != nil {
				t.Fatalf("Succeed: %v", err)
			}
		}
		fresh, err := store.Enqueue(ctx, newJob("user:1", id), epoch.Add(time.Hour))
		if err != nil {
			t.Fatalf("Enqueue after done: %v", err)
		}
		if fresh.ID == job.ID {
			t.Error("a finished job was reused; a re-encode must be a new job")
		}
	})
}

func TestEnqueueRejectsBadJobs(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	id := seedScope(t, store, "user:1")

	tests := []struct {
		name    string
		job     mempher.NewJob
		now     time.Time
		wantErr error
	}{
		{
			name:    "no scope",
			job:     mempher.NewJob{Kind: mempher.JobKindEncode, Episodes: []mempher.EpisodeID{id}},
			now:     epoch,
			wantErr: mempher.ErrInvalidScope,
		},
		{
			name:    "unknown kind",
			job:     mempher.NewJob{Kind: "nonesuch", Scope: "user:1", Episodes: []mempher.EpisodeID{id}},
			now:     epoch,
			wantErr: mempher.ErrInvalidJobKind,
		},
		{
			name:    "no episodes",
			job:     mempher.NewJob{Kind: mempher.JobKindEncode, Scope: "user:1"},
			now:     epoch,
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name: "a zero episode id",
			job: mempher.NewJob{
				Kind: mempher.JobKindEncode, Scope: "user:1",
				Episodes: []mempher.EpisodeID{{}},
			},
			now:     epoch,
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "unresolved now",
			job:     newJob("user:1", id),
			wantErr: mempher.ErrInvalidConfig,
		},
		{
			name:    "a scope with no episodes",
			job:     newJob("user:nobody", id),
			now:     epoch,
			wantErr: mempher.ErrNotFound,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if _, err := store.Enqueue(t.Context(), tc.job, tc.now); !errors.Is(err, tc.wantErr) {
				t.Fatalf("err = %v, want %v", err, tc.wantErr)
			}
		})
	}
}

func TestLease(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	// Three episodes, each with its encode job from Append.
	for i := range 3 {
		if _, err := store.Append(ctx,
			episode("user:1", fmt.Sprintf("episode %d", i), epoch)); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}

	t.Run("claims up to the limit and counts the attempt", func(t *testing.T) {
		leased, err := store.Lease(ctx, lease("w1", epoch, 2))
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		if len(leased) != 2 {
			t.Fatalf("leased %d jobs, want 2", len(leased))
		}
		for _, job := range leased {
			switch {
			case job.State != mempher.JobStateRunning:
				t.Errorf("State = %q, want running", job.State)
			case job.Attempts != 1:
				t.Errorf("Attempts = %d, want 1 (counted at claim time)", job.Attempts)
			case job.LeasedBy != "w1":
				t.Errorf("LeasedBy = %q, want w1", job.LeasedBy)
			case !job.LeasedUntil.Equal(epoch.Add(time.Minute)):
				t.Errorf("LeasedUntil = %s, want %s", job.LeasedUntil, epoch.Add(time.Minute))
			}
		}
	})

	t.Run("a second worker gets only what is left", func(t *testing.T) {
		leased, err := store.Lease(ctx, lease("w2", epoch, 10))
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		if len(leased) != 1 {
			t.Errorf("leased %d, want the 1 remaining", len(leased))
		}
		empty, err := store.Lease(ctx, lease("w3", epoch, 10))
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		if len(empty) != 0 {
			t.Errorf("leased %d from an empty queue", len(empty))
		}
	})
}

func TestLeaseHonoursRunAfterAndKind(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	id := seedScope(t, store, "user:1")
	// Append already queued one job at epoch; retime it into the future.
	pending, err := store.Lease(ctx, lease("setup", epoch, 10))
	if err != nil {
		t.Fatalf("Lease: %v", err)
	}
	later := epoch.Add(time.Hour)
	for _, job := range pending {
		if err := store.Fail(ctx, job.ID, "setup", errors.New("deliberate"), epoch, later); err != nil {
			t.Fatalf("Fail: %v", err)
		}
	}

	t.Run("a job is invisible before its run_after", func(t *testing.T) {
		leased, err := store.Lease(ctx, lease("w1", epoch.Add(time.Minute), 10))
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		if len(leased) != 0 {
			t.Errorf("leased %d jobs before run_after", len(leased))
		}
	})

	t.Run("and claimable after it", func(t *testing.T) {
		leased, err := store.Lease(ctx, lease("w1", later, 10))
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		if len(leased) != 1 {
			t.Errorf("leased %d jobs after run_after, want 1", len(leased))
		}
		if leased[0].Attempts != 2 {
			t.Errorf("Attempts = %d, want 2 after one failure", leased[0].Attempts)
		}
		if err := store.Succeed(ctx, leased[0].ID, "w1", later); err != nil {
			t.Fatalf("Succeed: %v", err)
		}
	})

	t.Run("a kind filter excludes other kinds", func(t *testing.T) {
		if _, err := store.Enqueue(ctx, newJob("user:1", id), later); err != nil {
			t.Fatalf("Enqueue: %v", err)
		}
		req := lease("w1", later.Add(time.Hour), 10)
		req.Kinds = []mempher.JobKind{mempher.JobKindEncode}
		leased, err := store.Lease(ctx, req)
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		if len(leased) != 1 {
			t.Errorf("leased %d encode jobs, want 1", len(leased))
		}
	})
}

// TestLeaseIsExclusive is why SKIP LOCKED is there: many workers, one queue, no
// job handed out twice.
func TestLeaseIsExclusive(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	const jobs = 12
	for i := range jobs {
		if _, err := store.Append(ctx,
			episode("user:1", fmt.Sprintf("episode %d", i), epoch)); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}

	const workers = 4
	var (
		wg    sync.WaitGroup
		mu    sync.Mutex
		claim = make(map[mempher.JobID]mempher.WorkerID)
		total int
	)
	start := make(chan struct{})
	for w := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			worker := mempher.WorkerID(fmt.Sprintf("w%d", w))
			<-start
			for {
				leased, err := store.Lease(ctx, lease(worker, epoch, 3))
				if err != nil {
					t.Errorf("%s Lease: %v", worker, err)
					return
				}
				if len(leased) == 0 {
					return
				}
				mu.Lock()
				for _, job := range leased {
					if other, dup := claim[job.ID]; dup {
						t.Errorf("job %s leased by both %s and %s", job.ID, other, worker)
					}
					claim[job.ID] = worker
					total++
				}
				mu.Unlock()
			}
		}()
	}
	close(start)
	wg.Wait()

	if total != jobs {
		t.Errorf("%d jobs claimed in total, want %d", total, jobs)
	}
}

func TestSucceedAndFail(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	seedScope(t, store, "user:1")

	leased, err := store.Lease(ctx, lease("w1", epoch, 1))
	if err != nil || len(leased) != 1 {
		t.Fatalf("Lease: %v (%d jobs)", err, len(leased))
	}
	job := leased[0]

	t.Run("a worker that does not hold the lease is refused", func(t *testing.T) {
		err := store.Succeed(ctx, job.ID, "impostor", epoch)
		if !errors.Is(err, mempher.ErrJobNotLeased) {
			t.Errorf("Succeed err = %v, want mempher.ErrJobNotLeased", err)
		}
		err = store.Fail(ctx, job.ID, "impostor", errors.New("x"), epoch, epoch)
		if !errors.Is(err, mempher.ErrJobNotLeased) {
			t.Errorf("Fail err = %v, want mempher.ErrJobNotLeased", err)
		}
	})

	t.Run("an unknown job is not found", func(t *testing.T) {
		if err := store.Succeed(ctx, mempher.JobID{9, 9}, "w1", epoch); !errors.Is(err, mempher.ErrNotFound) {
			t.Errorf("err = %v, want mempher.ErrNotFound", err)
		}
	})

	t.Run("succeeding clears the lease and keeps the row", func(t *testing.T) {
		if err := store.Succeed(ctx, job.ID, "w1", epoch); err != nil {
			t.Fatalf("Succeed: %v", err)
		}
		done, err := store.Job(ctx, job.ID)
		if err != nil {
			t.Fatalf("Job: %v", err)
		}
		switch {
		case done.State != mempher.JobStateDone:
			t.Errorf("State = %q, want done", done.State)
		case done.LeasedBy != "":
			t.Errorf("LeasedBy = %q, want empty", done.LeasedBy)
		case !done.LeasedUntil.IsZero():
			t.Errorf("LeasedUntil = %s, want zero", done.LeasedUntil)
		case done.Attempts != 1:
			t.Errorf("Attempts = %d, want 1", done.Attempts)
		}
	})

	t.Run("succeeding twice is refused", func(t *testing.T) {
		if err := store.Succeed(ctx, job.ID, "w1", epoch); !errors.Is(err, mempher.ErrJobNotLeased) {
			t.Errorf("err = %v, want mempher.ErrJobNotLeased", err)
		}
	})
}

// TestFailRetriesThenDies walks a job through every attempt, which is the path
// that keeps a broken extractor from retrying for ever.
func TestFailRetriesThenDies(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	id := seedScope(t, store, "user:2")
	// Replace Append's job with one that dies after two attempts.
	first, err := store.Lease(ctx, lease("setup", epoch, 10))
	if err != nil {
		t.Fatalf("Lease: %v", err)
	}
	for _, job := range first {
		if err := store.Succeed(ctx, job.ID, "setup", epoch); err != nil {
			t.Fatalf("Succeed: %v", err)
		}
	}
	spec := newJob("user:2", id)
	spec.MaxAttempts = 2
	queued, err := store.Enqueue(ctx, spec, epoch)
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}

	boom := errors.New("the embedder is down")
	at := epoch
	for attempt := 1; attempt <= 2; attempt++ {
		at = at.Add(time.Minute)
		leased, err := store.Lease(ctx, lease("w1", at, 10))
		if err != nil {
			t.Fatalf("Lease attempt %d: %v", attempt, err)
		}
		if len(leased) != 1 {
			t.Fatalf("attempt %d leased %d jobs, want 1", attempt, len(leased))
		}
		if leased[0].Attempts != attempt {
			t.Errorf("attempt %d reports Attempts %d", attempt, leased[0].Attempts)
		}
		if err := store.Fail(ctx, queued.ID, "w1", boom, at, at.Add(time.Second)); err != nil {
			t.Fatalf("Fail attempt %d: %v", attempt, err)
		}

		after, err := store.Job(ctx, queued.ID)
		if err != nil {
			t.Fatalf("Job: %v", err)
		}
		wantState := mempher.JobStatePending
		if attempt == 2 {
			wantState = mempher.JobStateDead
		}
		if after.State != wantState {
			t.Errorf("after attempt %d state = %q, want %q", attempt, after.State, wantState)
		}
		if after.LastError == "" {
			t.Error("the failure was not recorded")
		}
	}

	t.Run("a dead job is never leased again", func(t *testing.T) {
		leased, err := store.Lease(ctx, lease("w1", at.Add(time.Hour), 10))
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		for _, job := range leased {
			if job.ID == queued.ID {
				t.Error("a dead job was leased")
			}
		}
	})
}

// TestReclaim covers the crash path: a worker takes a job and never comes back.
func TestReclaim(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	seedScope(t, store, "user:1")

	leased, err := store.Lease(ctx, lease("doomed", epoch, 1))
	if err != nil || len(leased) != 1 {
		t.Fatalf("Lease: %v (%d)", err, len(leased))
	}
	job := leased[0]

	t.Run("a live lease is left alone", func(t *testing.T) {
		n, err := store.Reclaim(ctx, epoch.Add(30*time.Second))
		if err != nil {
			t.Fatalf("Reclaim: %v", err)
		}
		if n != 0 {
			t.Errorf("reclaimed %d live leases, want 0", n)
		}
	})

	t.Run("an expired lease returns to pending", func(t *testing.T) {
		expired := epoch.Add(2 * time.Minute)
		n, err := store.Reclaim(ctx, expired)
		if err != nil {
			t.Fatalf("Reclaim: %v", err)
		}
		if n != 1 {
			t.Fatalf("reclaimed %d, want 1", n)
		}
		back, err := store.Job(ctx, job.ID)
		if err != nil {
			t.Fatalf("Job: %v", err)
		}
		switch {
		case back.State != mempher.JobStatePending:
			t.Errorf("State = %q, want pending", back.State)
		case back.LeasedBy != "":
			t.Errorf("LeasedBy = %q, want empty", back.LeasedBy)
		case back.Attempts != 1:
			t.Errorf("Attempts = %d, want the attempt still counted", back.Attempts)
		case back.LastError == "":
			t.Error("the expiry was not recorded")
		}

		// And it can be claimed again.
		again, err := store.Lease(ctx, lease("w2", expired, 1))
		if err != nil {
			t.Fatalf("Lease: %v", err)
		}
		if len(again) != 1 || again[0].ID != job.ID {
			t.Errorf("the reclaimed job was not re-leasable: %v", again)
		}
	})

	t.Run("unresolved now is refused", func(t *testing.T) {
		if _, err := store.Reclaim(ctx, time.Time{}); !errors.Is(err, mempher.ErrInvalidConfig) {
			t.Errorf("err = %v, want mempher.ErrInvalidConfig", err)
		}
	})
}

// TestReclaimExhaustedJobDies is the case a naive reclaim gets wrong. Attempts
// are counted at lease time, so returning an exhausted job to pending would leave
// a row the next lease could not legally claim.
func TestReclaimExhaustedJobDies(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	id := seedScope(t, store, "user:3")
	for _, job := range mustLease(t, store, lease("setup", epoch, 10)) {
		if err := store.Succeed(ctx, job.ID, "setup", epoch); err != nil {
			t.Fatalf("Succeed: %v", err)
		}
	}
	spec := newJob("user:3", id)
	spec.MaxAttempts = 1
	queued, err := store.Enqueue(ctx, spec, epoch)
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}

	// The only attempt is claimed, then the worker vanishes.
	if got := mustLease(t, store, lease("doomed", epoch, 1)); len(got) != 1 {
		t.Fatalf("leased %d, want 1", len(got))
	}
	expired := epoch.Add(2 * time.Minute)
	if _, err := store.Reclaim(ctx, expired); err != nil {
		t.Fatalf("Reclaim: %v", err)
	}

	after, err := store.Job(ctx, queued.ID)
	if err != nil {
		t.Fatalf("Job: %v", err)
	}
	if after.State != mempher.JobStateDead {
		t.Errorf("State = %q, want dead: an exhausted job must not go back to pending",
			after.State)
	}
	// The real proof: leasing again must neither return it nor error.
	leased, err := store.Lease(ctx, lease("w1", expired, 10))
	if err != nil {
		t.Fatalf("Lease after reclaim: %v", err)
	}
	for _, job := range leased {
		if job.ID == queued.ID {
			t.Error("an exhausted job was leased again")
		}
	}
}

func mustLease(t *testing.T, store *postgres.Store, req mempher.LeaseRequest) []mempher.Job {
	t.Helper()
	leased, err := store.Lease(t.Context(), req)
	if err != nil {
		t.Fatalf("Lease: %v", err)
	}
	return leased
}

func TestJobLookup(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	seedScope(t, store, "user:1")

	leased := mustLease(t, store, lease("w1", epoch, 1))
	if len(leased) != 1 {
		t.Fatalf("leased %d, want 1", len(leased))
	}

	got, err := store.Job(ctx, leased[0].ID)
	if err != nil {
		t.Fatalf("Job: %v", err)
	}
	if got.ID != leased[0].ID || got.Kind != mempher.JobKindEncode {
		t.Errorf("Job returned %+v", got)
	}

	if _, err := store.Job(ctx, mempher.JobID{}); !errors.Is(err, mempher.ErrNotFound) {
		t.Errorf("zero id err = %v, want mempher.ErrNotFound", err)
	}
	if _, err := store.Job(ctx, mempher.JobID{7, 7, 7}); !errors.Is(err, mempher.ErrNotFound) {
		t.Errorf("unknown id err = %v, want mempher.ErrNotFound", err)
	}
}

// TestFailTruncatesEnormousErrors keeps a stack trace from becoming a row.
func TestFailTruncatesEnormousErrors(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()
	seedScope(t, store, "user:1")

	leased := mustLease(t, store, lease("w1", epoch, 1))
	huge := errors.New(strings.Repeat("x", 50_000))
	if err := store.Fail(ctx, leased[0].ID, "w1", huge, epoch, epoch.Add(time.Second)); err != nil {
		t.Fatalf("Fail: %v", err)
	}
	got, err := store.Job(ctx, leased[0].ID)
	if err != nil {
		t.Fatalf("Job: %v", err)
	}
	if len(got.LastError) > 3000 {
		t.Errorf("LastError is %d bytes, want it bounded", len(got.LastError))
	}
	if got.LastError == "" {
		t.Error("the failure was not recorded at all")
	}
}

// jobless appends an episode that implies no work, so a test can enqueue exactly
// the job it means to and lease exactly that one.
func jobless(t *testing.T, store *postgres.Store, scope mempher.ScopeID) mempher.EpisodeID {
	t.Helper()
	cmd := episode(scope, "no job implied", epoch)
	cmd.Jobs = nil
	got, err := store.Append(t.Context(), cmd)
	if err != nil {
		t.Fatalf("Append to %q: %v", scope, err)
	}
	return got.Episodes[0].ID
}

// early is a run_after before any job an Append enqueues, so that a job a test
// means to lease sorts to the front of the queue and Lease(1) claims that one
// and no other. Leasing is ordered by run_after and then id.
var early = epoch.Add(-time.Hour)

// leaseTheOldest claims exactly one job and insists it is the expected one, so a
// test manufacturing a job state cannot quietly operate on a different job.
func leaseTheOldest(
	t *testing.T,
	store *postgres.Store,
	worker mempher.WorkerID,
	want mempher.JobID,
) mempher.Job {
	t.Helper()
	leased := mustLease(t, store, lease(worker, epoch, 1))
	if len(leased) != 1 || leased[0].ID != want {
		t.Fatalf("leased %d jobs, want exactly job %s", len(leased), want)
	}
	return leased[0]
}

// enqueueLeasable adds one job that will sort to the front of the queue, so the
// next Lease(1) claims it.
func enqueueLeasable(
	t *testing.T,
	store *postgres.Store,
	scope mempher.ScopeID,
	maxAttempts int,
) mempher.Job {
	t.Helper()
	id := jobless(t, store, scope)
	job, err := store.Enqueue(t.Context(), mempher.NewJob{
		Kind:        mempher.JobKindEncode,
		Scope:       scope,
		Episodes:    []mempher.EpisodeID{id},
		RunAfter:    early,
		MaxAttempts: maxAttempts,
	}, epoch)
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	return job
}

// killJob enqueues one job with a single attempt and spends it, which is the
// only way a job reaches the dead state.
func killJob(t *testing.T, store *postgres.Store, scope mempher.ScopeID) mempher.Job {
	t.Helper()
	ctx := t.Context()
	job := enqueueLeasable(t, store, scope, 1)
	leaseTheOldest(t, store, "doomed", job.ID)
	if err := store.Fail(ctx, job.ID, "doomed", errors.New("the model was on fire"),
		epoch, epoch.Add(time.Second)); err != nil {
		t.Fatalf("Fail: %v", err)
	}
	dead, err := store.Job(ctx, job.ID)
	if err != nil {
		t.Fatalf("Job: %v", err)
	}
	if dead.State != mempher.JobStateDead {
		t.Fatalf("State = %q after exhausting its attempts, want dead", dead.State)
	}
	return dead
}

func TestJobsListsAndFilters(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	// One encode job each, enqueued by Append itself.
	first := seedScope(t, store, "user:1")
	seedScope(t, store, "user:2")
	// And one extract job, so the kind filter has something to exclude.
	if _, err := store.Enqueue(ctx, mempher.NewJob{
		Kind:     mempher.JobKindExtract,
		Scope:    "user:1",
		Episodes: []mempher.EpisodeID{first},
	}, epoch); err != nil {
		t.Fatalf("Enqueue extract: %v", err)
	}

	t.Run("everything", func(t *testing.T) {
		got, err := store.Jobs(ctx, ops.JobQuery{})
		if err != nil {
			t.Fatalf("Jobs: %v", err)
		}
		if len(got) != 3 {
			t.Fatalf("Jobs returned %d, want 3", len(got))
		}
		// Ordered by id, which is uuidv7 and therefore creation order.
		for i := 1; i < len(got); i++ {
			if got[i-1].ID.String() >= got[i].ID.String() {
				t.Errorf("job %d is not ordered after job %d", i, i-1)
			}
		}
	})

	t.Run("by scope", func(t *testing.T) {
		got, err := store.Jobs(ctx, ops.JobQuery{Scope: "user:2"})
		if err != nil {
			t.Fatalf("Jobs: %v", err)
		}
		if len(got) != 1 || got[0].Scope != "user:2" {
			t.Errorf("Jobs for user:2 returned %d jobs, want 1", len(got))
		}
	})

	t.Run("by kind", func(t *testing.T) {
		got, err := store.Jobs(ctx, ops.JobQuery{Kinds: []mempher.JobKind{mempher.JobKindExtract}})
		if err != nil {
			t.Fatalf("Jobs: %v", err)
		}
		if len(got) != 1 || got[0].Kind != mempher.JobKindExtract {
			t.Errorf("Jobs of kind extract returned %d jobs, want 1", len(got))
		}
	})

	t.Run("by state", func(t *testing.T) {
		pending, err := store.Jobs(ctx, ops.JobQuery{
			States: []mempher.JobState{mempher.JobStatePending},
		})
		if err != nil {
			t.Fatalf("Jobs: %v", err)
		}
		if len(pending) != 3 {
			t.Errorf("pending jobs = %d, want 3", len(pending))
		}
		dead, err := store.Jobs(ctx, ops.JobQuery{
			States: []mempher.JobState{mempher.JobStateDead},
		})
		if err != nil {
			t.Fatalf("Jobs: %v", err)
		}
		if len(dead) != 0 {
			t.Errorf("dead jobs = %d, want none yet", len(dead))
		}
	})

	t.Run("pages by After", func(t *testing.T) {
		page, err := store.Jobs(ctx, ops.JobQuery{Limit: 2})
		if err != nil {
			t.Fatalf("Jobs: %v", err)
		}
		if len(page) != 2 {
			t.Fatalf("first page = %d jobs, want 2", len(page))
		}
		rest, err := store.Jobs(ctx, ops.JobQuery{After: page[1].ID})
		if err != nil {
			t.Fatalf("Jobs: %v", err)
		}
		if len(rest) != 1 {
			t.Errorf("second page = %d jobs, want 1", len(rest))
		}
	})
}

func TestJobsRejectsBadQueries(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	if _, err := store.Jobs(ctx, ops.JobQuery{
		Kinds: []mempher.JobKind{"consolidate"},
	}); !errors.Is(err, mempher.ErrInvalidJobKind) {
		t.Errorf("unknown kind err = %v, want mempher.ErrInvalidJobKind", err)
	}
	if _, err := store.Jobs(ctx, ops.JobQuery{
		States: []mempher.JobState{"stuck"},
	}); !errors.Is(err, mempher.ErrInvalidJobState) {
		t.Errorf("unknown state err = %v, want mempher.ErrInvalidJobState", err)
	}
	if _, err := store.Jobs(ctx, ops.JobQuery{Limit: -1}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("negative limit err = %v, want mempher.ErrInvalidConfig", err)
	}
}

func TestJobStatsCountsByKindAndState(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	seedScope(t, store, "user:1")
	seedScope(t, store, "user:2")
	dead := killJob(t, store, "user:3")

	got, err := store.Stats(ctx, "")
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	buckets := map[string]ops.JobCount{}
	for _, count := range got {
		buckets[string(count.Kind)+"/"+string(count.State)] = count
	}
	if pending := buckets["encode/pending"]; pending.Jobs != 2 {
		t.Errorf("encode/pending = %d jobs, want 2", pending.Jobs)
	}
	if got := buckets["encode/dead"]; got.Jobs != 1 {
		t.Errorf("encode/dead = %d jobs, want 1", got.Jobs)
	}
	// Depth without age cannot tell a busy queue from a stalled one.
	if oldest := buckets["encode/pending"].Oldest; !oldest.Equal(epoch) {
		t.Errorf("oldest pending = %s, want %s", oldest, epoch)
	}
	if _, reported := buckets["extract/pending"]; reported {
		t.Error("an empty bucket was reported, which the port says it never is")
	}

	// A scope narrows it to that scope's work.
	scoped, err := store.Stats(ctx, dead.Scope)
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if len(scoped) != 1 || scoped[0].State != mempher.JobStateDead || scoped[0].Jobs != 1 {
		t.Errorf("Stats for %q = %+v, want one dead job", dead.Scope, scoped)
	}
}

func TestRetryRevivesADeadJob(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	dead := killJob(t, store, "user:1")
	later := epoch.Add(time.Hour)

	revived, err := store.Retry(ctx, dead.ID, later)
	if err != nil {
		t.Fatalf("Retry: %v", err)
	}
	switch {
	case revived.State != mempher.JobStatePending:
		t.Errorf("State = %q, want pending", revived.State)
	case revived.Attempts != 0:
		t.Errorf("Attempts = %d, want them reset to 0", revived.Attempts)
	case !revived.RunAfter.Equal(later):
		t.Errorf("RunAfter = %s, want %s", revived.RunAfter, later)
	case revived.LeasedBy != "":
		t.Errorf("LeasedBy = %q on a revived job", revived.LeasedBy)
	case revived.LastError == "":
		t.Error("LastError was cleared; the row no longer says what went wrong")
	}

	// And it is claimable again, which is the whole point.
	leased := mustLease(t, store, lease("w1", later, 1))
	if len(leased) != 1 || leased[0].ID != dead.ID {
		t.Errorf("leased %d jobs, want the revived one back", len(leased))
	}
}

func TestRetryRejectsWhatIsNotDead(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	seedScope(t, store, "user:1")
	pending, err := store.Jobs(ctx, ops.JobQuery{})
	if err != nil || len(pending) != 1 {
		t.Fatalf("Jobs = %d, %v; want the one job Append enqueued", len(pending), err)
	}

	if _, err := store.Retry(ctx, pending[0].ID, epoch); !errors.Is(err, mempher.ErrJobNotDead) {
		t.Errorf("retrying a pending job err = %v, want mempher.ErrJobNotDead", err)
	}
	if _, err := store.Retry(ctx, mempher.JobID{9, 9, 9}, epoch); !errors.Is(err, mempher.ErrNotFound) {
		t.Errorf("retrying an unknown job err = %v, want mempher.ErrNotFound", err)
	}
	if _, err := store.Retry(ctx, pending[0].ID, time.Time{}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("retrying with no clock err = %v, want mempher.ErrInvalidConfig", err)
	}
}

// TestRetryRefusesWorkAlreadyQueued covers the partial unique index: a dead job
// whose work has since been enqueued again cannot be revived onto it.
func TestRetryRefusesWorkAlreadyQueued(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	dead := killJob(t, store, "user:1")
	if _, err := store.Enqueue(ctx, mempher.NewJob{
		Kind:     dead.Kind,
		Scope:    dead.Scope,
		Episodes: dead.Episodes,
	}, epoch); err != nil {
		t.Fatalf("Enqueue the same work again: %v", err)
	}

	if _, err := store.Retry(ctx, dead.ID, epoch); !errors.Is(err, mempher.ErrJobOutstanding) {
		t.Errorf("Retry err = %v, want mempher.ErrJobOutstanding", err)
	}
	// And the dead row is still dead, rather than half-revived.
	still, err := store.Job(ctx, dead.ID)
	if err != nil {
		t.Fatalf("Job: %v", err)
	}
	if still.State != mempher.JobStateDead {
		t.Errorf("State = %q after a refused retry, want dead", still.State)
	}
}

func TestPurgeDeletesOnlyFinishedJobs(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	// One job in each state. Each is leased straight after it is enqueued,
	// while it is the only claimable job, and the pending one is created last
	// so that nothing is ever ambiguous about which job a Lease returns.
	done := enqueueLeasable(t, store, "user:done", 0)
	leaseTheOldest(t, store, "w2", done.ID)
	if err := store.Succeed(ctx, done.ID, "w2", epoch); err != nil {
		t.Fatalf("Succeed: %v", err)
	}

	killJob(t, store, "user:dead")

	running := enqueueLeasable(t, store, "user:running", 0)
	leaseTheOldest(t, store, "w1", running.ID)

	seedScope(t, store, "user:pending")

	horizon := epoch.Add(time.Hour)
	deleted, err := store.Purge(ctx, ops.PurgeRequest{Before: horizon})
	if err != nil {
		t.Fatalf("Purge: %v", err)
	}
	if deleted != 2 {
		t.Errorf("Purge deleted %d rows, want the done one and the dead one", deleted)
	}

	left, err := store.Jobs(ctx, ops.JobQuery{})
	if err != nil {
		t.Fatalf("Jobs: %v", err)
	}
	for _, job := range left {
		if job.State == mempher.JobStateDone || job.State == mempher.JobStateDead {
			t.Errorf("job %s survived the purge in state %s", job.ID, job.State)
		}
	}
	// Outstanding work is untouched, whatever its age.
	if len(left) != 2 {
		t.Errorf("%d jobs left, want the pending one and the running one", len(left))
	}
}

func TestPurgeRespectsItsHorizonAndLimit(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	for _, scope := range []mempher.ScopeID{"a", "b", "c"} {
		killJob(t, store, scope)
	}

	// Nothing is older than the epoch itself, so nothing goes.
	deleted, err := store.Purge(ctx, ops.PurgeRequest{Before: epoch})
	if err != nil {
		t.Fatalf("Purge: %v", err)
	}
	if deleted != 0 {
		t.Errorf("Purge deleted %d rows before the horizon, want 0", deleted)
	}

	// Bounded, so a deep history is a loop of short transactions.
	deleted, err = store.Purge(ctx, ops.PurgeRequest{Before: epoch.Add(time.Hour), Limit: 2})
	if err != nil {
		t.Fatalf("Purge: %v", err)
	}
	if deleted != 2 {
		t.Errorf("Purge with Limit 2 deleted %d rows, want 2", deleted)
	}
}

func TestPurgeRefusesOutstandingWork(t *testing.T) {
	t.Parallel()
	store, _ := migrated(t)
	ctx := t.Context()

	for _, state := range []mempher.JobState{mempher.JobStatePending, mempher.JobStateRunning} {
		_, err := store.Purge(ctx, ops.PurgeRequest{
			Before: epoch.Add(time.Hour),
			States: []mempher.JobState{state},
		})
		if !errors.Is(err, mempher.ErrInvalidJobState) {
			t.Errorf("purging %s jobs err = %v, want mempher.ErrInvalidJobState", state, err)
		}
	}
	if _, err := store.Purge(ctx, ops.PurgeRequest{}); !errors.Is(err, mempher.ErrInvalidConfig) {
		t.Errorf("purge with no horizon err = %v, want mempher.ErrInvalidConfig", err)
	}
}
