-- 0003_job_introspection.sql: the indexes that make the queue answerable.
--
-- Rendered as a Go text/template before it runs, like every migration here. It
-- substitutes nothing: this file adds no column and no constraint, only the two
-- access paths that operating the queue needs.
--
-- Everything this stage exposes -- listing jobs, counting them, reviving a dead
-- one, pruning finished ones, and backfilling the work L0 implies -- reads data
-- that already exists. Nothing new is recorded, because nothing new is true:
-- what was missing was a way to ask.
--
-- Migrations run inside a transaction, so CONCURRENTLY is not available here.
-- Both indexes therefore take a lock on mempher.jobs while they build. That is
-- a pause in enqueueing and leasing, not in appending: mempher.episodes is not
-- touched, so the write path stays open throughout, and a queue drained to
-- anything near empty builds these in well under a second.

-- ---------------------------------------------------------------------------
-- Queue: operational access paths
-- ---------------------------------------------------------------------------

-- Finding work by what happened to it, across every scope.
--
-- This is the dead-job hunt, and it is the reason the whole stage exists: until
-- now the queue could be asked about a job whose id you already had, and nothing
-- would hand you the id of the job that died. Ordering by id is ordering by
-- creation time, because the ids are uuidv7, so one column serves both the
-- filter and the cursor that pages it.
--
-- It is a full index rather than one partial on 'dead'. A partial index would be
-- a fraction of the size and would leave a listing of pending or running work
-- with no path at all, which is the same blind spot moved somewhere quieter.
-- The cost of keeping it whole is one more index to maintain on every state
-- change, and a size that tracks the table's history -- which is what
-- mempher.jobs pruning is for.
CREATE INDEX jobs_state_idx
    ON mempher.jobs (state, id);

-- Finding work by whose memory it belongs to.
--
-- "Why has this user's memory not caught up?" is the question a queue is
-- actually asked, and answering it means every job of one scope regardless of
-- state, oldest first. State is deliberately not in this key: a scope holds few
-- enough jobs that scanning them beats maintaining a wider index, and the
-- selective half of the predicate is the scope.
CREATE INDEX jobs_scope_idx
    ON mempher.jobs (scope_id, id);

COMMENT ON INDEX mempher.jobs_state_idx IS
    'Lists jobs by lifecycle state across scopes; the dead-job hunt.';

COMMENT ON INDEX mempher.jobs_scope_idx IS
    'Lists one scope''s jobs in creation order, whatever state they are in.';
