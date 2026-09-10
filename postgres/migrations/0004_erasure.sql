-- 0004_erasure.sql: the one way content leaves L0.
--
-- Rendered as a Go text/template before it runs, like every migration here, and
-- substituting nothing. It adds no table, no column and no index: it widens one
-- guard by exactly one hole.
--
-- 0001 said that removing a scope's history means dropping the append-only
-- trigger deliberately, "which is the amount of friction that decision
-- deserves". That was right about the friction and wrong about the mechanism.
-- Dropping the trigger unguards the table for every session for as long as it is
-- gone, including the ones appending while an erasure runs, and it needs
-- ownership of the table to do it. The friction should be attached to the
-- statement doing the erasing, not to the table it runs against.
--
-- So the trigger stays on for ever, and learns one exception: a DELETE is
-- permitted inside a transaction that has said, in that transaction, that it is
-- erasing. UPDATE and TRUNCATE are still refused, unconditionally and for ever.
-- That keeps the half of the invariant worth keeping -- a row that exists is
-- exactly what was recorded -- and gives up only the half that cannot survive a
-- right to erasure.
--
-- This is a guard against accident, not a permission boundary. Anything that
-- could set the flag could also have dropped the trigger; what it buys is that
-- no hotfix, no ORM, and no well-meaning cleanup script deletes from L0 without
-- saying so first, and that the saying is visible in the statement.
--
-- Two comments in earlier migrations are made stale by this one and cannot be
-- edited, because migrations are forward-only:
--
--   0001, on mempher.jobs.episode_ids, and 0002, on mempher.facts.episode_ids,
--   both say an array of episode ids needs no foreign key "because episodes are
--   never deleted". They now can be. Nothing is added here to enforce it,
--   because PostgreSQL still has no array foreign key: erasure removes those
--   rows itself, in the same transaction, and that is why it has to be one
--   operation rather than a DELETE anyone can write.

-- The guard, with its one exception.
--
-- current_setting's second argument makes an unset flag NULL rather than an
-- error, so the overwhelmingly common case -- nobody is erasing anything -- is
-- the cheap path and the refusing path.
CREATE OR REPLACE FUNCTION mempher.reject_l0_mutation() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    -- Statement-level BEFORE triggers ignore their return value, so this
    -- permits the statement by declining to raise.
    IF TG_OP = 'DELETE' AND current_setting('mempher.erasing', true) = 'on' THEN
        RETURN NULL;
    END IF;

    RAISE EXCEPTION
        'mempher.episodes is append-only: % is not permitted', TG_OP
        USING HINT =
            'append a correcting episode, rebuild the projection from L0, or '
            'erase through mempher.Memory.Forget, which declares itself';
END $$;

COMMENT ON FUNCTION mempher.reject_l0_mutation() IS
    'Guards the append-only invariant on mempher.episodes. A DELETE is permitted '
    'only inside a transaction that has SET LOCAL mempher.erasing = ''on''; '
    'UPDATE and TRUNCATE are never permitted.';
