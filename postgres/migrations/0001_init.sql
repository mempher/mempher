-- 0001_init.sql: L0 (episodes), its vector projection, and the job queue.
--
-- Rendered as a Go text/template before it runs. Two deployment decisions are
-- substituted, both recorded in mempher.schema_config so drift is detectable
-- at startup rather than at query time:
--
--   {{.VectorDimensions}}   width of the embedding, from Embedder.Dimensions()
--   {{.TextSearchConfig}}   PostgreSQL text search configuration for FTS
--
-- The migration runner has already created the mempher schema and
-- mempher.schema_migrations, and holds an advisory lock. Migrations are
-- forward-only: this file never changes once applied anywhere.

-- uuidv7(), uuid_extract_timestamp() and the temporal constraints the next
-- stage needs are all PostgreSQL 18. Fail here, with a legible message, rather
-- than on a missing function three statements later.
DO $$
BEGIN
    IF current_setting('server_version_num')::int < 180000 THEN
        RAISE EXCEPTION
            'mempher requires PostgreSQL 18 or newer, found %',
            current_setting('server_version');
    END IF;
END $$;

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: the vector type and HNSW
CREATE EXTENSION IF NOT EXISTS btree_gin;   -- lets one GIN index cover scope_id + tsvector


-- ---------------------------------------------------------------------------
-- Configuration
-- ---------------------------------------------------------------------------

-- One row, enforced by the primary key. Startup compares these against the
-- configured Embedder: a model swapped for one of a different width is caught
-- as ErrEmbedderMismatch instead of failing on every insert.
CREATE TABLE mempher.schema_config (
    only_row           boolean     PRIMARY KEY DEFAULT true CHECK (only_row),
    vector_dimensions  int         NOT NULL CHECK (vector_dimensions BETWEEN 1 AND 2000),
    text_search_config text        NOT NULL,
    created_at         timestamptz NOT NULL DEFAULT now()
);

INSERT INTO mempher.schema_config (vector_dimensions, text_search_config)
VALUES ({{.VectorDimensions}}, '{{.TextSearchConfig}}');


-- ---------------------------------------------------------------------------
-- Scopes
-- ---------------------------------------------------------------------------

-- A scope is the unit of isolation, and this table exists to allocate its
-- episode sequence. Append upserts here and takes last_seq + 1 in the same
-- statement, so seq is dense and gap-free without a second round trip or an
-- explicit lock; concurrent appends to one scope serialise on this row, which
-- is the correct semantics for a totally ordered log.
CREATE TABLE mempher.scopes (
    id         text        NOT NULL,
    last_seq   bigint      NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL,

    CONSTRAINT scopes_pkey        PRIMARY KEY (id),
    CONSTRAINT scopes_id_len      CHECK (length(id) BETWEEN 1 AND 200),
    CONSTRAINT scopes_seq_nonneg  CHECK (last_seq >= 0)
);


-- ---------------------------------------------------------------------------
-- L0: episodes. Immutable, append-only, the only source of truth.
-- ---------------------------------------------------------------------------

-- Nothing in this library issues UPDATE or DELETE against this table. Every
-- other table here is a projection that can be dropped and rebuilt from it.
--
-- There is deliberately no vector column. An embedding cannot be produced on
-- the write path without a model call, and it cannot be added afterwards
-- without an UPDATE, so it lives in episode_encodings, keyed by model, where
-- re-embedding is an insert rather than a mutation of L0.
CREATE TABLE mempher.episodes (
    id          uuid        NOT NULL DEFAULT uuidv7(),
    scope_id    text        NOT NULL,
    seq         bigint      NOT NULL,
    content     text        NOT NULL,
    role        text        NOT NULL,
    actor       text        NOT NULL DEFAULT '',
    source      text        NOT NULL,

    -- occurred_at is event time: when the thing happened in the world.
    -- ingested_at is system time: when this library learned of it. Both come
    -- from the caller's injected Clock, never from the server, so a replay
    -- reproduces byte-identical rows. The id's own uuidv7 timestamp is
    -- separate: it is physical arrival time, from the database.
    occurred_at timestamptz NOT NULL,
    ingested_at timestamptz NOT NULL,

    binding     jsonb       NOT NULL DEFAULT '{}'::jsonb,

    -- Generated, so it exists the instant the row lands: an episode is
    -- lexically searchable with no worker involved. Immutable content means it
    -- never needs recomputing. The configuration is fixed at DDL time because
    -- a generated column requires an immutable expression.
    content_tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('{{.TextSearchConfig}}', content)) STORED,

    CONSTRAINT episodes_pkey          PRIMARY KEY (id),
    CONSTRAINT episodes_scope_fkey    FOREIGN KEY (scope_id) REFERENCES mempher.scopes (id),
    CONSTRAINT episodes_scope_seq_key UNIQUE (scope_id, seq),
    CONSTRAINT episodes_seq_positive  CHECK (seq > 0),
    CONSTRAINT episodes_content_len   CHECK (length(content) BETWEEN 1 AND 1048576),
    CONSTRAINT episodes_role_valid    CHECK (role IN ('user', 'assistant', 'system', 'tool', 'observation')),
    CONSTRAINT episodes_source_len    CHECK (length(source) BETWEEN 1 AND 100),
    CONSTRAINT episodes_actor_len     CHECK (length(actor) <= 200),
    -- Binding is flat string-to-string, and the schema says so as well as the
    -- Go type. A CHECK cannot contain a subquery, so this is asserted with
    -- jsonpath: strict mode matters, because lax mode unwraps arrays and would
    -- let {"a": ["x"]} pass as strings. Both functions are IMMUTABLE, which is
    -- what a CHECK requires.
    CONSTRAINT episodes_binding_flat  CHECK (
        jsonb_typeof(binding) = 'object'
        AND NOT jsonb_path_exists(binding, 'strict $.* ? (@.type() != "string")')
    ),
    CONSTRAINT episodes_binding_keys  CHECK (
        jsonb_array_length(jsonb_path_query_array(binding, 'strict $.keyvalue()')) <= 32
    )
);

-- Replay order for consolidation, and the as-of cut for the lexical channel.
CREATE INDEX episodes_scope_ingested_idx
    ON mempher.episodes (scope_id, ingested_at DESC, seq DESC);

-- Event-time windows.
CREATE INDEX episodes_scope_occurred_idx
    ON mempher.episodes (scope_id, occurred_at DESC);

-- Composite so scope selectivity is available inside the index rather than as a
-- filter over every lexical match in the database. This is what btree_gin buys.
CREATE INDEX episodes_scope_tsv_idx
    ON mempher.episodes USING gin (scope_id, content_tsv);

-- jsonb_path_ops is half the size of the default operator class and supports
-- @>, which is the only binding operator this library uses.
CREATE INDEX episodes_binding_idx
    ON mempher.episodes USING gin (binding jsonb_path_ops);


-- The invariant, enforced by the database rather than by discipline in Go.
--
-- L0 being the only source of truth is what makes a bad extractor recoverable
-- and forgetting safe, and it holds only if nothing ever rewrites history --
-- including this library, including a hotfix at 3am, including a well-meaning
-- UPDATE that looks like it only patches one field. A statement-level trigger
-- fires even when the statement would have matched no rows, so an attempt fails
-- loudly instead of silently succeeding against an empty set.
--
-- Corrections are appended. Removing a scope's history means dropping this
-- trigger deliberately, which is the amount of friction that decision deserves.
CREATE FUNCTION mempher.reject_l0_mutation() RETURNS trigger
    LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION
        'mempher.episodes is append-only: % is not permitted', TG_OP
        USING HINT = 'append a correcting episode, or rebuild the projection from L0';
END $$;

COMMENT ON FUNCTION mempher.reject_l0_mutation() IS
    'Guards the append-only invariant on mempher.episodes.';

CREATE TRIGGER episodes_append_only
    BEFORE UPDATE OR DELETE OR TRUNCATE ON mempher.episodes
    FOR EACH STATEMENT EXECUTE FUNCTION mempher.reject_l0_mutation();


-- ---------------------------------------------------------------------------
-- Projection: vector encodings
-- ---------------------------------------------------------------------------

-- Derived from L0 and disposable: TRUNCATE this table and workers rebuild it.
-- Keyed by model so several vector spaces coexist and changing embedder is a
-- backfill, not a migration.
--
-- scope_id and ingested_at are copied from the episode row rather than passed
-- in, by an INSERT that selects them from mempher.episodes. They are here
-- because an HNSW scan cannot filter on a joined table, and they cannot drift
-- because the row they came from is immutable.
CREATE TABLE mempher.episode_encodings (
    episode_id  uuid        NOT NULL,
    model       text        NOT NULL,
    scope_id    text        NOT NULL,
    ingested_at timestamptz NOT NULL,
    content_vec vector({{.VectorDimensions}}) NOT NULL,

    -- How novel the episode was against what the scope already held. Nothing
    -- writes it in this stage; forgetting will. It sits here rather than on
    -- episodes because it needs the vector, which does not exist at write time,
    -- and an episode is never updated.
    surprise    real        NOT NULL DEFAULT 0 CHECK (surprise >= 0 AND surprise <= 1),
    encoded_at  timestamptz NOT NULL,

    CONSTRAINT episode_encodings_pkey       PRIMARY KEY (episode_id, model),
    CONSTRAINT episode_encodings_ep_fkey    FOREIGN KEY (episode_id) REFERENCES mempher.episodes (id),
    CONSTRAINT episode_encodings_scope_fkey FOREIGN KEY (scope_id)   REFERENCES mempher.scopes (id),
    CONSTRAINT episode_encodings_model_len  CHECK (length(model) BETWEEN 1 AND 200)
);

CREATE INDEX episode_encodings_ann_idx
    ON mempher.episode_encodings USING hnsw (content_vec vector_cosine_ops);

-- Backs both the scope/as-of narrowing around an ANN scan and the anti-join
-- that finds episodes still lacking an encoding.
CREATE INDEX episode_encodings_scope_idx
    ON mempher.episode_encodings (model, scope_id, ingested_at DESC);


-- ---------------------------------------------------------------------------
-- Job queue
-- ---------------------------------------------------------------------------

-- Deferred work always names the episodes it derives from, so there is no
-- opaque payload column: L0 stays the only input to any projection, and no
-- jsonb blob is waiting to become an interface{}. PostgreSQL has no array
-- foreign key, which costs nothing here because episodes are never deleted.
CREATE TABLE mempher.jobs (
    id           uuid        NOT NULL DEFAULT uuidv7(),
    kind         text        NOT NULL,
    scope_id     text        NOT NULL,
    episode_ids  uuid[]      NOT NULL,
    state        text        NOT NULL DEFAULT 'pending',
    attempts     int         NOT NULL DEFAULT 0,
    max_attempts int         NOT NULL DEFAULT 5,
    run_after    timestamptz NOT NULL,
    leased_until timestamptz,
    leased_by    text,
    last_error   text,
    created_at   timestamptz NOT NULL,
    updated_at   timestamptz NOT NULL,

    CONSTRAINT jobs_pkey          PRIMARY KEY (id),
    CONSTRAINT jobs_scope_fkey    FOREIGN KEY (scope_id) REFERENCES mempher.scopes (id),
    CONSTRAINT jobs_kind_valid    CHECK (kind IN ('encode')),
    CONSTRAINT jobs_state_valid   CHECK (state IN ('pending', 'running', 'done', 'dead')),
    CONSTRAINT jobs_episodes_set  CHECK (cardinality(episode_ids) > 0),
    CONSTRAINT jobs_attempts_ok   CHECK (attempts >= 0 AND max_attempts > 0 AND attempts <= max_attempts),
    -- A lease exists exactly while the job is running, so a crashed worker
    -- cannot leave a job that looks claimable and claimed at once.
    CONSTRAINT jobs_lease_ok      CHECK (
        (state = 'running') = (leased_until IS NOT NULL AND leased_by IS NOT NULL)
    )
);

-- The lease query: ready jobs of a kind, oldest first, taken with
-- FOR UPDATE SKIP LOCKED.
CREATE INDEX jobs_ready_idx
    ON mempher.jobs (kind, run_after, id) WHERE state = 'pending';

-- Reclaiming leases a dead worker still holds.
CREATE INDEX jobs_expiring_idx
    ON mempher.jobs (leased_until) WHERE state = 'running';

-- Enqueueing the same work twice while it is still outstanding collapses onto
-- the existing job. Done and dead rows are excluded so the same episode can be
-- re-encoded later, after a model change.
CREATE UNIQUE INDEX jobs_inflight_key
    ON mempher.jobs (kind, scope_id, episode_ids) WHERE state IN ('pending', 'running');
