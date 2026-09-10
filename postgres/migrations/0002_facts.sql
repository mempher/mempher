-- 0002_facts.sql: L1 (facts), the extraction markers, and the extract job kind.
--
-- Rendered as a Go text/template before it runs, with the same substitutions as
-- 0001; only {{.TextSearchConfig}} is used here.
--
-- L1 is a projection of L0 and nothing more. Every row in this file can be
-- truncated and rebuilt by replaying mempher.episodes through the same
-- Extractor, which is the only repair a bad extractor has and the reason none of
-- these tables is protected the way mempher.episodes is.

-- Temporal primary keys (PERIOD ... WITHOUT OVERLAPS) need GiST operator classes
-- for the scalar parts of the key, which btree_gist supplies. It is created here
-- rather than in 0001 because nothing before this migration needed it.
CREATE EXTENSION IF NOT EXISTS btree_gist;


-- ---------------------------------------------------------------------------
-- L1: facts. Derived from L0, disposable, and bounded in event time.
-- ---------------------------------------------------------------------------

-- A fact is a triple with a rendered statement beside it, and the two halves do
-- different jobs. The triple is identity: it is what makes the same claim
-- extracted twice one row rather than two, and what a retraction names. The
-- statement is the payload: it is what goes in front of a model, and the only
-- column the text index sees.
--
-- valid is event time -- when the claim held in the world. asserted_at is system
-- time -- when this library learned it. Keeping them apart is the whole point of
-- the table: "lived in Paris until March" and "we found this out on Tuesday" are
-- different questions, and a store that conflates them answers the first with
-- the second.
--
-- What is deliberately NOT here is belief history: the row says what is
-- currently believed about the past, not what was believed at some past instant.
-- Retraction moves valid in place rather than closing one row and opening
-- another. That keeps the temporal key exact -- WITHOUT OVERLAPS cannot be made
-- partial, so superseded rows in this table would collide with live ones -- and
-- it is honest about which layer is reproducible: L0 is, a projection is
-- current. Full bitemporality is an additive migration if it is ever wanted.
CREATE TABLE mempher.facts (
    -- A surrogate key beside the real one. The temporal primary key below moves
    -- when a window is closed, so a retraction needs something that does not:
    -- this is the handle an Extractor is given and hands back.
    id          uuid        NOT NULL DEFAULT uuidv7(),

    scope_id    text        NOT NULL,

    -- Facts are keyed by extractor for the same reason encodings are keyed by
    -- model: two extractors' opinions must not silently merge into one scope,
    -- and changing extractor must be a backfill rather than a migration.
    extractor   text        NOT NULL,

    subject     text        NOT NULL,
    predicate   text        NOT NULL,
    object      text        NOT NULL,
    statement   text        NOT NULL,

    -- Half-open, [from, to): a fact ending at noon and one starting at noon do
    -- not overlap, so a change of state needs no gap and no tie-break. An
    -- unbounded upper end means still true, which is the common case.
    valid       tstzrange   NOT NULL,

    confidence  real        NOT NULL DEFAULT 1 CHECK (confidence >= 0 AND confidence <= 1),

    -- Provenance: the episodes this claim was read from. PostgreSQL has no array
    -- foreign key, which costs nothing because episodes are never deleted. It is
    -- the first thing asked of a fact that looks wrong.
    episode_ids uuid[]      NOT NULL,

    asserted_at timestamptz NOT NULL,
    updated_at  timestamptz NOT NULL,

    -- Generated, like episodes.content_tsv, so a fact is searchable the instant
    -- it lands and no second projection hangs off this one. Only the statement
    -- is indexed: the triple is machine identity, and matching a query against
    -- "allergic_to" would rank on a slug the caller never wrote.
    statement_tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('{{.TextSearchConfig}}', statement)) STORED,

    -- The identity of a fact is what it claims and when it held. WITHOUT
    -- OVERLAPS makes the database reject a scope that believes the same triple
    -- over two overlapping windows -- the exact corruption an extractor
    -- re-reading an episode would otherwise cause -- instead of leaving recall
    -- to return the claim twice.
    --
    -- object is part of the key so that being allergic to two things is two
    -- facts rather than one overwriting the other. Contradiction between
    -- different objects of one predicate is not a constraint violation and
    -- cannot be: only the extractor knows whether "lives_in Berlin" replaces
    -- "lives_in Paris" or joins it, so it says so with a retraction.
    CONSTRAINT facts_pkey PRIMARY KEY
        (scope_id, extractor, subject, predicate, object, valid WITHOUT OVERLAPS),

    CONSTRAINT facts_id_key         UNIQUE (id),
    CONSTRAINT facts_scope_fkey     FOREIGN KEY (scope_id) REFERENCES mempher.scopes (id),

    -- These lengths are not arbitrary. Everything in the primary key above ends
    -- up in one GiST index tuple, which has a hard size limit, so the five key
    -- columns are sized to fit inside it together with the range and its
    -- overhead. Mirrors the limits in the Go package.
    CONSTRAINT facts_extractor_len  CHECK (length(extractor) BETWEEN 1 AND 200),
    CONSTRAINT facts_subject_len    CHECK (length(subject)   BETWEEN 1 AND 200),
    CONSTRAINT facts_predicate_len  CHECK (length(predicate) BETWEEN 1 AND 100),
    CONSTRAINT facts_object_len     CHECK (length(object)    BETWEEN 1 AND 1000),
    CONSTRAINT facts_statement_len  CHECK (length(statement) BETWEEN 1 AND 4000),

    -- A predicate is the key that makes two extractions of one claim the same
    -- claim, so it may not carry whitespace: "lives_in" and "lives in" must not
    -- become two relations.
    CONSTRAINT facts_predicate_slug CHECK (predicate !~ '\s'),

    CONSTRAINT facts_episodes_set   CHECK (cardinality(episode_ids) > 0),

    -- A window must begin. An unbounded lower end would place the claim before
    -- every episode that could have produced it, and empty holds at no instant
    -- at all, which is a mistake rather than a way to say "never true".
    CONSTRAINT facts_valid_bounded  CHECK (NOT lower_inf(valid) AND NOT isempty(valid)),
    -- '[)' is asserted, not assumed, so a caller building the range by hand
    -- cannot introduce an inclusive upper bound that makes two adjacent windows
    -- overlap by exactly one instant.
    CONSTRAINT facts_valid_halfopen CHECK (
        lower_inc(valid) AND (upper_inf(valid) OR NOT upper_inc(valid))
    )
);

-- The read path: everything believed about one scope, by one extractor, valid at
-- an instant. The GiST primary key can serve this, but a btree on the equality
-- prefix answers the common query without touching it.
CREATE INDEX facts_scope_lookup_idx
    ON mempher.facts (scope_id, extractor, subject, predicate);

-- The system-time cut, which bounds facts to those already learned.
CREATE INDEX facts_scope_asserted_idx
    ON mempher.facts (scope_id, extractor, asserted_at DESC);

-- Relevance ordering when a scope holds more facts than the budget fits.
-- Composite for the same reason as episodes_scope_tsv_idx: scope selectivity
-- belongs inside the index, not as a filter over every match in the database.
CREATE INDEX facts_scope_tsv_idx
    ON mempher.facts USING gin (scope_id, statement_tsv);


-- ---------------------------------------------------------------------------
-- Projection bookkeeping: which episodes an extractor has read
-- ---------------------------------------------------------------------------

-- Most episodes yield no durable fact, so "has this been extracted?" cannot be
-- answered from mempher.facts: an episode with no facts and an episode never
-- read look identical there. This table is the difference, and without it a
-- scope full of small talk is re-extracted, at model prices, on every pass.
--
-- scope_id and ingested_at are copied from the episode row by an INSERT that
-- selects them, exactly as in mempher.episode_encodings, and for the same two
-- reasons: they make the pending anti-join a single-table scan, and they cannot
-- drift because the row they came from is immutable.
CREATE TABLE mempher.fact_extractions (
    episode_id   uuid        NOT NULL,
    extractor    text        NOT NULL,
    scope_id     text        NOT NULL,
    ingested_at  timestamptz NOT NULL,
    -- How many facts the extraction that read this episode asserted, including
    -- zero. Cheap to record and the first number to look at when an extractor is
    -- suspected of finding something durable in every "ok, thanks". It counts
    -- the extraction, not the episode, so a job covering several episodes
    -- records the same total against each.
    fact_count   int         NOT NULL DEFAULT 0 CHECK (fact_count >= 0),
    extracted_at timestamptz NOT NULL,

    CONSTRAINT fact_extractions_pkey       PRIMARY KEY (episode_id, extractor),
    CONSTRAINT fact_extractions_ep_fkey    FOREIGN KEY (episode_id) REFERENCES mempher.episodes (id),
    CONSTRAINT fact_extractions_scope_fkey FOREIGN KEY (scope_id)   REFERENCES mempher.scopes (id),
    CONSTRAINT fact_extractions_extr_len   CHECK (length(extractor) BETWEEN 1 AND 200)
);

-- Backs the anti-join that finds episodes an extractor has not read yet.
CREATE INDEX fact_extractions_scope_idx
    ON mempher.fact_extractions (extractor, scope_id, ingested_at DESC);


-- ---------------------------------------------------------------------------
-- Queue: the extract job kind
-- ---------------------------------------------------------------------------

-- The kind set is closed by a CHECK, so adding one is a forward migration. That
-- is the intended cost: it means a binary from before this migration cannot
-- enqueue work a binary from before this migration cannot drain.
--
-- 'extract' is separate from 'encode' rather than one consolidating kind because
-- the two projections are backfilled independently -- an embedding change reruns
-- every encode and no extraction, a prompt fix reruns the reverse -- and one
-- kind would tie a cheap backfill to an expensive one.
ALTER TABLE mempher.jobs DROP CONSTRAINT jobs_kind_valid;
ALTER TABLE mempher.jobs ADD CONSTRAINT jobs_kind_valid
    CHECK (kind IN ('encode', 'extract'));
