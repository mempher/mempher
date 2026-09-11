# Security policy

## Reporting a vulnerability

Report it privately, through
[GitHub's advisory form](https://github.com/mempher/mempher/security/advisories/new).
Please do not open a public issue for one.

Include the version, what an attacker gets, and the smallest way to reproduce
it. A proof of concept is welcome and never required.

mempher is maintained by one person, so please allow a few days for a first
reply.

## Supported versions

While mempher is v0.x, fixes go to the latest release only.

## Scope

In scope: this library's Go code and the SQL it embeds. The failures worth
reporting are anything that lets data cross a `Scope`, anything that modifies or
removes an episode outside `Forget`, and anything that gets a query past the
parameterisation.

Out of scope: your own `Embedder` and `Extractor`, your database's configuration
and network exposure, and PostgreSQL or pgvector themselves — please report those
upstream.
