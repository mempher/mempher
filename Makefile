# mempher development tasks.
#
# `make test` runs the fast suite with no Docker: every test that needs a
# database guards on testing.Short(). `make test-all` runs everything, with the
# race detector, against a real PostgreSQL 18 container.

GO      ?= go
PKGS    ?= ./...
COVER   ?= coverage.out

.DEFAULT_GOAL := help

.PHONY: help
help: ## List the available targets
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: test
test: ## Fast tests only, no Docker required
	$(GO) test -short $(PKGS)

.PHONY: test-all
test-all: ## Every test, with the race detector; needs Docker
	$(GO) test -race -count=1 $(PKGS)

.PHONY: cover
cover: ## Every test with a coverage profile, then report the total
	$(GO) test -race -count=1 -coverprofile=$(COVER) -covermode=atomic $(PKGS)
	@$(GO) tool cover -func=$(COVER) | tail -1

.PHONY: lint
lint: ## Run golangci-lint
	golangci-lint run $(PKGS)

.PHONY: fmt
fmt: ## Format and group imports
	gofmt -w .
	goimports -w .

.PHONY: vuln
vuln: ## Check dependencies for known vulnerabilities
	go run golang.org/x/vuln/cmd/govulncheck@v1.8.0 $(PKGS)

.PHONY: tidy
tidy: ## Tidy and verify the module
	$(GO) mod tidy
	$(GO) mod verify

.PHONY: check
check: fmt tidy lint test-all vuln ## Everything CI runs
