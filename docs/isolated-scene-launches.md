# Isolated scene launch verification

Isolation materializes committed HEAD into a Crossby-owned detached worktree.
It does not grant a security boundary: Git metadata, normal tool environment,
credentials and global configuration remain shared or accessible according to
the tool's existing permissions. See [the user guide](../README.md#scenes).

## Native contracts

Codex 0.162.1 was checked with a disposable CODEX_HOME containing harmless local
MCP definitions, using only app-server initialize/config-read requests and no
model turn. The actual adapter output was passed to the server. A quoted override
key such as `mcp_servers."drop".enabled=false` failed startup. Bare
`mcp_servers.drop.enabled=false` disabled the named server. For complex names,
`mcp_servers={"dot.drop"={enabled=false},"drop"={enabled=false}}` disabled both
literal names while preserving the selected server and existing command/argument
fields. Fixture configuration bytes remained unchanged. The native
[CLI parser](https://github.com/openai/codex/blob/main/codex-rs/utils/cli/src/config_override.rs)
and [override layering](https://github.com/openai/codex/blob/main/codex-rs/config/src/overrides.rs)
explain this contract: keys split on dots, while values parse as TOML.

Git 2.50.1 was checked with real temporary repositories. HEAD-based
[`check-attr --source`](https://git-scm.com/docs/git-check-attr) detects effective
checkout filters without writing the original index or executing the filter.
Regression cases cover committed, info and global attributes, modified local
attributes, unused drivers and empty local overrides of global commands.
Target-worktree conditional includes are checked after owned registration and
before population; tests cover both automatic cleanup and retained refusals,
without executing the configured filter or post-checkout hook. Additional cases
cover inherited `GIT_ATTR_SOURCE` and ordinary/conditional `attr.tree` settings
pointing at a different commit, plus conditional hook paths through committed
directory, file and chained symlinks, including a literal filename beginning
with Git pathspec magic. A native macOS probe also reproduced an escaping
case-variant link chain; case and Unicode-normalization aliases and duplicate
link identities are now refused consistently across platforms. Older
Git lacking this option refuses the configured-driver preflight. Relative
worktree bindings are validated by canonical identity, including after native
`worktree repair`; cleanup remains conditional on duplicate ownership receipts.

## Process ownership

Tests use local executables and real temporary Git repositories. Isolated Plan
runs dispatch through the existing Codex terminal Plan transport. Normal exit
and SIGTERM cancellation both stop heartbeat descendants before the checkout is
removed. The new transcript transport propagates terminal size changes and
restores the caller's signal handler. A bounded native macOS regression delays
kqueue registration until a real child exits, without pre-reaping it; ESRCH must
complete teardown rather than wait on an unregistered watcher.

These checks do not claim ownership of deliberately detached processes or
protection against concurrently malicious repository metadata changes. Ambiguous
ownership or failed process teardown retains the checkout for inspection.
