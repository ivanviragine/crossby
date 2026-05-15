# Publish Plan: PyPI + Public Repo

Steps to take crossby from its current branch state to a tagged public
release on PyPI. Replaces an earlier draft that predates PR #46 and #47;
the old "squash history + force-push" step has been dropped — git history
is documentation now that there are multiple contributors.

---

## Phase 0 — Decisions to make before starting

These are real choices, not boilerplate. Picking them up front avoids
the rest of the plan branching.

- [ ] **Version**: keep `0.2.x` (preserves apparent maturity from prior
      private releases) or reset to `0.1.0` (signals "first public").
      Either defensible; pick and write it in the release notes.
      Current value in `pyproject.toml`: `0.2.3`. The
      `crossby/__init__.py` `__version__` must stay in lockstep
      (see the `steps.md` prep note).
- [ ] **Squash history?** Recommendation: **no**. PR #46 has co-author
      attribution; PR #47 has 28 commits documenting the design
      narrative. Squashing destroys both. Open-source projects almost
      never do this.
- [ ] **Merge PR #47 (`claude/sync-fidelity-pass`) before publishing?**
      Recommendation: **yes**, if its scope is in. The plan below
      assumes it has merged into `main`.
- [ ] **Release notes location**: a `CHANGELOG.md` in the repo, or
      GitHub Releases only? `CHANGELOG.md` is more durable; GH
      Releases is faster.
- [ ] **Trusted publishing or API token?** Trusted publishing
      (`pypa/gh-action-pypi-publish`) is strongly preferable for a
      public repo — no long-lived secret to leak.

---

## Phase 1 — Repo hygiene

- [ ] **Re-scan open issues and PRs**; close or label stale ones.
      Document any explicit "won't fix" decisions in the issue thread.
- [ ] **Audit remote branches.** Keep `main` and the branch hosting
      the current PR. Decide deliberately about
      `worktree-review-fixes`, `refactor/codebase-review-cleanup`,
      `feat/42-*`, `fix/gemini-parity`, etc. — they may have salvage
      value or may be done.
- [ ] **Remove `REVIEW_FIXES.md`** — internal working notes from an
      earlier review cycle; not user-facing.
- [ ] **Confirm legacy hygiene** (these should already be gone):
      `KNOWLEDGE.md`, `.coverage`, `.wade*`.
- [ ] **Re-read top-level docs** for accuracy after recent merges:
      `README.md`, `CONTRIBUTING.md`, `steps.md`. They were heavily
      updated in PR #47; the surface descriptions should match what
      `crossby --help` actually shows.

---

## Phase 2 — Test the publish surface

Hard gates. Don't proceed past Phase 2 with anything red.

- [ ] `./scripts/check-all.sh` green
      (tests + lint + format check + mypy strict)
- [ ] `uv build` produces both wheel and sdist without errors
- [ ] **Wheel install smoke** in a fresh venv:
      ```bash
      python -m venv /tmp/crossby-publish-test
      /tmp/crossby-publish-test/bin/pip install dist/crossby-*.whl
      /tmp/crossby-publish-test/bin/crossby --version    # prints the right version
      /tmp/crossby-publish-test/bin/crossby --help       # lists every subcommand
      /tmp/crossby-publish-test/bin/crossby sync --help  # lists every new flag
      ```
- [ ] **Bundled skill resource smoke** — the `src/crossby/data/skill/`
      tree ships in the wheel. Verify `importlib.resources` resolves
      it from the installed package, not just from source:
      ```bash
      cd /tmp && mkdir publish-skill-test && cd publish-skill-test
      /tmp/crossby-publish-test/bin/crossby init --non-interactive --install-skill
      ls .claude/skills/crossby-sync/   # SKILL.md + agents/ + references/
      ```
- [ ] **Manual walkthrough**: run `steps.md` end-to-end against the
      installed wheel. Every checkbox in:
      - 1a–1i (sync: wizard / direct / dry-run / safety / plan-doctor-
        validate / translate strategy / report formats / plugins /
        legacy `.agents/`)
      - 2 (launch)
      - 3 (handoff)
      - 4 (convert)
      - 5 (stats)
      - 6 (console output sanity)
      - 6b (`crossby init --install-skill`)
      - 7 (edge cases incl. Bearer/`${VAR}` MCP rewrites)
      - 8 (packaging — but you've already done most of this in this
        phase)

      Run on the primary dev OS at minimum. A CI matrix for
      macOS/Linux/Windows can land post-publish.

---

## Phase 3 — Release

- [ ] **Bump version** in both `pyproject.toml` AND
      `src/crossby/__init__.py` `__version__`. They must match.
- [ ] **Draft release notes**. Cover the headline surface added since
      the last release at minimum:
      - `crossby agents convert` (cross-tool subagent format
        translation; PR #46)
      - `crossby sync --strategy translate` (per-file agent + skill
        translation with manual-fix blocks)
      - `--plan` / `--doctor` / `--validate-target` pre-write modes
      - `--report-format markdown-table` + persistent
        `.crossby/sync-report.md`
      - `crossby init --install-skill` and the bundled `crossby-sync`
        skill
      - Plugin discovery (`SyncConcern.PLUGINS`)
      - Cross-provider model translation in `crossby launch`
        (claude ↔ codex family mapping)
      - MCP transport rewrites for Codex (`Authorization: Bearer ${VAR}`
        → `bearer_token_env_var`, etc.)
      - **Behavior change worth flagging**: codex agents path moved
        from `.agents/` → `.codex/agents/` (per upstream Codex docs).
        Crossby logs a `structlog` warning when it detects the legacy
        layout on next sync; the old directory is left untouched but
        no longer participates.
- [ ] `git tag vX.Y.Z && git push origin vX.Y.Z`
- [ ] **Publish**:
      ```bash
      uv publish dist/crossby-*.whl dist/crossby-*.tar.gz
      ```
      Or, if trusted publishing is set up, push the tag and let the GH
      Action upload.
- [ ] **Post-publish smoke** in a clean env:
      ```bash
      python -m venv /tmp/crossby-pypi-check
      /tmp/crossby-pypi-check/bin/pip install --upgrade crossby
      /tmp/crossby-pypi-check/bin/crossby --version
      ```
- [ ] Verify https://pypi.org/project/crossby/ shows the new version.

---

## Phase 4 — Visibility

- [ ] Confirm repo visibility. If still private:
      `gh repo edit --visibility public`.
- [ ] Pin the release on the GitHub Releases page.
- [ ] Optional: brief announcement (HN / social / mailing list).

---

## Optional follow-ups (post-publish)

Not blocking the release, but worth queuing:

- CI matrix (macOS + Linux + Windows × Python 3.11 / 3.12 / 3.13)
  running `./scripts/check-all.sh` on every PR
- Trusted-publishing GH Action for subsequent versions
- `CHANGELOG.md` if Phase 0 chose GH Releases for v0.x; revisit for
  v0.y or v1.0
- Documentation site (mkdocs / docusaurus) if README + CONTRIBUTING
  start feeling cramped
