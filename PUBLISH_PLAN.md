# Publish Plan: PyPI + Public Repo

## Pre-flight

- **PyPI name "crossby"**: Available (confirmed 404)
- **GitHub repo**: Private, 9 issues (2 open: #13, #14), 9 PRs (2 open: #4, #15)

---

## Phase 1 — Clean GitHub State

- [ ] Close open issues (#13, #14)
- [ ] Close open PRs (#4, #15)
- [ ] Delete all remote feature branches

## Phase 2 — Clean Git History

- [ ] Squash all history into a single initial commit (`git checkout --orphan fresh`, commit, replace `main`)
- [ ] Force push to origin

## Phase 3 — Clean Repo Files

- [ ] Remove `REVIEW_FIXES.md`
- [ ] Remove `KNOWLEDGE.md`
- [ ] Remove `.coverage` if tracked
- [ ] Remove wade-managed files (`.wade/`, `.wade-managed`, `.wade.yml`, wade skills/hooks in `.claude/`)
- [ ] Update `.gitignore`

## Phase 4 — README Split (#20)

- [ ] Rewrite `README.md` — user-facing only (following Wade's style)
- [ ] Create `CONTRIBUTING.md` — dev setup, scripts, architecture, commits, releases

## Phase 5 — PyPI Publishing Prep

- [ ] Confirm `version = "0.1.0"` as first public release
- [ ] Verify `LICENSE` file is present
- [ ] Test build locally (`uv build`)
- [ ] Create PyPI API token or set up trusted publishing
- [ ] Publish (`uv publish` or `twine upload dist/*`)
- [ ] Optionally: set up GitHub Actions for automated releases

## Phase 6 — Make Public

- [ ] `gh repo edit --visibility public`
