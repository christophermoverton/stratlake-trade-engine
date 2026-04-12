# Contributing to StratLake Trade Engine

Thank you for your interest in contributing! This guide explains how external
contributors can propose changes, and how the repository owner reviews submitted
code.

---

## Who Can Contribute

Anyone can contribute via a **fork-and-pull-request** workflow. You do not need
write access to the repository. All contributions are reviewed before they are
merged.

---

## Workflow for External Contributors

### 1. Fork the repository

Click **Fork** on the GitHub repository page. This creates a personal copy under
your own GitHub account.

### 2. Clone your fork locally

```bash
git clone https://github.com/<your-username>/stratlake-trade-engine.git
cd stratlake-trade-engine
```

### 3. Add the upstream remote

```bash
git remote add upstream https://github.com/christophermoverton/stratlake-trade-engine.git
```

This lets you pull in changes from the main repository later:

```bash
git fetch upstream
git merge upstream/main
```

### 4. Create a branch for your work

Name the branch after the issue you are addressing:

```bash
git checkout -b fix/issue-206-reuse-policy-controls
```

Branch naming conventions:

| Type | Pattern | Example |
|------|---------|---------|
| Bug fix | `fix/issue-<number>-<short-desc>` | `fix/issue-203-fingerprint-hash` |
| Feature | `feat/issue-<number>-<short-desc>` | `feat/issue-206-reuse-policy` |
| Documentation | `docs/issue-<number>-<short-desc>` | `docs/issue-201-getting-started` |
| Refactor | `refactor/<short-desc>` | `refactor/campaign-runner` |

### 5. Set up the development environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# or: .\.venv\Scripts\Activate.ps1   # Windows PowerShell

pip install -e ".[dev]"
cp .env.example .env
# edit .env so MARKETLAKE_ROOT points to your curated data root
```

### 6. Make your changes

* Keep changes focused on the issue you are addressing.
* Follow existing code style (the project uses `ruff` for linting).
* Add or update tests in the `tests/` directory to cover your changes.
* Update documentation in `docs/` if your changes affect documented behavior.

### 7. Lint and test locally before pushing

```bash
# Lint
ruff check src/ tests/

# Run tests
pytest
```

All lint and test checks must pass before submitting.

### 8. Commit your changes

Write clear commit messages that reference the issue:

```bash
git add .
git commit -m "feat(orchestration): add --force-rerun and --disable-reuse flags (issue #206)"
```

### 9. Push your branch to your fork

```bash
git push origin fix/issue-206-reuse-policy-controls
```

### 10. Open a Pull Request

1. Go to **your fork** on GitHub.
2. Click **Compare & pull request**.
3. Set the base repository to `christophermoverton/stratlake-trade-engine` and
   the base branch to `main`.
4. Fill in the pull request template (it will appear automatically).
5. In the PR description, reference the issue with `Closes #206` so the issue
   is linked and will close automatically on merge.
6. Submit the pull request.

---

## What Happens After You Submit

1. **Automated CI** runs immediately — lint checks and the test suite execute
   against your branch. You can watch the results in the **Checks** tab of your
   PR.
2. The **repository owner is notified** and will review the code.
3. Feedback is left as PR review comments. Address each comment and push
   updated commits to the same branch — the PR updates automatically.
4. Once the review is approved and all CI checks pass, the PR will be merged.

> **Note:** CI checks run on pull requests from forks. No repository secrets
> are exposed to fork workflows. The owner may run additional internal checks
> before merging sensitive changes.

---

## Code Standards

| Area | Tool / Convention |
|------|-------------------|
| Linting | `ruff` (`ruff check src/ tests/`) |
| Tests | `pytest` (test files live in `tests/`) |
| Style | Follow patterns in the file you are editing |
| Imports | Standard library → third-party → local, one blank line between groups |
| Type hints | Preferred for all public functions |

---

## Asking Questions

If you are unsure about the design direction before writing code, leave a comment
on the relevant GitHub issue. The owner will respond with guidance before you
start implementing. See the comment on issue #206 as an example of this
pre-implementation alignment process.

---

## Commit Message Convention

```
<type>(<scope>): <short summary> (issue #<number>)
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Scope examples: `orchestration`, `cli`, `research`, `portfolio`, `alpha`

---

## Pull Request Checklist

Before marking your PR ready for review, confirm:

- [ ] All existing tests pass (`pytest`)
- [ ] New or modified code is covered by tests
- [ ] `ruff check` passes with no errors
- [ ] Relevant documentation in `docs/` is updated
- [ ] The PR description references the issue (`Closes #<number>`)
- [ ] Commit messages follow the convention above
