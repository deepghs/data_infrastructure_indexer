# Repository Guidelines

## Core Repository Rules
This repository is English-only. All tracked content must be written in English, including source code, comments, docstrings, Markdown files, generated documentation, examples, configuration text, commit messages, and any new human-readable strings added to the codebase. Do not introduce any other language into repository files.

This rule also applies when using Codex or any other agent. Conversation language should match the requester's language, but every repository-facing action must remain in English. That includes file edits, code comments, log messages, generated README content, commit messages, and any other text written into the repository or its git history.

## Commit Identity And Commit Message Policy
Every commit created for this repository must use the git identity:

```bash
git config user.name "narugo1992"
git config user.email "narugo1992@deepghs.org"
```

Verify the active identity before creating a commit:

```bash
git config user.name
git config user.email
```

If your local identity is different, override it before committing. Do not create commits under any other name or email.

Recent history follows a `dev(<author>): <summary>` style. New commits should keep that structure and use an English summary, preferably:

```text
dev(narugo1992): add zerochan session retry guard
```

The first line must stay in the `xxxx(<author>): xxxx` format, but commit messages are now expected to use a detailed multi-line body after that summary line. Follow the long-form style used in `~/oo-projects/pyfcstm` history:

- leave one blank line after the summary line
- add one concise explanatory paragraph describing the intent or user-visible outcome
- add a flat bullet list with concrete change points
- add a `Tests:` section when validation was run, with one bullet per command
- do not collapse the entire message into one line or rely on literal `\n` escape sequences inside a single shell string

Preferred commit message shape:

```text
dev(narugo1992): disable scheduled workflow triggers

Pause the non-date scheduled GitHub Actions workflows so they stop running automatically while keeping every cron expression visible for staged re-enablement.

- comment out each non-date workflow `schedule:` block instead of deleting it
- keep `workflow_dispatch` enabled so each workflow can still be triggered manually
- document the repository commit message expectations in AGENTS.md

Tests:
- python - <<'PY' ...
- git diff --check
```

When creating a multi-line commit from the shell, use one `-m` per paragraph instead of embedding `\n` in a single argument. For example:

```bash
git commit \
  -m "dev(narugo1992): disable scheduled workflow triggers" \
  -m "Pause the non-date scheduled GitHub Actions workflows so they stop running automatically while keeping every cron expression visible for staged re-enablement." \
  -m "- comment out each non-date workflow schedule block instead of deleting it
- keep workflow_dispatch enabled for manual recovery
- document the long-form commit standard in AGENTS.md" \
  -m "Tests:
- git diff --check"
```

For longer messages, prefer writing the message into a temporary file and passing it with `-F`:

```bash
cat <<'EOF' >/tmp/commit-message.txt
dev(narugo1992): disable scheduled workflow triggers

Pause the non-date scheduled GitHub Actions workflows so they stop running automatically while keeping every cron expression visible for staged re-enablement.

- comment out each non-date workflow `schedule:` block instead of deleting it
- keep `workflow_dispatch` enabled so each workflow can still be triggered manually
- document the repository commit message expectations in AGENTS.md

Tests:
- git diff --check
EOF

git commit -F /tmp/commit-message.txt
```

Do not use `git commit -m "subject\n\nbody"` style commands in this repository.

Keep commit scope narrow. One commit should usually cover one site adapter, one workflow change, or one focused bug fix.

Do not include local absolute paths in commit messages, commit bodies, repository files, generated artifacts, examples, logs, or screenshots checked into the repository. Paths such as `/home/<user>/...`, `/Users/<name>/...`, or `C:\Users\<name>\...` can expose private usernames, workstation layouts, or other sensitive local details. Rewrite them as repository-relative paths, sanitized placeholders, or generic examples before committing.

## Project Structure And Module Organization
`inf/` contains the runtime code. Each source site is isolated in its own package, including `inf/danbooru/`, `inf/e621/`, `inf/gelbooru/`, `inf/kemono/`, `inf/konachan/`, `inf/rule34/`, `inf/yande/`, and `inf/zerochan/`.

Within each site package, module names are task-oriented:

- `index.py` or `index_n.py`: primary sync or indexing jobs
- `tags.py` or `tags_versioned.py`: tag synchronization
- `dbsquash.py`: repository compaction or repack jobs
- `base.py`: site-specific session, auth, or shared helpers

Shared HTTP/session utilities belong in `inf/utils/`, especially reusable request logic such as `inf/utils/session.py`.

Top-level control files:

- `Makefile`: local helper targets
- `requirements.txt`, `requirements-test.txt`, `requirements-doc.txt`: dependency sets
- `pytest.ini`: pytest markers and timeout
- `codecov.yml`: coverage reporting rules
- `.github/workflows/`: scheduled jobs and the best reference for production entrypoints

The current checkout does not include committed `test/` or `docs/` directories, even though the Makefile references them. When adding tests, place them under `test/` and mirror the `inf/` layout.

## Build, Test, And Development Commands
Use Python 3.8 to match GitHub Actions.

- `python -m venv venv && source venv/bin/activate`
  Create and activate a local virtual environment.
- `pip install -r requirements.txt -r requirements-test.txt`
  Install runtime and test dependencies.
- `python -m inf.zerochan.index`
  Run a site sync job locally.
- `python -m inf.danbooru.index_n`
  Run the Danbooru indexer used by scheduled workflows.
- `python -m inf.e621.tags`
  Run a tag sync task for a single site package.
- `make unittest RANGE_DIR=. WORKERS=4`
  Run pytest with coverage when a `test/` tree exists.
- `make clean`
  Remove `build/`, `dist/`, and `*.egg-info`.
- `flake8 inf test`
  Run the baseline lint pass when tests are present.

When you change a site module, validate it with the closest real entrypoint instead of relying only on import checks.

## Coding Style And Naming Conventions
Follow the existing Python style already present in `inf/`:

- 4-space indentation
- snake_case for modules, functions, variables, and file names
- PascalCase for classes
- uppercase snake_case for environment variables and constants

Keep modules site-scoped and task-scoped. Reuse existing helper patterns before adding new abstractions. If a request/session helper already exists in `inf/utils/` or a site `base.py`, extend that path instead of duplicating network logic in a new module.

Match the surrounding style for imports, logging, retry behavior, pandas usage, and Hugging Face upload logic. Use English-only comments and log messages. Add comments only when they explain non-obvious behavior, rate limits, retry rules, or deployment details.

## Testing And Validation Guidelines
Pytest is the standard test runner. `pytest.ini` defines the markers `unittest`, `benchmark`, and `ignore`, and uses a default timeout of 300 seconds.

Test naming conventions:

- file names: `test_*.py`
- place tests under `test/<site>/`
- mirror the source layout where practical, for example `test/zerochan/test_index.py`

Prefer fast tests for parsing, normalization, transformation, pagination, retry handling, and session helpers. Avoid making the default test path depend on live external services. Networked sync jobs should be smoke-tested explicitly with the required environment variables and credentials.

Coverage is reported through Codecov. There is no strict repository-wide minimum enforced in the config, but you can require one locally with `MIN_COVERAGE=<n> make unittest`.

## Pull Request And Workflow Expectations
Pull requests should be written in English and should clearly state:

- which site packages or workflows changed
- which environment variables or secrets are required for validation
- which commands were run locally
- whether any `.github/workflows/*.yml` files were updated
- whether the change affects Hugging Face dataset layout, upload cadence, or repo naming

If you touch a scheduled job, verify that the corresponding workflow entrypoint still matches the module name and required secrets.

## Security And Configuration Notes
Do not commit secrets, tokens, cookies, private repository identifiers, or local credentials. Runtime commonly depends on environment variables such as `HF_TOKEN`, `REMOTE_REPOSITORY_*`, `ZEROCHAN_USERNAME`, `ZEROCHAN_PASSWORD`, `DANBOORU_USERNAME`, and `DANBOORU_APITOKEN`.

Do not commit local absolute filesystem paths or paste them into tracked examples, logs, fixtures, documentation, or commit history. Sanitize path examples to use repository-relative paths or placeholders so local machine details are not exposed.

When modifying sync behavior, keep private dataset or account details out of examples, logs, and test fixtures. If a change affects authentication or rate limiting, document the operational impact in English in the relevant code comments or pull request.
