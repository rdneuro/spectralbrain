# Release Guide — SpectralBrain

This document walks through the complete release process: from bumping
the version to seeing `spectralbrain` live on PyPI.  The workflow is
designed so that **creating a GitHub Release** is the only manual step;
everything else is automated.

---

## One-time setup (you only do this once)

### 1. Configure PyPI Trusted Publisher

The OIDC trusted publisher model eliminates API tokens entirely.  PyPI
authenticates your GitHub Actions workflow via short-lived OIDC
credentials — nothing is stored as a secret.

Go to <https://pypi.org/manage/account/publishing/> and add a **pending
publisher** with these exact values:

| Field            | Value                            |
|------------------|----------------------------------|
| PyPI project     | `spectralbrain`                  |
| Owner            | `<your-github-username-or-org>`  |
| Repository       | `spectralbrain`                  |
| Workflow name    | `publish.yml`                    |
| Environment name | `pypi`                           |

Repeat the same process on **TestPyPI** at
<https://test.pypi.org/manage/account/publishing/>, using environment
name `testpypi`.

### 2. Create GitHub Environments

In your repository, go to **Settings → Environments** and create two
environments:

**`pypi`** — for production PyPI.  Recommended protection rules:

- ✅ Required reviewers (add yourself) — this adds a manual approval
  step before the package goes live.
- ✅ Allow only the `main` branch.

**`testpypi`** — for TestPyPI.  You can skip protection rules here
since TestPyPI is a sandbox.

### 3. Verify CI is green

Push the `.github/workflows/ci.yml` and `.github/workflows/publish.yml`
files to your repository.  The CI workflow should trigger on the next
push to `main` or any PR.  Make sure it passes before attempting a
release.

---

## Making a release

### Step 1 — Ensure `main` is clean

```bash
git checkout main
git pull origin main
uv run pytest tests/ -v         # all tests pass
uv run ruff check spectralbrain/  # no lint errors
```

### Step 2 — Create a git tag

SpectralBrain uses `hatch-vcs` for versioning: the version number is
derived from the most recent git tag.  There is no version string to
edit manually in `pyproject.toml`.

```bash
# Choose the next version following semantic versioning.
# Examples: v0.1.0, v0.2.0, v1.0.0, v1.0.1
git tag v0.1.0
git push origin v0.1.0
```

Verify that `hatch-vcs` picks it up:

```bash
uv run python -c "import spectralbrain; print(spectralbrain.__version__)"
# Should print: 0.1.0
```

### Step 3 — Create a GitHub Release

Go to your repo's **Releases** page → **Draft a new release**:

1. **Choose a tag**: select the tag you just pushed (e.g., `v0.1.0`).
2. **Release title**: `v0.1.0` (or a descriptive name).
3. **Description**: paste the relevant section of your CHANGELOG, or
   click "Generate release notes" to auto-populate from merged PRs.
4. **Pre-release**: check this box for alpha/beta/rc versions.
5. Click **Publish release**.

### Step 4 — Watch the automation

Creating the release triggers `.github/workflows/publish.yml`.  Go to
the **Actions** tab to watch the progress:

1. **Build** job: builds the sdist + wheel, verifies the version
   matches the tag, uploads artifacts.
2. **Publish to TestPyPI** job: uploads to the TestPyPI sandbox
   (automatic, no approval needed).
3. **Publish to PyPI** job: waits for your manual approval (if you
   configured environment protection), then publishes.

The whole process takes about 2 minutes.

### Step 5 — Verify the release

```bash
# From TestPyPI (should work immediately):
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            spectralbrain

# From production PyPI (after approval):
pip install spectralbrain
python -c "import spectralbrain; print(spectralbrain.__version__)"
```

---

## How versioning works

SpectralBrain uses **hatch-vcs**, which reads version information
from git tags automatically.  The rules are:

| Git state                        | Resulting version        |
|----------------------------------|--------------------------|
| Exactly on tag `v0.2.0`         | `0.2.0`                  |
| 3 commits after `v0.2.0`        | `0.2.1.dev3+g<hash>`    |
| No tags at all                   | `0.1.0.dev0` (fallback) |
| Dirty working tree on `v0.2.0`  | `0.2.1.dev0+dirty`      |

This means you **never** need to edit a version string in a Python
file.  The single source of truth is the git tag.

---

## Releasing a patch / hotfix

```bash
git checkout main
# Make your fix, commit it.
git tag v0.1.1
git push origin main v0.1.1
# Create a GitHub Release for v0.1.1.
```

---

## Releasing a pre-release (alpha/beta/rc)

```bash
git tag v0.2.0a1    # alpha
git tag v0.2.0b1    # beta
git tag v0.2.0rc1   # release candidate
git push origin v0.2.0a1
# Create a GitHub Release and CHECK the "Pre-release" box.
```

Pre-releases go through the same pipeline.  PyPI will mark them as
pre-release, so `pip install spectralbrain` won't pick them up unless
the user opts in with `pip install --pre spectralbrain`.

---

## Troubleshooting

**"Version mismatch" error in the build job**
This means `hatch-vcs` derived a version that doesn't match the
release tag.  Usually caused by not fetching the full git history.
The workflow uses `fetch-depth: 0` to prevent this.  If you're
running locally, make sure the tag is on the current commit:
`git describe --tags --dirty`.

**"Token exchange failed" in the publish job**
The OIDC handshake between GitHub and PyPI failed.  Check that:
(a) the workflow filename in your trusted publisher config on PyPI
matches exactly (`publish.yml`), (b) the environment name matches
(`pypi`), and (c) the `id-token: write` permission is present.

**TestPyPI succeeds but PyPI fails**
If you're using environment protection rules, check the Actions tab —
the PyPI publish job may be waiting for manual approval.  Click
"Review deployments" and approve it.

**Package installs but `import spectralbrain` fails**
Run `pip show spectralbrain` to check it's installed, then
`python -c "import spectralbrain; print(spectralbrain.__file__)"`.
If the version shows `0.1.0.dev0`, the git tag wasn't present at
build time.  Rebuild with the tag on the commit.
