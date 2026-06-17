# Releasing imference-engine

Workers pin a **tagged** version of the engine (never `main`) so prod only moves
when you deliberately bump it. Development continues on `main`; a release is a tag.

## Cut a release

1. Make sure `main` is in a known-good state (tests pass: `pytest tests/test_wan.py`).
2. Bump the version **in both files, kept in sync** (the worker's `start.sh` override
   compares `importlib.metadata.version("imference-engine")` to the tag):
   - `pyproject.toml` → `[project] version = "X.Y.Z"`
   - `imference_engine/__init__.py` → `__version__ = "X.Y.Z"`
3. Commit, then tag and push:
   ```bash
   git commit -am "release vX.Y.Z"
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```
   The install URL `https://github.com/Publikey/imference-engine/archive/refs/tags/vX.Y.Z.tar.gz`
   is live as soon as the tag is pushed (public repo, no token).

## Ship it to the worker

Bump the tag in the worker that consumes the engine and redeploy:
- `gen-image-workers/workers/wan-video-diffuser/requirements.txt`
- `gen-image-workers/workers/wan-video-diffuser/requirements-prefetch.txt`

Prod uses **only** this pinned tag. To override per environment or roll back without
a code change, set `IMFERENCE_ENGINE_REF=vX.Y.Z` (or `main` to track dev) in the
worker's vault — `start.sh` reinstalls that ref on boot.

## Versioning

Semver. Bump **minor** for new features / model recipes, **patch** for fixes,
**major** for breaking API or dependency-combo changes (e.g. the `[wan]` /
`[runtime]` diffusers pins). Tag must equal the `pyproject`/`__init__` version.
