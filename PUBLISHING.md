# Publishing RiboSeek to PyPI

This is an internal runbook for cutting a new release.

## Prerequisites

1. **PyPI account** with the `riboseek` project owner permission.
   Register at <https://pypi.org/account/register/> if needed.

2. **API tokens** (recommended over password). Generate at
   <https://pypi.org/manage/account/token/>. Create one for `riboseek`
   scope, store in `~/.pypirc`:

   ```ini
   [pypi]
   username = __token__
   password = pypi-AgEIcHlwaS5vcmcCJ...   # your token

   [testpypi]
   username = __token__
   password = pypi-AgENdGVzdC5weXBpLm9yZw...
   ```

3. **Build + upload tools**:

   ```bash
   pip install --upgrade build twine
   ```

## Cutting a release

### 1. Bump the version

Edit `src/riboseek/_version.py` and `pyproject.toml`:

```python
# src/riboseek/_version.py
__version__ = "0.1.1"
```

Commit:

```bash
git commit -am "Release 0.1.1"
git tag -a v0.1.1 -m "RiboSeek 0.1.1"
git push origin main --tags
```

### 2. Build sdist + wheel

```bash
rm -rf dist build *.egg-info
python -m build
ls dist/
# riboseek-0.1.1-cp311-cp311-linux_x86_64.whl
# riboseek-0.1.1.tar.gz
```

For multi-platform wheels (Windows/macOS/Linux × Python 3.9-3.12) use
`cibuildwheel`:

```bash
pip install cibuildwheel
cibuildwheel --output-dir wheelhouse
```

### 3. Upload to TestPyPI first

```bash
twine upload --repository testpypi dist/*
```

Then verify install + smoke test in a fresh venv:

```bash
python -m venv /tmp/rstest && /tmp/rstest/bin/pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    riboseek
/tmp/rstest/bin/python -c "from riboseek import Searcher; s = Searcher.from_pretrained(); print(len(s.encoded_chains), 'chains')"
```

### 4. Upload to real PyPI

Once TestPyPI checks out:

```bash
twine upload dist/*
```

Verify:

```bash
pip install --upgrade riboseek
riboseek info
```

### 5. Create the GitHub release with the full database asset

The 16 K-chain database is hosted as a release asset, not bundled in the
wheel:

```bash
# Build a gzip-compressed version of the full encoded chain set:
gzip -9 -k -c \
    /data2/wangding/projects/RNA3dSeek/data/processed/alphabet_leakfree/encoded_chains.json \
    > encoded_chains.json.gz
ls -la encoded_chains.json.gz       # ~10 MB

# Then attach it to the v0.1.1 GitHub release:
gh release create v0.1.1 encoded_chains.json.gz \
    --title "RiboSeek 0.1.1" \
    --notes "First public PyPI release. Full 16,641-chain SA-20 database attached."
```

The `riboseek download-db` CLI command pulls this asset by URL; the URL
template in `src/riboseek/cli.py` (`FULL_DB_URL`) is pinned to v0.1.0 —
**update it** if the asset URL changes.

## Things to double-check before the first PyPI upload

- [ ] `riboseek` namespace on PyPI is still available (it was as of 2026-05-26).
- [ ] Email `void-echo@outlook.com` in `pyproject.toml` is the maintainer
      address you want listed publicly.
- [ ] Demo database (`src/riboseek/data/demo_db.json`) does **not** leak
      anything you don't want public. It's 50 chain IDs + their SA-20
      labels; no coordinates, no sequences.
- [ ] The PUBLISHING.md and any internal-only files are not in MANIFEST.in
      so they stay out of the sdist (they currently are not — verify with
      `tar -tzf dist/riboseek-*.tar.gz | grep -i publish`).
