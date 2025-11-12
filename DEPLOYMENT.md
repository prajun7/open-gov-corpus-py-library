# PyPI Deployment Guide

This guide explains how to deploy OpenGovCorpus to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **TestPyPI Account**: Create an account at https://test.pypi.org/account/register/ (for testing)
3. **API Tokens**: Generate API tokens from your PyPI account settings

## Before Deployment

### 1. Update `setup.py`

Update these fields in `setup.py`:

```python
author_email="your-email@example.com",
url="https://github.com/yourusername/opengovcorpus",
    "Bug Reports": "https://github.com/yourusername/opengovcorpus/issues",
    "Source": "https://github.com/yourusername/opengovcorpus",
    "Documentation": "https://github.com/yourusername/opengovcorpus#readme",
}
```

### 2. Update Version

Update the version in:

- `setup.py`: `version="0.1.0"`
- `opengovcorpus/__init__.py`: `__version__ = "0.1.0"`

## Building the Package

### Step 1: Install Build Tools

```bash
pip install build twine
```

### Step 2: Clean Previous Builds

```bash
rm -rf dist/ build/ *.egg-info
```

### Step 3: Build the Package

```bash
python -m build
```

This creates:

- `dist/opengovcorpus-0.1.0.tar.gz` (source distribution)
- `dist/opengovcorpus-0.1.0-py3-none-any.whl` (wheel)

### Step 4: Check the Build

```bash
twine check dist/*
```

## Testing on TestPyPI

### Step 1: Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

You'll be prompted for:

- Username: `__token__`
- Password: Your TestPyPI API token (starts with `pypi-`)

### Step 2: Test Installation

```bash
pip install --index-url https://test.pypi.org/simple/ opengovcorpus
```

### Step 3: Verify It Works

```python
import opengovcorpus as og
print(opengovcorpus.__version__)
```

## Deploying to PyPI

### Step 1: Upload to PyPI

```bash
twine upload dist/*
```

You'll be prompted for:

- Username: `__token__`
- Password: Your PyPI API token (starts with `pypi-`)

### Step 2: Verify on PyPI

Visit: https://pypi.org/project/opengovcorpus/

### Step 3: Install from PyPI

```bash
pip install opengovcorpus
```

## Updating the Package

For each new release:

1. Update version in `setup.py` and `opengovcorpus/__init__.py`
2. Update `CHANGELOG.md` (if you have one)
3. Build: `python -m build`
4. Check: `twine check dist/*`
5. Upload: `twine upload dist/*`

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH` (e.g., `0.1.0`)
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes
- Verify your API token is correct
- Make sure you're using `__token__` as username
- Token should start with `pypi-` for PyPI or `pypi-AgEI...` for TestPyPI

## Files Included in Package

The package includes:

- All files in `opengovcorpus/` directory
- `README.md`
- `LICENSE`
- `requirements.txt`
- Files in `usage_examples/`

Excluded:

- `tests/` directory
- `initial_scraping_notebook_code/`
- `open-gov-corpus-env/`
- `*.pyc` files
- `.git/` directory

## Security Notes

- Never commit API tokens to git
- Use environment variables or `.pypirc` file for credentials
- Test on TestPyPI before production deployment
