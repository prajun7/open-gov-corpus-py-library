# PyPI Deployment Guide

## Prerequisites

### 1. Create PyPI Accounts

- **PyPI**: Create account at https://pypi.org/account/register/
- **TestPyPI**: Create account at https://test.pypi.org/account/register/ (for testing)

### 2. Generate API Tokens

#### For TestPyPI:

1. Log in to https://test.pypi.org
2. Go to Account Settings → API tokens (or visit https://test.pypi.org/manage/account/token/)
3. Click "Add API token"
4. Name it (e.g., `testpypi-uploader`)
5. Set scope to your project name: `opengovcorpus`
6. Click "Add token" and **copy the token immediately** (starts with `pypi-`)

#### For PyPI:

1. Log in to https://pypi.org
2. Go to Account Settings → API tokens (or visit https://pypi.org/manage/account/token/)
3. Click "Add API token"
4. Name it (e.g., `pypi-uploader`)
5. Set scope to your project name: `opengovcorpus` (not "Entire account")
6. Click "Add token" and **copy the token immediately** (starts with `pypi-`)

## Before Deployment

### Activate Virtual Environment

If you're using a virtual environment, activate it first:

```bash
# From project root directory
source open-gov-corpus-env/bin/activate
```

Or with full path:

```bash
source /path/to/open-gov-corpus-py-library/open-gov-corpus-env/bin/activate
```

You should see `(open-gov-corpus-env)` in your terminal prompt when activated.

### Update Version

Update version in:

- `pyproject.toml`: `version = "0.1.0"`
- `opengovcorpus/__init__.py`: `__version__ = "0.1.0"`

## Building the Package

### Step 1: Install Build Tools

```bash
python -m pip install --upgrade build twine
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

When prompted:

- **Username**: `__token__`
- **Password**: Your TestPyPI API token (starts with `pypi-`)

### Step 2: Test Installation

```bash
pip install --index-url https://test.pypi.org/simple/ opengovcorpus
```

### Step 3: Verify

```python
import opengovcorpus as og
print(og.__version__)
```

## Deploying to PyPI

### Step 1: Upload to PyPI

```bash
python -m twine upload dist/*
```

When prompted:

- **Username**: `__token__`
- **Password**: Your PyPI API token (starts with `pypi-`)

### Step 2: Verify on PyPI

Visit: https://pypi.org/project/opengovcorpus/

### Step 3: Install from PyPI

```bash
pip install opengovcorpus
```

## Updating the Package

For each new release:

1. Update version in `pyproject.toml` and `opengovcorpus/__init__.py`
2. Build: `python -m build`
3. Check: `twine check dist/*`
4. Upload: `python -m twine upload dist/*`

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH` (e.g., `0.1.0`)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

## Security Notes

- Never commit API tokens to git
- Use project-scoped tokens (not "Entire account")
- Test on TestPyPI before production deployment
- Store tokens securely (password manager)
