# Running Tests

This directory contains unit tests for the OpenGovCorpus library.

## Prerequisites

1. Activate the virtual environment:
   ```bash
   source open-gov-corpus-env/bin/activate
   ```

2. Ensure the package is installed in development mode:
   ```bash
   pip install -e .
   ```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run a specific test file
```bash
pytest tests/test_config.py
pytest tests/test_scraper.py
```

### Run a specific test function
```bash
pytest tests/test_config.py::test_config_save_and_load
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run with coverage report
```bash
pytest tests/ --cov=opengovcorpus --cov-report=html
```

### Run and stop on first failure
```bash
pytest tests/ -x
```

### Run tests in parallel (faster)
```bash
pytest tests/ -n auto
```

## Test Files

- `test_config.py` - Tests for configuration management (Config class, setup_config, load_config)
- `test_scraper.py` - Tests for web scraping functionality

## Test Structure

Each test file contains:
- Test functions prefixed with `test_`
- Isolated test environments using `tempfile.TemporaryDirectory()`
- Assertions to verify expected behavior
- Exception testing with `pytest.raises()`

## Example Output

```
============================= test session starts ==============================
platform darwin -- Python 3.12.2, pytest-9.0.1
collected 5 items

tests/test_config.py::test_config_save_and_load PASSED     [ 20%]
tests/test_config.py::test_config_missing_file PASSED      [ 40%]
tests/test_config.py::test_setup_config PASSED              [ 60%]
tests/test_scraper.py::test_invalid_url PASSED              [ 80%]
tests/test_scraper.py::test_scraper_initialization PASSED   [100%]

============================== 5 passed in 0.39s ===============================
```

