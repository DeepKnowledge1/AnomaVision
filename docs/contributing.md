# Contributing to AnomaVision

Thank you for your interest in contributing to AnomaVision! This document provides guidelines and instructions for contributing.

## 🚀 Quick Start

### 1. Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/AnomaVision.git
cd AnomaVision

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with dev tools
uv sync --extra dev --extra cpu

# Install pre-commit hooks (REQUIRED)
uv run pre-commit install
```

### 2. Create a Branch

```bash
# Create a new branch for your feature or fix
git checkout -b fix/your-fix-name
# or
git checkout -b feat/your-feature-name
```

## ✅ Before Submitting a PR

### Run Code Quality Checks Locally

**ALL of these must pass before submitting:**

```bash
# 1. Format your code
uv run black .
uv run isort .

# 2. Run linters
uv run flake8 anomavision/ anodet/ apps/ tests/

# 3. Run tests
uv run pytest -v

# 4. (Optional) Run pre-commit on all files
pre-commit run --all-files
```

### CI Will Enforce These Checks

⚠️ **Your PR will be blocked if:**
- Code is not formatted with Black
- Imports are not sorted with isort
- Flake8 finds critical issues (F811 redefinitions)
- Tests fail

## 📝 Code Quality Standards

### Formatting

- **Black**: Line length = 88 characters
- **isort**: Import sorting with Black profile

### Linting Rules

We use **lenient flake8** configuration to allow legacy code while blocking critical issues:

**Allowed (won't block PR):**
- E501: Line too long (Black handles this)
- E402: Module imports not at top
- F401: Unused imports (common in `__init__.py`)
- B006-B009: Bugbear warnings (require refactoring)

**Blocked (will fail CI):**
- F811: Redefinition of variables (bug risk)
- E722: Bare `except:` (security risk)

### Type Hints (Optional)

Type hints are encouraged but not required. If you add them:
```python
def process_image(img: np.ndarray, thresh: float = 0.5) -> Tuple[np.ndarray, float]:
    ...
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_padim.py

# Run with coverage
uv run pytest --cov=anomavision --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names: `test_padim_training_on_cpu()`

Example:
```python
import pytest
import torch
from anomavision import Padim

def test_padim_initialization():
    model = Padim(backbone="resnet18", device="cpu")
    assert model.backbone == "resnet18"

def test_padim_training_basic():
    # Your test here
    pass
```

## 📦 Dependency Changes

**If you modify `pyproject.toml`:**

```bash
# Regenerate the lockfile
uv lock --python 3.10

# Commit both files
git add pyproject.toml uv.lock
git commit -m "deps: add new dependency X"
```

⚠️ **CI will fail if `uv.lock` is not updated** (the `--locked` flag enforces this)

## 🎯 Pull Request Process

### 1. Fill Out the PR Template

When you create a PR, fill out all sections:

- **Related Issue**: Link to the issue this fixes
- **Description**: Explain what changed and why
- **Type of Change**: Bug fix, feature, docs, etc.
- **Hardware Testing**: Which extras did you test? (`cpu`, `cu121`, etc.)
- **Developer Checklist**: Confirm all items

### 2. PR Review Process

1. **Automated Checks** run first:
   - Code Quality (Black, isort, flake8)
   - Tests (Python 3.10, 3.11, 3.12)
   - CUDA matrix verification

2. **Maintainer Review**:
   - Code quality
   - Test coverage
   - Documentation updates

3. **Approval & Merge**:
   - At least 1 maintainer approval required
   - All CI checks must pass
   - No merge conflicts

## 🐛 Bug Reports

Use the [Bug Report Template](.github/ISSUE_TEMPLATE/bug-report.yml):

**Must include:**
- Operating System
- Python version
- Package manager used (uv, pip, poetry)
- Hardware bracket (`anomavision[cpu]`, `anomavision[cu121]`, etc.)
- Minimal reproducible example

## 🚀 Feature Requests

Use the [Feature Request Template](.github/ISSUE_TEMPLATE/feature-request.yml):

**Must include:**
- Feature description
- Use case & benefits
- Priority level

## 📖 Documentation

### Updating Documentation

If your PR changes functionality:

1. Update docstrings:
```python
def new_function(param: str) -> int:
    """
    Brief description.

    Args:
        param: Description of parameter

    Returns:
        Description of return value

    Example:
        >>> new_function("test")
        42
    """
    return 42
```

2. Update README.md if needed
3. Add examples to `examples/` if it's a new feature

## 🔧 Development Tips

### Recommended IDE Setup

**VSCode:**
```json
{
  "python.formatting.provider": "black",
  "python.linting.flake8Enabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Skip hooks (NOT RECOMMENDED)
git commit --no-verify
```

### Debugging CI Failures

If CI fails:

1. **Check the CI logs** on GitHub
2. **Reproduce locally**:
   ```bash
   # Run the same commands CI runs
   uv run black --check .
   uv run isort --check-only .
   uv run flake8 anomavision/ anodet/ apps/ tests/
   uv run pytest -v
   ```
3. **Fix issues**:
   ```bash
   uv run black .
   uv run isort .
   git add .
   git commit -m "fix: resolve code quality issues"
   git push
   ```

## 📞 Getting Help

- **Questions**: [GitHub Discussions](https://github.com/DeepKnowledge1/AnomaVision/discussions)
- **Bugs**: [Bug Report](https://github.com/DeepKnowledge1/AnomaVision/issues/new?template=bug-report.yml)
- **Features**: [Feature Request](https://github.com/DeepKnowledge1/AnomaVision/issues/new?template=feature-request.yml)

## 🎉 Recognition

Contributors are recognized in:
- GitHub Contributors page
- Release notes (for significant contributions)
- README acknowledgments (for major features)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to AnomaVision! 🚀**
