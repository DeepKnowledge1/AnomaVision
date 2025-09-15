# Contributing to AnomaVision

Thank you for your interest in contributing to AnomaVision! 🎉 We welcome contributions from the community and are pleased to have you join us.

## 🌟 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- 🐛 **Bug Reports**: Help us identify and fix issues
- 💡 **Feature Requests**: Suggest new features or enhancements
- 🔧 **Code Contributions**: Submit bug fixes or new features
- 📚 **Documentation**: Improve documentation, tutorials, or examples
- 🧪 **Testing**: Add or improve test coverage
- 🎨 **Examples**: Create new examples or improve existing ones

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/AnomaVision.git
cd AnomaVision
```

### 2. Set Up Development Environment

#### Option A: Using Poetry (Recommended)
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies and activate virtual environment
poetry install --dev
poetry shell
```

#### Option B: Using pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run automatically before each commit.

### 4. Verify Setup

```bash
# Run tests to ensure everything is working
pytest tests/ -v

# Check code formatting
black --check anodet tests

# Run linting
flake8 anodet
```

## 🔄 Development Workflow

### 1. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### Branch Naming Conventions:
- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/documentation-update` - Documentation changes
- `test/test-improvements` - Test additions/improvements
- `refactor/code-improvement` - Code refactoring

### 2. Make Your Changes

#### Code Style Guidelines

- **Python Style**: Follow PEP 8 with 88-character line limit (Black formatting)
- **Type Hints**: Add type hints to all new functions and methods
- **Docstrings**: Use Google-style docstrings for all public functions
- **Imports**: Use absolute imports, group imports (stdlib, third-party, local)

#### Example Code Style:

```python
from typing import Optional, Tuple

import torch
import numpy as np

from anodet.utils import get_logger


def process_batch(
    batch: torch.Tensor,
    threshold: float = 10.0,
    device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process a batch of images for anomaly detection.

    Args:
        batch: Input batch tensor of shape (B, C, H, W).
        threshold: Anomaly detection threshold.
        device: Device to run computation on.

    Returns:
        Tuple of (image_scores, score_maps).

    Raises:
        ValueError: If batch has invalid shape.
    """
    if batch.dim() != 4:
        raise ValueError(f"Expected 4D batch, got {batch.dim()}D")

    # Implementation here...
    return image_scores, score_maps
```

### 3. Add Tests

All new code should include tests:

```bash
# Create test file
touch tests/test_your_feature.py
```

#### Test Guidelines:
- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test complete workflows
- **Parametrized Tests**: Use `@pytest.mark.parametrize` for multiple test cases
- **Fixtures**: Use fixtures for common test data
- **Coverage**: Aim for >90% test coverage

#### Example Test:

```python
import pytest
import torch
from anodet.your_module import your_function


class TestYourFunction:
    """Test suite for your_function."""

    @pytest.mark.parametrize("input_size,expected_output", [
        ((2, 3, 224, 224), (2,)),
        ((1, 3, 224, 224), (1,)),
    ])
    def test_function_with_different_inputs(self, input_size, expected_output):
        """Test function with different input sizes."""
        batch = torch.randn(input_size)
        result = your_function(batch)
        assert result.shape == expected_output

    def test_function_error_handling(self):
        """Test function handles errors correctly."""
        with pytest.raises(ValueError, match="Expected 4D batch"):
            your_function(torch.randn(3, 224, 224))  # Missing batch dimension
```

### 4. Update Documentation

#### Code Documentation:
- Add docstrings to all new functions and classes
- Update existing docstrings if you modify function signatures
- Include examples in docstrings when helpful

#### README Updates:
- Update README.md if you add new features
- Add new dependencies to installation instructions
- Update API reference section if needed

#### Example Documentation:
- Add examples to `examples/` directory
- Include Jupyter notebooks for complex features
- Add visualization examples for new visualization functions

### 5. Run Quality Checks

Before committing, ensure all checks pass:

```bash
# Format code
black anodet tests

# Run linting
flake8 anodet

# Type checking
mypy anodet --ignore-missing-imports

# Run tests
pytest tests/ -v --cov=anodet

# Check test coverage
pytest tests/ --cov=anodet --cov-report=html
open htmlcov/index.html  # View coverage report
```

### 6. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add support for custom image dimensions

- Implement flexible resize and crop parameters
- Add tests for different image dimensions
- Update documentation with examples
- Fixes #123"
```

#### Commit Message Format:
```
type(scope): short description

Longer explanation if needed

- Bullet points for key changes
- Reference issues: Fixes #123, Closes #456
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `style`: Code style changes
- `chore`: Maintenance tasks

## 📝 Pull Request Process

### 1. Push Your Branch

```bash
git push origin feature/your-feature-name
```

### 2. Create Pull Request

1. Go to the [AnomaVision repository](https://github.com/DeepKnowledge1/AnomaVision)
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template

### 3. Pull Request Template

Your PR should include:

- **Description**: Clear description of changes
- **Type of Change**: Bug fix, new feature, documentation, etc.
- **Testing**: How you tested your changes
- **Checklist**: Completed items from the checklist
- **Screenshots**: If applicable (especially for visualizations)

### 4. Code Review Process

- All PRs require at least one review from a maintainer
- Address reviewer feedback promptly
- Keep PRs focused and reasonably sized
- Be responsive to comments and suggestions

### 5. CI Checks

All PRs must pass:
- ✅ **Tests**: All tests must pass
- ✅ **Linting**: Code must pass flake8 checks
- ✅ **Formatting**: Code must be formatted with Black
- ✅ **Coverage**: Test coverage should not decrease significantly

## 🐛 Reporting Bugs

When reporting bugs, please include:

### Bug Report Template:

```markdown
**Describe the Bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With parameters '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 1.13.1]
- CUDA version: [e.g., 11.7]
- AnomaVision version: [e.g., 2.0.46]

**Additional Context**
- Full error traceback
- Sample data if relevant
- Screenshots if applicable
```

### Before Submitting a Bug Report:

1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if the issue is already fixed
3. **Minimal reproduction** - provide the smallest example that reproduces the issue
4. **Check dependencies** - ensure all requirements are met

## 💡 Suggesting Features

When suggesting new features:

### Feature Request Template:

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the problem this feature would solve.

**Proposed Solution**
Describe how you envision this feature working.

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Any other context, mockups, or examples.
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_padim.py -v

# Run with coverage
pytest tests/ --cov=anodet --cov-report=html

# Run tests in parallel
pytest tests/ -n auto

# Run only failed tests from last run
pytest --lf
```

### Writing Tests

#### Test Organization:
```
tests/
├── conftest.py              # Shared fixtures
├── test_padim.py            # Core PaDiM tests
├── test_backends.py         # Backend tests
├── test_datasets.py         # Dataset tests
├── test_visualization.py    # Visualization tests
├── test_export.py           # Export functionality tests
└── integration/             # Integration tests
    ├── test_training_pipeline.py
    └── test_inference_pipeline.py
```

#### Test Best Practices:
- **Descriptive names**: `test_padim_training_with_custom_dimensions()`
- **Single responsibility**: Each test should test one thing
- **Independent tests**: Tests should not depend on each other
- **Use fixtures**: Reuse common test data and setup
- **Mock external dependencies**: Don't rely on external services
- **Test edge cases**: Empty inputs, invalid parameters, etc.

#### Example Test Structure:

```python
class TestFeatureName:
    """Test suite for feature_name functionality."""

    def test_normal_case(self, fixture_name):
        """Test the normal, expected use case."""
        # Arrange
        input_data = create_test_input()

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result.shape == expected_shape
        assert result.dtype == expected_dtype

    def test_edge_case(self):
        """Test edge cases and boundary conditions."""
        # Test with empty input, None values, etc.
        pass

    def test_error_handling(self):
        """Test that errors are handled correctly."""
        with pytest.raises(ValueError, match="Expected error message"):
            function_under_test(invalid_input)
```

## 📚 Documentation Guidelines

### Code Documentation

#### Docstring Format (Google Style):

```python
def complex_function(
    param1: torch.Tensor,
    param2: Optional[str] = None,
    param3: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Brief description of what the function does.

    Longer description if needed. Explain the algorithm, important
    details, or any non-obvious behavior.

    Args:
        param1: Description of param1. Include shape info for tensors.
        param2: Description of param2. Mention default behavior.
        param3: Description of param3.

    Returns:
        Tuple containing:
            - tensor_result: Description of first return value
            - dict_result: Description of second return value

    Raises:
        ValueError: When param1 has invalid shape.
        RuntimeError: When computation fails.

    Example:
        >>> import torch
        >>> x = torch.randn(2, 3, 224, 224)
        >>> result, info = complex_function(x, param2="test")
        >>> print(result.shape)
        torch.Size([2, 10])

    Note:
        This function modifies the input tensor in-place.
    """
```

### README Updates

When adding features, update relevant sections:
- **Installation** - new dependencies
- **Quick Start** - if API changes
- **Examples** - add new examples
- **API Reference** - document new functions
- **Performance** - update benchmarks if applicable

### Example Documentation

Create examples for new features:

```python
# examples/new_feature_example.py
"""
Example demonstrating the new feature.

This example shows how to use the new feature for anomaly detection
on custom datasets with flexible image dimensions.
"""
import torch
from torch.utils.data import DataLoader
import anodet

def main():
    """Demonstrate new feature usage."""
    # Step 1: Setup
    dataset = anomavision.AnodetDataset(
        "path/to/images",
        resize=[320, 240],  # New flexible dimensions feature
        crop_size=[224, 224]
    )

    # Step 2: Train
    model = anodet.Padim(backbone='resnet18')
    model.fit(DataLoader(dataset, batch_size=4))

    # Step 3: Detect
    scores, maps = model.predict(batch)
    print(f"Anomaly scores: {scores}")

if __name__ == "__main__":
    main()
```

## 🔧 Development Environment

### IDE Configuration

#### VS Code Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        "htmlcov": true
    }
}
```

### Debugging

#### Common Issues:

1. **Import Errors**:
   ```bash
   # Ensure you're in the right environment
   which python
   pip list | grep torch
   ```

2. **CUDA Issues**:
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Test Failures**:
   ```bash
   # Run specific failing test with verbose output
   pytest tests/test_file.py::test_function -v -s
   ```

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 2.1.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

For maintainers preparing releases:

1. **Update Version Numbers**:
   - `pyproject.toml`
   - `anodet/__init__.py`
   - Documentation

2. **Update CHANGELOG.md**:
   - Document all changes since last release
   - Group by type (Added, Changed, Fixed, Removed)

3. **Run Full Test Suite**:
   ```bash
   pytest tests/ -v --cov=anodet
   ```

4. **Create Release**:
   - Tag the release: `git tag v2.1.0`
   - Push tag: `git push origin v2.1.0`
   - Create GitHub release with changelog

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful** - Treat everyone with respect and courtesy
- **Be inclusive** - Welcome newcomers and help them learn
- **Be constructive** - Provide helpful feedback and suggestions
- **Be patient** - Remember that everyone has different experience levels

### Communication

- **GitHub Issues** - For bug reports and feature requests
- **GitHub Discussions** - For questions and general discussion
- **Pull Requests** - For code review and collaboration
- **Discord** - For real-time chat (coming soon)

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributors page

## 📞 Getting Help

If you need help with contributing:

1. **Read the Documentation** - Check README and existing docs
2. **Search Issues** - Someone might have had the same question
3. **Ask in Discussions** - GitHub Discussions for questions
4. **Contact Maintainers** - Email for sensitive issues

### Maintainers

Current maintainers:
- **Core Team**: [@DeepKnowledge1](https://github.com/DeepKnowledge1)

## 🙏 Thank You

Thank you for contributing to AnomaVision! Your contributions help make this project better for everyone. Every contribution, no matter how small, is valued and appreciated.

---

**Happy Contributing! 🚀**
