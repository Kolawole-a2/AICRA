# Contributing to AICRA

Thank you for your interest in contributing to AICRA! This guide will help you get started with contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Make (optional, for using Makefile commands)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AICRA
   ```

2. **Install dependencies**
   ```bash
   make setup
   ```
   
   Or manually:
   ```bash
   pip install -e .
   pip install -r requirements/dev.txt
   pre-commit install
   ```

3. **Verify installation**
   ```bash
   make test
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the coding standards and conventions outlined below.

### 3. Run Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=aicra --cov-report=html
```

### 4. Run Linters

```bash
# Run all linters
make lint

# Run specific linter
ruff check aicra/
black --check aicra/
isort --check-only aicra/
mypy aicra/
pylint aicra/
```

### 5. Format Code

```bash
# Format all code
make format

# Or manually
black aicra/ tests/
isort aicra/ tests/
ruff check --fix aicra/ tests/
```

### 6. Run Security Checks

```bash
# Run security audits
make audit

# Or manually
pip-audit
safety check
```

### 7. Commit Changes

```bash
git add .
git commit -m "feat: add new feature"
```

### 8. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style

We use the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting
- **ruff** for linting
- **mypy** for type checking
- **pylint** for additional code analysis

### Type Hints

All functions and methods must have type hints:

```python
def process_data(data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    """Process data and return metrics."""
    # Implementation
    return {"metric": 0.95}
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(
    train_data: Dataset, 
    val_data: Dataset, 
    config: Dict[str, Any]
) -> Model:
    """Train a machine learning model.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        config: Training configuration
        
    Returns:
        Trained model
        
    Raises:
        ValueError: If data is invalid
    """
    # Implementation
```

### Error Handling

Use specific exception types and provide meaningful error messages:

```python
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found: {data_path}")

if len(data) == 0:
    raise ValueError("Dataset is empty")
```

### Logging

Use the standard logging module:

```python
import logging

logger = logging.getLogger(__name__)

def process_file(file_path: Path) -> None:
    logger.info(f"Processing file: {file_path}")
    try:
        # Processing logic
        logger.info("File processed successfully")
    except Exception as e:
        logger.error(f"Failed to process file: {e}")
        raise
```

## Testing Guidelines

### Test Structure

- Place tests in the `tests/` directory
- Use descriptive test names
- Group related tests in classes
- Use fixtures for common setup

### Test Naming

```python
def test_model_training_with_valid_data():
    """Test that model training works with valid data."""
    # Test implementation

def test_model_training_raises_error_with_empty_data():
    """Test that model training raises error with empty data."""
    # Test implementation
```

### Test Coverage

- Aim for 85%+ code coverage
- Test both success and failure cases
- Include edge cases and boundary conditions
- Mock external dependencies

### Example Test

```python
import pytest
from unittest.mock import Mock, patch
from aicra.pipelines.training import TrainingPipeline

class TestTrainingPipeline:
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = Mock()
        self.pipeline = TrainingPipeline(self.settings)
    
    def test_train_model_with_valid_data(self):
        """Test model training with valid data."""
        # Arrange
        train_data = Mock()
        val_data = Mock()
        
        # Act
        model = self.pipeline.train_model(train_data, val_data)
        
        # Assert
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_train_model_raises_error_with_invalid_data(self):
        """Test that training raises error with invalid data."""
        # Arrange
        train_data = None
        val_data = Mock()
        
        # Act & Assert
        with pytest.raises(ValueError, match="Training data is required"):
            self.pipeline.train_model(train_data, val_data)
```

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

The hooks include:
- Code formatting (Black, isort)
- Linting (ruff, pylint)
- Type checking (mypy)
- Security checks (detect-secrets)
- File checks (trailing whitespace, end-of-file)

## Pull Request Guidelines

### Before Submitting

1. **Run all checks**
   ```bash
   make lint
   make typecheck
   make test
   make audit
   ```

2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update changelog** if applicable

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Coverage maintained

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in different environments
4. **Documentation** review

## Release Process

### Versioning

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Update version** in `pyproject.toml`
2. **Update changelog**
3. **Create release branch**
4. **Run full test suite**
5. **Create GitHub release**
6. **Update documentation**

## Documentation

### Code Documentation

- Use docstrings for all public functions
- Include type hints
- Provide examples for complex functions
- Document configuration options

### User Documentation

- Update `README.md` for new features
- Add examples to `USAGE.md`
- Update `CONTRIBUTING.md` if needed

### API Documentation

- Document all public APIs
- Include parameter descriptions
- Provide usage examples
- Document return values

## Performance Guidelines

### Code Performance

- Use vectorized operations when possible
- Avoid unnecessary loops
- Use appropriate data structures
- Profile code for bottlenecks

### Memory Management

- Use generators for large datasets
- Clean up resources properly
- Monitor memory usage
- Use appropriate data types

### Example

```python
# Good: Vectorized operation
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate metrics using vectorized operations."""
    accuracy = np.mean(y_true == y_pred)
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    return {"accuracy": accuracy, "precision": precision}

# Bad: Loop-based operation
def calculate_metrics_slow(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate metrics using loops (inefficient)."""
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    accuracy = correct / len(y_true)
    return {"accuracy": accuracy}
```

## Security Guidelines

### Code Security

- Never commit secrets or API keys
- Use environment variables for sensitive data
- Validate all inputs
- Use secure random number generation

### Dependency Security

- Keep dependencies updated
- Use `pip-audit` to check for vulnerabilities
- Use `safety` for additional security checks
- Pin dependency versions

### Example

```python
import os
import secrets
from pathlib import Path

# Good: Use environment variables
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is required")

# Good: Use secure random
random_token = secrets.token_urlsafe(32)

# Good: Validate inputs
def process_file(file_path: str) -> None:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
```

## Getting Help

### Resources

- **Documentation**: Check `README.md` and `USAGE.md`
- **Issues**: Search existing GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Ask for help in pull requests

### Contact

- **Email**: team@aicra.com
- **GitHub**: Create an issue or discussion
- **Slack**: Join our community channel

## License

By contributing to AICRA, you agree that your contributions will be licensed under the same license as the project.
