# Contributing to LLM-Heuristics

We welcome contributions! This document provides guidelines for contributing to the project.

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/NeuroHackademy2025/llm-heuristics.git
cd llm-heuristics

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e .[dev,test]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_dicom_analyzer.py
```

### Code Quality

```bash
# Format code
ruff check llm_heuristics/ --fix
ruff format llm_heuristics/
```

## Contributing Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite** to ensure everything works
6. **Submit a pull request**

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable and function names

### Testing

- Write tests for all new functionality
- Ensure existing tests continue to pass
- Use descriptive test names

### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Update CHANGELOG.md** with your changes
4. **Ensure all tests pass**
5. **Request review** from maintainers

### Reporting Issues

When reporting issues, please include:

- **Operating system** and version
- **Python version**
- **LLM-Heuristics version**
- **Detailed description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**

### Feature Requests

For feature requests, please:

- **Describe the feature** in detail
- **Explain the use case** and benefits
- **Consider implementation complexity**
- **Check if similar features exist**

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/NeuroHackademy2025/llm-heuristics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NeuroHackademy2025/llm-heuristics/discussions)

Thank you for contributing to LLM-Heuristics! 