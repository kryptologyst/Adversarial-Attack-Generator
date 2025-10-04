# Contributing to Adversarial Attack Generator

Thank you for your interest in contributing to the Adversarial Attack Generator! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of machine learning and adversarial attacks

### Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/your-username/adversarial-attack-generator.git
cd adversarial-attack-generator
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install development dependencies:**
```bash
pip install -r requirements-dev.txt
```

4. **Install pre-commit hooks:**
```bash
pre-commit install
```

## 📝 Code Style

### Python Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run these tools before committing:

```bash
black .
isort .
flake8 .
mypy .
```

### Documentation Style

- Use **Google-style docstrings** for functions and classes
- Include **type hints** for all function parameters and return values
- Write **clear, concise comments** for complex logic
- Update **README.md** for new features

## 🔧 Development Workflow

### 1. Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Making Changes

- Write **focused commits** with clear messages
- Add **tests** for new functionality
- Update **documentation** as needed
- Follow **existing code patterns**

### 3. Testing

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_attacks.py -v
```

### 4. Submitting Changes

1. **Push your branch:**
```bash
git push origin feature/your-feature-name
```

2. **Create a Pull Request:**
   - Provide a clear description
   - Reference any related issues
   - Include screenshots for UI changes

## 🧪 Adding New Features

### New Attack Methods

To add a new attack method:

1. **Implement the attack** in `attacks/adversarial_attacks.py`:
```python
def new_attack_method(self, image, label, **kwargs):
    """New attack method implementation."""
    # Your implementation here
    return adversarial_image
```

2. **Add configuration** in `config.yaml`:
```yaml
attack:
  new_method_param: 0.1
```

3. **Update the web interface** in `app.py`:
```python
elif method == 'new_method':
    adv_image = attacks.new_attack_method(image, label, **config)
```

4. **Add tests** in `tests/test_attacks.py`:
```python
def test_new_attack_method():
    """Test new attack method."""
    # Your tests here
```

### New Model Architectures

To add a new model architecture:

1. **Implement the model** in `models/cnn_model.py`:
```python
class NewModel(nn.Module):
    """New model architecture."""
    def __init__(self):
        # Your implementation here
        pass
```

2. **Update the factory function**:
```python
def create_model(model_type='new_model', **kwargs):
    if model_type == 'new_model':
        return NewModel(**kwargs)
```

3. **Add configuration options** and **update documentation**

## 🐛 Bug Reports

When reporting bugs, please include:

- **Python version** and **operating system**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Error messages** and **stack traces**
- **Minimal code example** if applicable

## 💡 Feature Requests

For feature requests, please:

- **Describe the feature** clearly
- **Explain the use case** and benefits
- **Provide examples** if possible
- **Consider implementation complexity**

## 📊 Testing Guidelines

### Unit Tests

- Test **individual functions** and **methods**
- Use **mock objects** for external dependencies
- Aim for **high code coverage**
- Test **edge cases** and **error conditions**

### Integration Tests

- Test **end-to-end workflows**
- Test **web interface** functionality
- Test **model training** and **attack generation**

### Test Structure

```
tests/
├── test_models.py          # Model architecture tests
├── test_attacks.py         # Attack method tests
├── test_config.py          # Configuration tests
├── test_ui.py              # Web interface tests
└── fixtures/               # Test data and fixtures
```

## 📚 Documentation

### Code Documentation

- **Docstrings** for all public functions and classes
- **Type hints** for better IDE support
- **Inline comments** for complex logic

### User Documentation

- **README.md** updates for new features
- **API documentation** for new functions
- **Example notebooks** for complex workflows

## 🔒 Security Considerations

When contributing:

- **Never commit** API keys or sensitive data
- **Validate user inputs** in web interfaces
- **Use secure defaults** for configurations
- **Consider attack implications** of new features

## 🏷️ Release Process

### Version Numbering

We use **Semantic Versioning** (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] Update version in `setup.py`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release tag
- [ ] Publish to PyPI (if applicable)

## 🤝 Community Guidelines

### Code of Conduct

- **Be respectful** and **inclusive**
- **Provide constructive feedback**
- **Help others** learn and grow
- **Follow community standards**

### Communication

- **Use clear, professional language**
- **Be patient** with questions
- **Provide context** in discussions
- **Stay on topic**

## 📞 Getting Help

- **GitHub Issues** for bug reports and feature requests
- **GitHub Discussions** for questions and ideas
- **Pull Request comments** for code review discussions

## 🎯 Areas for Contribution

We especially welcome contributions in:

- **New attack methods** and **defense mechanisms**
- **Performance optimizations**
- **Additional datasets** support
- **Web interface improvements**
- **Documentation** and **examples**
- **Testing** and **code quality**

Thank you for contributing to the Adversarial Attack Generator! 🎯
