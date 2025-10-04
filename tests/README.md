# Adversarial Attack Generator - Tests

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_attacks.py

# Run with verbose output
pytest -v

# Run tests in parallel
pytest -n auto
```

## Test Structure

```
tests/
├── test_models.py          # Model architecture tests
├── test_attacks.py         # Attack method tests
├── test_config.py          # Configuration tests
├── test_ui.py              # Web interface tests
├── test_train.py           # Training script tests
└── fixtures/               # Test data and fixtures
    ├── sample_images/
    └── test_configs/
```

## Writing Tests

### Example Test Structure

```python
import pytest
import torch
from models.cnn_model import create_model
from attacks.adversarial_attacks import AdversarialAttacks

class TestAdversarialAttacks:
    def setup_method(self):
        """Setup for each test method."""
        self.model = create_model('simple')
        self.attacks = AdversarialAttacks(self.model, 'cpu')
        self.sample_image = torch.randn(1, 1, 28, 28)
        self.sample_label = torch.tensor([5])
    
    def test_fgsm_attack(self):
        """Test FGSM attack functionality."""
        epsilon = 0.25
        adv_image = self.attacks.fgsm_attack(
            self.sample_image, 
            self.sample_label, 
            epsilon
        )
        
        # Check output shape
        assert adv_image.shape == self.sample_image.shape
        
        # Check pixel range
        assert torch.all(adv_image >= 0)
        assert torch.all(adv_image <= 1)
        
        # Check perturbation magnitude
        perturbation = torch.abs(adv_image - self.sample_image)
        assert torch.all(perturbation <= epsilon + 1e-6)
    
    def test_pgd_attack(self):
        """Test PGD attack functionality."""
        epsilon = 0.25
        alpha = 0.01
        num_iter = 10
        
        adv_image = self.attacks.pgd_attack(
            self.sample_image,
            self.sample_label,
            epsilon,
            alpha,
            num_iter
        )
        
        # Check output shape
        assert adv_image.shape == self.sample_image.shape
        
        # Check pixel range
        assert torch.all(adv_image >= 0)
        assert torch.all(adv_image <= 1)
    
    def test_attack_success_rate(self):
        """Test attack success rate calculation."""
        # Create a simple test dataset
        test_data = [(self.sample_image, self.sample_label)]
        
        success_rate, accuracy_drop = self.attacks.attack_success_rate(
            test_data, 'fgsm', 0.25
        )
        
        assert 0 <= success_rate <= 1
        assert accuracy_drop >= 0
```

### Fixtures

```python
# conftest.py
import pytest
import torch
from torchvision import datasets, transforms

@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    from models.cnn_model import create_model
    return create_model('simple')

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    image = torch.randn(1, 1, 28, 28)
    label = torch.tensor([5])
    return image, label

@pytest.fixture
def mnist_dataset():
    """Create a small MNIST dataset for testing."""
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root='./test_data', 
        train=False, 
        download=True, 
        transform=transform
    )
    return dataset
```

## Test Categories

### Unit Tests
- Test individual functions and methods
- Use mock objects for external dependencies
- Test edge cases and error conditions

### Integration Tests
- Test end-to-end workflows
- Test component interactions
- Test with real data

### Performance Tests
- Test attack generation speed
- Test memory usage
- Test scalability

## Continuous Integration

The project includes GitHub Actions for automated testing:

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
```

## Test Data

Test data is stored in the `tests/fixtures/` directory:

- **Sample images**: Pre-generated test images
- **Model weights**: Pre-trained models for testing
- **Configuration files**: Test configurations
- **Expected outputs**: Ground truth for comparison

## Coverage Requirements

The project aims for:
- **90%+ code coverage** for core modules
- **80%+ coverage** for UI components
- **100% coverage** for critical attack methods

## Debugging Tests

```bash
# Run tests with debugging
pytest --pdb

# Run specific test with debugging
pytest tests/test_attacks.py::TestAdversarialAttacks::test_fgsm_attack --pdb

# Run tests with detailed output
pytest -vvv

# Run tests and stop on first failure
pytest -x
```
