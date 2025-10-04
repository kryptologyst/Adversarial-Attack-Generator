# Adversarial Attack Generator

A comprehensive tool for generating and analyzing adversarial examples using state-of-the-art attack methods. This project provides both a powerful Python API and an intuitive web interface for researchers, developers, and security professionals.

## Features

### Advanced Attack Methods
- **FGSM (Fast Gradient Sign Method)** - Fast and effective white-box attack
- **PGD (Projected Gradient Descent)** - Iterative attack with projection
- **BIM (Basic Iterative Method)** - Iterative version of FGSM
- **DeepFool** - Minimal perturbation attack
- **C&W (Carlini & Wagner)** - Strong optimization-based attack

### Modern Architecture
- **Modern CNN** with batch normalization, dropout, and residual connections
- **Simple CNN** for quick experimentation
- **Configurable hyperparameters** and model architectures
- **Automatic model saving and loading**

### Interactive Web Interface
- **Streamlit-based UI** with real-time visualization
- **Interactive attack generation** with customizable parameters
- **Comprehensive statistics dashboard**
- **Model training interface**
- **Attack success rate analysis**

### Comprehensive Evaluation
- **Attack success rate calculation**
- **Robustness evaluation metrics**
- **Visual comparison of original vs adversarial images**
- **Confidence score analysis**
- **Database storage for attack results**

### Configuration Management
- **YAML-based configuration**
- **Environment-specific settings**
- **Runtime parameter adjustment**
- **Configuration validation**

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/kryptologyst/Adversarial-Attack-Generator.git
cd Adversarial-Attack-Generator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train a model:**
```bash
python train.py
```

4. **Launch the web interface:**
```bash
streamlit run app.py
```

### Basic Usage

#### Python API

```python
from models.cnn_model import create_model, load_pretrained_model
from attacks.adversarial_attacks import AdversarialAttacks
import torch

# Load a pre-trained model
model = load_pretrained_model('checkpoints/best_model.pth', 'modern', 'cpu')

# Initialize attack generator
attacks = AdversarialAttacks(model, 'cpu')

# Generate adversarial example
image = torch.randn(1, 1, 28, 28)  # Your input image
label = torch.tensor([5])  # True label

# FGSM attack
adversarial_image = attacks.fgsm_attack(image, label, epsilon=0.25)

# Check predictions
original_pred = model(image).argmax(dim=1)
adversarial_pred = model(adversarial_image).argmax(dim=1)

print(f"Original prediction: {original_pred.item()}")
print(f"Adversarial prediction: {adversarial_pred.item()}")
```

#### Web Interface

1. Open your browser and navigate to `http://localhost:8501`
2. Go to the "Model Training" tab to train a new model
3. Use the "Attack Generator" tab to create adversarial examples
4. View statistics in the "Statistics" dashboard

## 📁 Project Structure

```
adversarial-attack-generator/
├── 📁 models/                 # Model architectures
│   └── cnn_model.py          # CNN implementations
├── 📁 attacks/               # Attack methods
│   └── adversarial_attacks.py # Attack implementations
├── 📁 config/                # Configuration management
│   └── config_manager.py     # Config handling
├── 📁 data/                  # Data storage (auto-created)
├── 📁 checkpoints/           # Model checkpoints (auto-created)
├── 📄 app.py                 # Streamlit web application
├── 📄 train.py               # Model training script
├── 📄 0158.py                # Original implementation
├── 📄 config.yaml            # Configuration file
├── 📄 requirements.txt       # Python dependencies
├── 📄 setup.py               # Package setup
└── 📄 README.md              # This file
```

## 🔧 Configuration

The application uses YAML configuration files for easy customization:

```yaml
# config.yaml
model:
  model_type: "modern"
  learning_rate: 0.001
  batch_size: 64
  num_epochs: 10

attack:
  fgsm_epsilon: 0.25
  pgd_epsilon: 0.25
  pgd_num_iter: 40

ui:
  page_title: "Adversarial Attack Generator"
  theme: "light"
```

## Attack Methods Comparison

| Method | Speed | Effectiveness | Perturbation Size | Use Case |
|--------|-------|---------------|-------------------|----------|
| FGSM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Quick testing |
| PGD | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | Strong attacks |
| BIM | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Balanced approach |
| DeepFool | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Minimal perturbation |
| C&W | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Strongest attacks |

## Examples

### Training a Robust Model

```python
from train import Trainer

# Initialize trainer with modern architecture
trainer = Trainer(
    model_type='modern',
    learning_rate=0.001,
    batch_size=64,
    num_epochs=10
)

# Train the model
history = trainer.train()

# Evaluate robustness
robustness_results = trainer.evaluate_adversarial_robustness()
```

### Batch Attack Generation

```python
from attacks.adversarial_attacks import evaluate_attack_robustness

# Evaluate model robustness
results = evaluate_attack_robustness(
    model=model,
    test_loader=test_loader,
    epsilon_values=[0.1, 0.2, 0.3, 0.4, 0.5]
)

print(f"FGSM success rate at ε=0.3: {results['fgsm'][0.3]['success_rate']:.2%}")
```

## Research Applications

This tool is designed for:

- **Security Research** - Evaluating model robustness
- **Adversarial Training** - Improving model defenses
- **Attack Development** - Testing new attack methods
- **Education** - Learning about adversarial attacks
- **Benchmarking** - Comparing model robustness

## Performance Metrics

The tool provides comprehensive evaluation metrics:

- **Attack Success Rate** - Percentage of successful attacks
- **Accuracy Drop** - Reduction in model accuracy
- **Confidence Analysis** - Prediction confidence changes
- **Perturbation Analysis** - Visual and statistical perturbation metrics
- **Robustness Score** - Overall model robustness assessment

## Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black .

# Type checking
mypy .
```

### Adding New Attack Methods

1. Implement the attack in `attacks/adversarial_attacks.py`
2. Add configuration parameters in `config.yaml`
3. Update the web interface in `app.py`
4. Add tests and documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## References

- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples.
- Madry, A., et al. (2017). Towards deep learning models resistant to adversarial attacks.
- Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks.
- Moosavi-Dezfooli, S. M., et al. (2016). DeepFool: a simple and accurate method to fool deep neural networks.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Additional attack methods (JSMA, SPSA)
- [ ] Defense mechanisms (adversarial training, defensive distillation)
- [ ] Support for other datasets (CIFAR-10, ImageNet)
- [ ] Real-time attack visualization
- [ ] Model comparison tools
- [ ] API endpoints for integration
- [ ] Docker containerization
- [ ] Cloud deployment options


# Adversarial-Attack-Generator
