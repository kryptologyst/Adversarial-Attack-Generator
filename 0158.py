# Project 158. Adversarial attack generator
# Description:
# Adversarial attacks involve subtly modifying input data to fool a machine learning model. These changes are often imperceptible to humans but can cause incorrect predictions. In this project, we implement a basic Fast Gradient Sign Method (FGSM) attack to generate adversarial examples that trick a trained classifier (e.g., MNIST CNN).

# Python Implementation: FGSM Adversarial Attack on MNIST CNN
# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
 
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# Load MNIST
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
 
# Simple CNN for MNIST (already trained or just for demo)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
 
    def forward(self, x):
        return self.net(x)
 
# Load model (check if pre-trained model exists, otherwise train a quick one)
model = SimpleCNN().to(device)

# Check if pre-trained model exists
model_path = "mnist_cnn.pth"
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Loaded pre-trained model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Training a new model...")
        # Quick training for demo
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(3):  # Quick training
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print("💾 Model saved")
else:
    print("🔄 Training a new model...")
    # Quick training for demo
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(3):  # Quick training
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print("💾 Model saved")

model.eval()
 
# FGSM Attack Function
def fgsm_attack(image, label, epsilon):
    image = image.clone().detach().to(device)
    label = label.clone().detach().to(device)
    image.requires_grad = True
    
    output = model(image)
    loss = F.cross_entropy(output, label)  # Use cross_entropy instead of nll_loss
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image.detach()
 
# Run a single adversarial attack
epsilon = 0.25
print(f"\n🎯 Running FGSM attack with epsilon = {epsilon}")
print("=" * 50)

for i, (data, target) in enumerate(test_loader):
    data, target = data.to(device), target.to(device)
    
    with torch.no_grad():
        original_pred = model(data).argmax(dim=1).item()
        original_confidence = F.softmax(model(data), dim=1).max().item()
 
    adv_data = fgsm_attack(data, target, epsilon)
    
    with torch.no_grad():
        adv_pred = model(adv_data).argmax(dim=1).item()
        adv_confidence = F.softmax(model(adv_data), dim=1).max().item()
 
    print(f"📊 Sample {i+1}:")
    print(f"   True Label: {target.item()}")
    print(f"   Original Prediction: {original_pred} (Confidence: {original_confidence:.3f})")
    print(f"   Adversarial Prediction: {adv_pred} (Confidence: {adv_confidence:.3f})")
    print(f"   Attack Success: {'✅' if original_pred != adv_pred else '❌'}")
    print()
 
    # Display original vs adversarial image
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Original\nPred: {original_pred} ({original_confidence:.3f})")
    plt.imshow(data.squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")
 
    plt.subplot(1, 3, 2)
    plt.title(f"Adversarial\nPred: {adv_pred} ({adv_confidence:.3f})")
    plt.imshow(adv_data.squeeze().cpu().numpy(), cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    perturbation = (adv_data - data).squeeze().cpu().numpy()
    plt.title("Perturbation")
    plt.imshow(perturbation, cmap="RdBu", vmin=-epsilon, vmax=epsilon)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    if i >= 2:  # Show only first 3 samples
        break

print("\n🎉 Demo completed! Check out the modern implementation:")
print("   - Run 'python train.py' to train a modern model")
print("   - Run 'streamlit run app.py' for the interactive web interface")



# 🧠 What This Project Demonstrates:
# Creates adversarial examples using FGSM (Fast Gradient Sign Method)

# Shows how small pixel perturbations can fool a classifier

# Visualizes and compares original vs adversarial images