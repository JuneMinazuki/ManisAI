import torch
import torchvision.models as models

# Load MobileNetV3-Large pretrained on ImageNet
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)

# Load MobileNetV3-Small pretrained on ImageNet
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)

import torch.nn as nn
# Modify the final layer for a custom number of classes (e.g., 8)
mobilenet_v3_large.classifier[8] = nn.Linear(in_features=1280, out_features=10)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(root='train_data_path/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

import torch.optim as optim
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet_v3_large.parameters(), lr=0.001)
# Training loop
num_epochs = 5
mobilenet_v3_large.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = mobilenet_v3_large(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    mobilenet_v3_large.eval()  # Set the model to evaluation mode

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = mobilenet_v3_large(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')