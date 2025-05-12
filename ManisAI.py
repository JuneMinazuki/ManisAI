import torch
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights

#Dataset Directory
train_cd = 'Training/'
test_cd = 'Testing/'

#Parameter
learning_rate = 0.001
num_epochs = 50
batch_size = 64
step_size = 10
gamma = 0.1
weight_decay = 1e-4

# Load MobileNetV3-Large pretrained on ImageNet
mobilenet_v3_large = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

import torch.nn as nn
# Modify the final layer for a custom number of classes
mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=8)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define data transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder(root=train_cd, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Get the class names from your dataset's classes attribute
class_names = train_dataset.classes
kuih_dict = {i + 1: class_name for i, class_name in enumerate(class_names)}

import torch.optim as optim
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet_v3_large.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training loop
mobilenet_v3_large.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in train_loader:
        optimizer.zero_grad() # Add this to reset gradients

        # Forward pass
        outputs = mobilenet_v3_large(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

#Eval
import torch.nn.functional as func

correct = 0
total = 0

val_dataset = datasets.ImageFolder(root=test_cd, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) # No need to shuffle the validation set

mobilenet_v3_large.eval()

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = mobilenet_v3_large(inputs)

        # Apply softmax to get probabilities (along dimension 1, which is the class dimension)
        probabilities = func.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(inputs.size(0)):
            sample_probabilities = probabilities[i].tolist()
            print(f"{i+1} - Predicted: {class_names[predicted[i]]}, Answer: {class_names[labels[i]]}")
            print(sample_probabilities)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')

torch.save(mobilenet_v3_large.state_dict(), 'mobilenetv3_trained.pth')