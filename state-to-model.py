import torch.nn as nn
import torch
from torchvision.models import mobilenet_v3_large

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model_path = 'mobilenetv3_trained.pth'  # Replace with the actual path
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
model = mobilenet_v3_large(num_classes=8, weights=None)
model.load_state_dict(state_dict)

torch.save(model, 'manisAI.pth')