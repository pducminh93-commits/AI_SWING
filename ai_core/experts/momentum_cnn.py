import torch
import torch.nn as nn
import torch.nn.functional as F

class MomentumCNNExpert(nn.Module):
    def __init__(self, input_channels, seq_len=30, num_classes=32):
        super(MomentumCNNExpert, self).__init__()
        # Convolutional layers để soi hình thái nến
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # Tự động tính toán flattened_size dựa trên seq_len thực tế
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, seq_len)
            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        return self.fc2(F.relu(self.fc1(x)))
