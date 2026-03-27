import torch
import torch.nn as nn
import torch.nn.functional as F

class MomentumCNNExpert(nn.Module):
    """
    Expert model for analyzing momentum using a 1D CNN.
    This model processes 4H/1H timeframe data to capture short-term price movements
    and momentum shifts, treating the time series data like a 1D image.
    """
    def __init__(self, input_channels, num_classes=3):
        """
        Initialize the 1D CNN model.
        
        Args:
            input_channels (int): Number of input features (e.g., OHLCV + indicators).
            num_classes (int): Number of output classes (e.g., 0: Bearish, 1: Neutral, 2: Bullish).
        """
        super(MomentumCNNExpert, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Placeholder for calculating flattened size. This will need to be adjusted
        # based on the actual sequence length of the input data.
        # For a sequence length of 60 (e.g., 60 hours), after 3 pooling layers (60 -> 30 -> 15 -> 7):
        self.flattened_size = 64 * 7 
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).
                             Note: The input needs to be permuted to fit Conv1d's expectation.
        
        Returns:
            torch.Tensor: The output logits for each class.
        """
        # Conv layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, self.flattened_size)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
