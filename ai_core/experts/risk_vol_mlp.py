import torch
import torch.nn as nn
import torch.nn.functional as F

class RiskVolatilityMLP(nn.Module):
    """
    Expert model for analyzing risk and volatility using a simple MLP.
    This model takes in volatility indicators (like ATR) and other risk-related
    features to assess current market conditions and provide a risk score or
    predict a volatility regime. It can also be used for uncertainty estimation.
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, output_dim=2):
        """
        Initialize the MLP model.
        
        This version is designed for Monte Carlo Dropout to estimate uncertainty.
        By predicting both a mean and a variance (log_var), we can sample from a
        Gaussian distribution during inference.
        
        Args:
            input_dim (int): Number of input features (e.g., ATR, std dev, etc.).
            hidden_dim1 (int): Number of neurons in the first hidden layer.
            hidden_dim2 (int): Number of neurons in the second hidden layer.
            output_dim (int): Number of output values. For uncertainty, this is typically 2
                              (mean and log variance). For simple risk score, it could be 1.
        """
        super(RiskVolatilityMLP, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim2, output_dim)
        
        # Dropout for regularization and MC Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, training=False):
        """
        Forward pass through the network.
        
        The 'training' flag is used to control dropout behavior. For Monte Carlo
        Dropout, we want dropout to be active even during inference.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            training (bool): If True, dropout is applied. Set to True during
                             inference for MC Dropout.
        
        Returns:
            torch.Tensor: The output tensor, which could be [mean, log_var] or a simple score.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # No activation on the final layer
        output = self.fc_out(x)
        
        return output
