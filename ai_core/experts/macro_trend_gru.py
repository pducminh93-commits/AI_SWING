import torch
import torch.nn as nn

class MacroTrendGRU(nn.Module):
    """
    Expert model for analyzing long-term trends using a GRU.
    This model processes daily timeframe data to capture major market movements.
    """
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0.2):
        """
        Initialize the GRU model.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of neurons in the hidden layer.
            n_layers (int): Number of GRU layers.
            output_dim (int): Number of output values (e.g., 1 for trend prediction).
            dropout (float): Dropout probability.
        """
        super(MacroTrendGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, h):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            h (torch.Tensor): Initial hidden state.
        
        Returns:
            torch.Tensor: The output from the final fully connected layer.
            torch.Tensor: The last hidden state.
        """
        # Get GRU output
        out, h = self.gru(x, h)
        
        # Take the output from the last time step
        out = out[:, -1, :]
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        return out, h
        
    def init_hidden(self, batch_size, device):
        """
        Initializes the hidden state.
        
        Args:
            batch_size (int): The size of the batch.
            device (str): The device to run on ('cpu' or 'cuda').
        
        Returns:
            torch.Tensor: The initial hidden state.
        """
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
