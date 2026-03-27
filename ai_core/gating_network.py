# ai_core/gating_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    """
    A gating network for a Mixture of Experts (MoE) model.
    It takes the input data and decides which expert should have the most influence
    on the final output by assigning a weight to each expert.
    """
    def __init__(self, input_dim, num_experts):
        """
        Initialize the Gating Network.
        
        Args:
            input_dim (int): The dimension of the input features.
            num_experts (int): The number of expert models.
        """
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        
        # A simple MLP to act as the gate
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts)
        )

    def forward(self, x):
        """
        Forward pass to get the expert weights.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).
        
        Returns:
            torch.Tensor: A tensor of weights for each expert, with shape
                          (batch_size, num_experts). The weights are produced by
                          a softmax, so they sum to 1 for each item in the batch.
        """
        # Get the logits from the gating network
        logits = self.gate(x)
        
        # Apply softmax to get a probability distribution over the experts
        weights = F.softmax(logits, dim=1)
        
        return weights
