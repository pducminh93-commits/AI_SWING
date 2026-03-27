import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Import các chuyên gia từ folder experts
from ai_core.experts.macro_trend_gru import MacroTrendGRU
from ai_core.experts.momentum_cnn import MomentumCNNExpert
from ai_core.experts.risk_vol_mlp import RiskVolatilityMLP

class ActorCriticMoE(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len=30):
        super(ActorCriticMoE, self).__init__()
        self.seq_len = seq_len
        
        # 1. Khởi tạo các Chuyên gia (Experts)
        self.trend_expert = MacroTrendGRU(input_dim=state_dim, hidden_dim=64, n_layers=2, output_dim=32)
        self.momentum_expert = MomentumCNNExpert(input_channels=state_dim, num_classes=32)
        self.risk_expert = RiskVolatilityMLP(input_dim=state_dim, output_dim=16)
        
        # 2. Gating Network (Mạng cổng) - Quyết định trọng số của từng chuyên gia
        # Tổng hợp: 32 (Trend) + 32 (Momentum) + 16 (Risk) = 80 features
        combined_dim = 32 + 32 + 16
        
        self.actor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, state_dim)
        
        # Expert 1: GRU soi xu hướng
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        trend_out, _ = self.trend_expert(x, h0) # (batch, 32)
        
        # Expert 2: CNN soi hình thái nến (Cần permute sang [batch, channels, seq])
        x_cnn = x.permute(0, 2, 1) 
        momentum_out = self.momentum_expert(x_cnn) # (batch, 32)
        
        # Expert 3: MLP soi rủi ro (Chỉ lấy nến cuối cùng)
        risk_out = self.risk_expert(x[:, -1, :]) # (batch, 16)
        
        # Hợp thể ý kiến các chuyên gia
        combined = torch.cat((trend_out, momentum_out, risk_out), dim=1)
        
        return self.actor(combined), self.critic(combined)

    def select_action(self, state):
        # state: (1, seq_len, state_dim)
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state, action):
        action_probs, state_values = self.forward(state)
        dist = Categorical(action_probs)
        return dist.log_prob(action), state_values.squeeze(), dist.entropy()
