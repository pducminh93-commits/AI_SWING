import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from ai_core.experts.macro_trend_gru import MacroTrendGRU
from ai_core.experts.momentum_cnn import MomentumCNNExpert
from ai_core.experts.risk_vol_mlp import RiskVolatilityMLP

class ActorCriticMoE(nn.Module):
    def __init__(self, state_dim, action_dim, seq_len=30):
        super(ActorCriticMoE, self).__init__()
        self.trend_expert = MacroTrendGRU(input_dim=state_dim, hidden_dim=64, n_layers=2, output_dim=32)
        self.momentum_expert = MomentumCNNExpert(input_channels=state_dim, seq_len=seq_len, num_classes=32)
        self.risk_expert = RiskVolatilityMLP(input_dim=state_dim, output_dim=16)
        
        combined_dim = 32 + 32 + 16
        self.actor = nn.Sequential(nn.Linear(combined_dim, 128), nn.ReLU(), nn.Linear(128, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(combined_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, 64).to(x.device)
        trend_out, _ = self.trend_expert(x, h0)
        momentum_out = self.momentum_expert(x.permute(0, 2, 1))
        risk_out = self.risk_expert(x[:, -1, :])
        
        combined = torch.cat((trend_out, momentum_out, risk_out), dim=1)
        return self.actor(combined), self.critic(combined)

class PPOAgent:
    def __init__(self, state_dim, action_dim, seq_len=30, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.policy = ActorCriticMoE(state_dim, action_dim, seq_len)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCriticMoE(state_dim, action_dim, seq_len)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
        self.gamma, self.eps_clip, self.K_epochs = gamma, eps_clip, K_epochs

    def select_action(self, state):
        with torch.no_grad():
            probs, _ = self.policy_old(state)
            dist = Categorical(probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
        self.buffer['states'].append(state.cpu())
        self.buffer['actions'].append(action.cpu().item())
        self.buffer['logprobs'].append(logprob.cpu().item())
        return action.cpu().item(), logprob.cpu().item()

    def update(self):
        if not self.buffer['rewards']: return
        rewards, discounted_reward = [], 0
        for r, t in zip(reversed(self.buffer['rewards']), reversed(self.buffer['is_terminals'])):
            if t: discounted_reward = 0
            discounted_reward = r + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Stack states properly
        old_states = torch.stack(self.buffer['states'], dim=0).squeeze(1)
        old_actions = torch.tensor(self.buffer['actions']).long()
        old_logprobs = torch.tensor(self.buffer['logprobs']).float()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*nn.MSELoss()(state_values, rewards) - 0.01*dist_entropy
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        for k in self.buffer: self.buffer[k] = []

    def evaluate(self, state, action):
        probs, state_values = self.policy(state)
        dist = Categorical(probs)
        if isinstance(action, (int, float)):
            action = torch.tensor(action, dtype=torch.long).to(probs.device)
        return dist.log_prob(action), state_values.squeeze(-1), dist.entropy()
