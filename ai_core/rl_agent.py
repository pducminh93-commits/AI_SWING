import torch
import torch.nn as nn
from torch.distributions import Categorical

# ==========================================
# 1. BỘ NHỚ TẠM THỜI (ROLLOUT BUFFER)
# ==========================================
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# ==========================================
# 2. KIẾN TRÚC MẠNG NƠ-RON (ACTOR-CRITIC)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # Shared Feature Extractor (Phần thân chung)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor: Quyết định hành động (Long/Short/Hold)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: Đánh giá giá trị của trạng thái hiện tại (Value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature_layer(state)
        return self.actor(features), self.critic(features)

    def select_action(self, state):
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs, state_values = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy

# ==========================================
# 3. BỘ ĐIỀU KHIỂN PPO (AGENT)
# ==========================================
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer() # Tên là buffer để tránh nhầm lẫn
        
        self.policy = ActorCritic(state_dim, action_dim).to(torch.device('cpu'))
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim).to(torch.device('cpu'))
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # state ở đây nên là Tensor đã chuẩn bị sẵn từ Cell 3
        with torch.no_grad():
            action, action_logprob = self.policy_old.select_action(state)
        
        # Lưu vào buffer ngay lập tức để tiện quản lý
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        
        return action.item(), action_logprob

    def update(self):
        if not self.buffer.rewards: return # Nếu chưa có dữ liệu thì không update

        # 1. Tính toán Monte Carlo rewards (Discounted rewards)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Chuẩn hóa Rewards (Giúp AI học ổn định hơn)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Chuyển đổi list sang tensors
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        # 2. Tối ưu hóa Policy trong K epochs
        for _ in range(self.K_epochs):
            # Đánh giá hành động cũ với Policy mới
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Tính tỉ lệ (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Tính Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Tính Loss tổng hợp
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Cập nhật Gradient
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Sao chép trọng số mới sang Policy cũ
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Xóa sạch bộ nhớ để chuẩn bị cho Epoch tiếp theo
        self.buffer.clear()
