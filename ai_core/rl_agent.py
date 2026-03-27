# ai_core/rl_agent.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    An Actor-Critic network for Reinforcement Learning.
    This network shares some layers between the actor (policy) and critic (value)
    and then has separate final layers for each.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize the Actor-Critic network.
        
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Number of neurons in the hidden layers.
        """
        super(ActorCritic, self).__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor specific layer
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic specific layer
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        A forward pass that returns both policy and value.
        This is separated from the action selection logic.
        """
        shared_features = self.shared_layers(state)
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        return action_logits, state_value

    def select_action(self, state):
        """
        Selects an action based on the current policy (actor).
        
        Args:
            state (torch.Tensor): The current state of the environment.
        
        Returns:
            int: The action chosen by the policy.
            torch.Tensor: The log probability of the chosen action.
        """
        action_logits, _ = self.forward(state)
        dist = Categorical(logits=action_logits)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

    def evaluate(self, state, action):
        action_logits, state_value = self.forward(state)
        dist = Categorical(logits=action_logits)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPOAgent:
    """
    A Proximal Policy Optimization (PPO) agent.
    This class contains the logic for training the Actor-Critic network.
    """
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action_logits, _ = self.policy_old.forward(state)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # clear buffer
        memory.clear()
