"""
ppo_trainer.py

PPO (Proximal Policy Optimization) Trainer for HMARL system.

Implements centralized training for multi-agent learning with:
- Shared policy across store agents
- Actor-Critic architecture
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)

Supports:
- Centralized Training, Decentralized Execution (CTDE)
- Parameter sharing
- Experience pooling from multiple agents
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    Actor network for PPO.
    
    Maps observations to action probabilities.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Initialize policy network.
        
        Args:
            obs_dim: Observation space dimensionality
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            Action logits [batch_size, action_dim]
        """
        return self.network(obs)
    
    def get_action_probs(self, obs: torch.Tensor) -> Categorical:
        """
        Get action probability distribution.
        
        Args:
            obs: Observation tensor
        
        Returns:
            Categorical distribution over actions
        """
        logits = self.forward(obs)
        return Categorical(logits=logits)


class ValueNetwork(nn.Module):
    """
    Critic network for PPO.
    
    Maps observations to state values.
    """
    
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        """
        Initialize value network.
        
        Args:
            obs_dim: Observation space dimensionality
            hidden_dim: Hidden layer size
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
        
        Returns:
            State values [batch_size, 1]
        """
        return self.network(obs)


class PPOTrainer:
    """
    PPO trainer for multi-agent reinforcement learning.
    
    Supports centralized training with parameter sharing across agents.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        shared_policy: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize PPO trainer.
        
        Args:
            obs_dim: Observation space dimensionality
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            shared_policy: Whether to share policy across agents
            device: Device for training ('cpu' or 'cuda')
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.shared_policy = shared_policy
        self.device = torch.device(device)
        
        # Initialize networks
        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.value_net = ValueNetwork(obs_dim, hidden_dim).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        
        # Training statistics
        self.num_updates = 0
        self.total_policy_loss = 0.0
        self.total_value_loss = 0.0
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action using current policy.
        
        Args:
            observation: Agent observation (numpy array)
            deterministic: If True, select argmax action; otherwise sample
        
        Returns:
            Action index
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_dist = self.policy_net.get_action_probs(obs_tensor)
            
            if deterministic:
                action = action_dist.probs.argmax(dim=-1)
            else:
                action = action_dist.sample()
            
            return action.item()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
        
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def update(
        self,
        experiences: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        num_epochs: int = 4
    ):
        """
        Update policy and value networks using PPO algorithm.
        
        Args:
            experiences: List of experience dicts from agents
            batch_size: Optional batch size for updates
            num_epochs: Number of epochs to train on each batch
        """
        if len(experiences) == 0:
            return
        
        # Extract data from experiences
        observations = []
        actions = []
        rewards = []
        dones = []
        
        for exp in experiences:
            observations.append(exp['observation'])
            actions.append(exp['action'])
            rewards.append(exp['reward'])
            dones.append(exp.get('done', False))
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Compute values
        with torch.no_grad():
            values = self.value_net(obs_tensor).squeeze(-1).cpu().numpy().tolist()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Get old log probs
        with torch.no_grad():
            old_action_dist = self.policy_net.get_action_probs(obs_tensor)
            old_log_probs = old_action_dist.log_prob(actions_tensor)
        
        # PPO update for multiple epochs
        for epoch in range(num_epochs):
            # Forward pass
            action_dist = self.policy_net.get_action_probs(obs_tensor)
            log_probs = action_dist.log_prob(actions_tensor)
            entropy = action_dist.entropy().mean()
            
            # Compute policy loss (clipped surrogate objective)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            values_pred = self.value_net(obs_tensor).squeeze(-1)
            value_loss = nn.MSELoss()(values_pred, returns_tensor)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update policy network
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
            # Track statistics
            self.total_policy_loss += policy_loss.item()
            self.total_value_loss += value_loss.item()
        
        self.num_updates += 1
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'num_updates': self.num_updates
        }, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        self.num_updates = checkpoint['num_updates']
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dict with training metrics
        """
        return {
            'num_updates': self.num_updates,
            'avg_policy_loss': self.total_policy_loss / max(1, self.num_updates),
            'avg_value_loss': self.total_value_loss / max(1, self.num_updates)
        }
