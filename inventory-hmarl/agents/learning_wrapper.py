"""
learning_wrapper.py

Learning Wrapper for enabling RL-based behavior in agents.

This wrapper:
- Wraps any BaseAgent
- Optionally attaches a learner (e.g., PPO)
- Overrides act() when learning is enabled
- Collects experiences for training
- Updates learner using reconciliation-derived rewards

Supports:
- Centralized Training, Decentralized Execution (CTDE)
- Parameter sharing across agents
- Seamless switching between rule-based and learning-based
"""

from typing import Dict, Any, Optional, List
import numpy as np
from agents.base_agent import BaseAgent


class LearningWrapper:
    """
    Wrapper that enables learning-based behavior for any agent.
    
    When learning is enabled, overrides the agent's act() method
    to use a learned policy instead of the default rule-based policy.
    
    Collects experiences (s, a, r, s') for training and updates
    the learner using reconciliation-derived rewards.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        learner: Optional[Any] = None,
        learning_enabled: bool = False,
        buffer_size: int = 1000
    ):
        """
        Initialize learning wrapper.
        
        Args:
            agent: BaseAgent instance to wrap
            learner: Optional learner (e.g., PPOTrainer)
            learning_enabled: Whether to use learned policy
            buffer_size: Maximum experience buffer size
        """
        self.agent = agent
        self.learner = learner
        self.learning_enabled = learning_enabled
        self.buffer_size = buffer_size
        
        # Experience buffer for training
        self.experience_buffer = []
        
        # Track current transition
        self.current_observation = None
        self.current_action = None
        
        # Statistics
        self.num_learning_steps = 0
        self.num_rule_based_steps = 0
    
    def observe(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Delegate observation to wrapped agent.
        
        Args:
            state: Full environment state
        
        Returns:
            Agent's local observation vector
        """
        observation = self.agent.observe(state)
        self.current_observation = observation
        return observation
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select action using learned policy or rule-based policy.
        
        If learning is enabled and learner is attached:
            Use learner.select_action()
        Otherwise:
            Use agent's default act() method
        
        Args:
            observation: Agent's local observation
        
        Returns:
            Action index
        """
        if self.learning_enabled and self.learner is not None:
            # Use learned policy
            action = self.learner.select_action(observation)
            self.num_learning_steps += 1
        else:
            # Use rule-based policy
            action = self.agent.act(observation)
            self.num_rule_based_steps += 1
        
        self.current_action = action
        return action
    
    def receive_feedback(self, reconciliation_report: Dict[str, Any]) -> float:
        """
        Process reconciliation feedback and store experience.
        
        Computes reward using agent's receive_feedback() method,
        then stores the experience tuple for learning.
        
        Args:
            reconciliation_report: Agent-specific reconciliation metrics
        
        Returns:
            Scalar reward
        """
        # Compute reward using agent's method
        reward = self.agent.receive_feedback(reconciliation_report)
        
        # If learning is enabled, store experience
        if self.learning_enabled and self.current_observation is not None:
            experience = {
                'observation': self.current_observation.copy(),
                'action': self.current_action,
                'reward': reward,
                'done': False  # Will be set to True at episode end
            }
            
            self.experience_buffer.append(experience)
            
            # Limit buffer size
            if len(self.experience_buffer) > self.buffer_size:
                self.experience_buffer.pop(0)
        
        return reward
    
    def mark_episode_end(self):
        """
        Mark the last experience as terminal (done=True).
        
        Called at the end of each episode for episodic learning.
        """
        if self.experience_buffer:
            self.experience_buffer[-1]['done'] = True
    
    def update_learner(self, batch_size: Optional[int] = None):
        """
        Update learner using collected experiences.
        
        Called at the end of each episode or after collecting
        sufficient experiences.
        
        Args:
            batch_size: Optional batch size for update
        """
        if not self.learning_enabled or self.learner is None:
            return
        
        if len(self.experience_buffer) == 0:
            return
        
        # Update learner with experiences
        self.learner.update(self.experience_buffer, batch_size=batch_size)
        
        # Clear buffer after update (for on-policy methods like PPO)
        self.experience_buffer.clear()
    
    def get_experiences(self) -> List[Dict[str, Any]]:
        """
        Get collected experiences without clearing buffer.
        
        Used for centralized training where experiences from
        multiple agents are pooled together.
        
        Returns:
            List of experience dicts
        """
        return self.experience_buffer.copy()
    
    def clear_experiences(self):
        """Clear experience buffer."""
        self.experience_buffer.clear()
    
    def enable_learning(self):
        """Enable learning-based behavior."""
        self.learning_enabled = True
    
    def disable_learning(self):
        """Disable learning-based behavior (use rule-based)."""
        self.learning_enabled = False
    
    def set_learner(self, learner: Any):
        """
        Attach or replace learner.
        
        Args:
            learner: Learner instance (e.g., PPOTrainer)
        """
        self.learner = learner
    
    def reset(self):
        """Reset wrapper and agent for new episode."""
        self.agent.reset()
        self.current_observation = None
        self.current_action = None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get wrapper and agent statistics.
        
        Returns:
            Dict with performance metrics
        """
        agent_stats = self.agent.get_stats()
        wrapper_stats = {
            'learning_enabled': self.learning_enabled,
            'num_learning_steps': self.num_learning_steps,
            'num_rule_based_steps': self.num_rule_based_steps,
            'buffer_size': len(self.experience_buffer),
            'has_learner': self.learner is not None
        }
        
        return {**agent_stats, **wrapper_stats}
    
    def __repr__(self) -> str:
        mode = "Learning" if self.learning_enabled else "Rule-based"
        return f"LearningWrapper({self.agent}, mode={mode})"
    
    # Delegate attribute access to wrapped agent
    def __getattr__(self, name):
        """Delegate attribute access to wrapped agent."""
        return getattr(self.agent, name)
