"""
base_agent.py

Abstract base class for all agents in the HMARL system.

This interface ensures all agents (Store, Warehouse, Supplier) follow
a consistent contract, supporting both rule-based and learning-based behavior.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the HMARL system.
    
    All agents must implement:
    - observe(state) → local observation
    - act(observation) → action
    - receive_feedback(reconciliation_report) → reward
    
    This abstraction supports:
    - Rule-based policies (deterministic act())
    - Learning-based policies (via LearningWrapper)
    - Swappable policies without environment changes
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier (e.g., 'store_1', 'warehouse')
            agent_type: Type of agent ('store', 'warehouse', 'supplier')
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        
        # Track last observation and action for learning
        self.last_observation = None
        self.last_action = None
        
        # Episode tracking
        self.episode_rewards = []
        self.total_reward = 0.0
    
    @abstractmethod
    def observe(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Process full environment state into agent's local observation.
        
        This method extracts relevant information from the global state
        and constructs the agent's local observation vector.
        
        Args:
            state: Full environment state dict from digital twin
                   Contains: stores, warehouses, suppliers, day, etc.
        
        Returns:
            Local observation vector (numpy array)
            Shape depends on agent type
        """
        pass
    
    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """
        Select action based on current observation.
        
        For rule-based agents: deterministic policy
        For learning agents: will be overridden by LearningWrapper
        
        Args:
            observation: Agent's local observation vector
        
        Returns:
            Action index (discrete action space)
        """
        pass
    
    @abstractmethod
    def receive_feedback(self, reconciliation_report: Dict[str, Any]) -> float:
        """
        Process reconciliation feedback and compute reward.
        
        This is the ONLY source of reward signals in the system.
        Reconciliation report contains post-decision metrics:
        - service_level
        - holding_cost
        - stockout_penalty
        - excess_inventory
        - etc.
        
        Args:
            reconciliation_report: Agent-specific metrics from reconciliation engine
        
        Returns:
            Scalar reward signal
        """
        pass
    
    def reset(self):
        """
        Reset agent state for new episode.
        
        Called at the start of each training episode.
        Clears episode-specific tracking variables.
        """
        self.last_observation = None
        self.last_action = None
        self.total_reward = 0.0
    
    def get_observation_space(self) -> int:
        """
        Get dimensionality of observation space.
        
        Returns:
            Number of features in observation vector
        """
        raise NotImplementedError("Subclass must implement get_observation_space()")
    
    def get_action_space(self) -> int:
        """
        Get number of discrete actions.
        
        Returns:
            Number of possible actions
        """
        raise NotImplementedError("Subclass must implement get_action_space()")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics for logging/debugging.
        
        Returns:
            Dict with agent performance metrics
        """
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'total_reward': self.total_reward,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'num_steps': len(self.episode_rewards)
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id='{self.agent_id}', type='{self.agent_type}')"
