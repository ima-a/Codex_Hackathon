"""
hmarl_env.py

Gym-compatible multi-agent environment wrapper for HMARL system.

Integrates:
- Digital twin simulation
- Multi-agent system (Store, Warehouse, Supplier agents)
- Reconciliation engine
- Standard Gym interface

Supports:
- Multi-agent observations and actions
- Reconciliation-driven rewards
- Centralized Training, Decentralized Execution (CTDE)
- Compatible with standard RL libraries (Stable-Baselines3, RLlib, etc.)
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.digital_twin import DigitalTwin
from agents.store_agent import StoreAgent
from agents.warehouse_agent import WarehouseAgent
from agents.supplier_agent import SupplierAgent


class HMARLEnvironment(gym.Env):
    """
    Gym-compatible multi-agent environment for HMARL training.
    
    This wrapper provides a standard Gym interface for the digital twin
    with integrated multi-agent system and reconciliation engine.
    
    Observation Space:
        Dict of agent_id -> observation vector
        
    Action Space:
        Dict of agent_id -> discrete action
        
    Reward:
        Dict of agent_id -> scalar reward (from reconciliation)
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        config: Dict[str, Any],
        agent_configs: Optional[Dict[str, Dict]] = None,
        max_steps: int = 90,
        warmup_days: int = 0
    ):
        """
        Initialize HMARL environment.
        
        Args:
            config: Digital twin configuration
            agent_configs: Optional agent-specific configurations
            max_steps: Maximum steps per episode
            warmup_days: Warmup period (not counted in rewards)
        """
        super().__init__()
        
        self.config = config
        self.agent_configs = agent_configs or {}
        self.max_steps = max_steps
        self.warmup_days = warmup_days
        
        # Initialize digital twin
        self.digital_twin = DigitalTwin(config)
        
        # Initialize agents (for observation/action space definition)
        self._initialize_agents()
        
        # Define observation and action spaces
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        
        # State tracking
        self.current_state = None
        self.last_actions = None
        
        # Statistics
        self.episode_rewards = {agent_id: [] for agent_id in self.agent_ids}
        self.cumulative_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
    
    def _initialize_agents(self):
        """Initialize agent instances for space definitions."""
        self.agents = {}
        self.agent_ids = []
        
        # Store agents
        store_ids = list(self.config.get('STORE_CONFIG', {}).keys())
        for store_id in store_ids:
            agent_id = f'store_{store_id}'
            agent_config = self.agent_configs.get(agent_id, {})
            self.agents[agent_id] = StoreAgent(agent_id, store_id, agent_config)
            self.agent_ids.append(agent_id)
        
        # Warehouse agents
        warehouse_ids = list(self.config.get('WAREHOUSE_CONFIG', {}).keys())
        for warehouse_id in warehouse_ids:
            agent_id = f'warehouse_{warehouse_id}'
            agent_config = self.agent_configs.get(agent_id, {})
            self.agents[agent_id] = WarehouseAgent(agent_id, warehouse_id, agent_config)
            self.agent_ids.append(agent_id)
        
        # Supplier agents
        supplier_ids = list(self.config.get('SUPPLIER_CONFIG', {}).keys())
        for supplier_id in supplier_ids:
            agent_id = f'supplier_{supplier_id}'
            agent_config = self.agent_configs.get(agent_id, {})
            self.agents[agent_id] = SupplierAgent(agent_id, supplier_id, agent_config)
            self.agent_ids.append(agent_id)
    
    def _create_observation_space(self) -> spaces.Dict:
        """
        Create multi-agent observation space.
        
        Returns:
            Dict space with agent_id -> Box space
        """
        obs_spaces = {}
        
        for agent_id, agent in self.agents.items():
            obs_dim = agent.get_observation_space()
            obs_spaces[agent_id] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
        
        return spaces.Dict(obs_spaces)
    
    def _create_action_space(self) -> spaces.Dict:
        """
        Create multi-agent action space.
        
        Returns:
            Dict space with agent_id -> Discrete space
        """
        action_spaces = {}
        
        for agent_id, agent in self.agents.items():
            action_dim = agent.get_action_space()
            action_spaces[agent_id] = spaces.Discrete(action_dim)
        
        return spaces.Dict(action_spaces)
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state.
        
        Returns:
            Dict of agent_id -> initial observation
        """
        # Reset digital twin
        self.current_state = self.digital_twin.reset()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_count += 1
        self.last_actions = None
        
        # Reset statistics
        self.episode_rewards = {agent_id: [] for agent_id in self.agent_ids}
        self.cumulative_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        
        # Reset agents
        for agent in self.agents.values():
            agent.reset()
        
        # Get initial observations
        observations = self._get_observations()
        
        return observations
    
    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Dict of agent_id -> action
        
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        # Store actions
        self.last_actions = actions
        
        # Execute actions in digital twin
        next_state, _, _, _ = self.digital_twin.step(actions)
        
        # Compute reconciliation reports
        reconciliation_reports = self._compute_reconciliation(
            self.current_state,
            actions,
            next_state
        )
        
        # Get rewards from reconciliation
        rewards = {}
        for agent_id, agent in self.agents.items():
            report = reconciliation_reports.get(agent_id, {})
            reward = agent.receive_feedback(report)
            rewards[agent_id] = reward
            
            # Track rewards (skip warmup period)
            if self.current_step >= self.warmup_days:
                self.episode_rewards[agent_id].append(reward)
                self.cumulative_rewards[agent_id] += reward
        
        # Update state
        self.current_state = next_state
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get next observations
        observations = self._get_observations()
        
        # Build info dict
        info = self._build_info(reconciliation_reports)
        
        return observations, rewards, done, info
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get observations for all agents.
        
        Returns:
            Dict of agent_id -> observation vector
        """
        observations = {}
        
        for agent_id, agent in self.agents.items():
            observations[agent_id] = agent.observe(self.current_state)
        
        return observations
    
    def _compute_reconciliation(
        self,
        state: Dict[str, Any],
        actions: Dict[str, int],
        next_state: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute reconciliation reports for all agents.
        
        This integrates with the reconciliation engine to compute
        post-decision metrics and derive reward signals.
        
        Args:
            state: Current state
            actions: Actions taken
            next_state: Next state
        
        Returns:
            Dict of agent_id -> reconciliation_report
        """
        reports = {}
        
        # Store agents reconciliation
        for agent_id in self.agent_ids:
            if agent_id.startswith('store_'):
                store_id = agent_id.replace('store_', '')
                store_state = next_state['stores'].get(store_id, {})
                
                # Extract metrics
                inventory = sum(store_state.get('inventory', {}).values())
                lost_sales = sum(store_state.get('daily_lost_sales', {}).values())
                demand = sum(store_state.get('daily_demand', {}).values())
                
                # Compute service level
                service_level = 1.0 if lost_sales == 0 else (demand - lost_sales) / demand if demand > 0 else 1.0
                
                # Compute costs
                holding_cost = inventory * 0.1  # $0.10 per unit per day
                stockout_penalty = lost_sales * 10.0  # $10 per lost sale
                excess_inventory = max(0, inventory - 500)  # Excess above target
                
                reports[agent_id] = {
                    'service_level': service_level,
                    'holding_cost': holding_cost,
                    'stockout_penalty': stockout_penalty,
                    'excess_inventory': excess_inventory,
                    'inventory': inventory,
                    'lost_sales': lost_sales,
                    'demand': demand
                }
        
        # Warehouse agents reconciliation
        for agent_id in self.agent_ids:
            if agent_id.startswith('warehouse_'):
                warehouse_id = agent_id.replace('warehouse_', '')
                warehouse_state = next_state['warehouses'].get(warehouse_id, {})
                
                # Extract metrics
                warehouse_inv = sum(warehouse_state.get('inventory', {}).values())
                
                # Compute average store service level
                store_service_levels = []
                store_stockout_count = 0
                for store_agent_id in self.agent_ids:
                    if store_agent_id.startswith('store_'):
                        store_report = reports.get(store_agent_id, {})
                        store_service_levels.append(store_report.get('service_level', 1.0))
                        if store_report.get('lost_sales', 0) > 0:
                            store_stockout_count += 1
                
                avg_store_service_level = np.mean(store_service_levels) if store_service_levels else 1.0
                
                reports[agent_id] = {
                    'avg_store_service_level': avg_store_service_level,
                    'holding_cost': warehouse_inv * 0.05,  # $0.05 per unit per day
                    'store_stockout_count': store_stockout_count,
                    'inventory': warehouse_inv
                }
        
        # Supplier agents reconciliation
        for agent_id in self.agent_ids:
            if agent_id.startswith('supplier_'):
                # Simple fulfillment metrics
                reports[agent_id] = {
                    'fulfillment_rate': 1.0,  # Assume 100% for now
                    'delay_penalty': 0.0
                }
        
        return reports
    
    def _build_info(self, reconciliation_reports: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Build info dict with episode statistics.
        
        Args:
            reconciliation_reports: Reconciliation reports for all agents
        
        Returns:
            Info dict
        """
        info = {
            'step': self.current_step,
            'episode': self.episode_count,
            'reconciliation': reconciliation_reports,
            'cumulative_rewards': self.cumulative_rewards.copy(),
            'state': self.current_state
        }
        
        # Add digital twin metrics if available
        if hasattr(self.digital_twin, 'get_metrics'):
            info['digital_twin_metrics'] = self.digital_twin.get_metrics()
        
        return info
    
    def render(self, mode='human'):
        """
        Render environment state.
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
        """
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            
            # Print agent observations and actions
            for agent_id in self.agent_ids:
                agent = self.agents[agent_id]
                obs = agent.last_observation
                action = self.last_actions.get(agent_id, None) if self.last_actions else None
                
                print(f"\n{agent_id}:")
                print(f"  Observation: {obs}")
                print(f"  Action: {action}")
                print(f"  Cumulative Reward: {self.cumulative_rewards[agent_id]:.2f}")
            
            # Print digital twin state summary
            print(f"\nDigital Twin State:")
            for store_id, store_state in self.current_state.get('stores', {}).items():
                inventory = sum(store_state.get('inventory', {}).values())
                print(f"  {store_id}: Inventory = {inventory:.0f}")
            
            for warehouse_id, warehouse_state in self.current_state.get('warehouses', {}).items():
                inventory = sum(warehouse_state.get('inventory', {}).values())
                print(f"  {warehouse_id}: Inventory = {inventory:.0f}")
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get episode statistics.
        
        Returns:
            Dict with episode performance metrics
        """
        stats = {}
        
        for agent_id in self.agent_ids:
            rewards = self.episode_rewards[agent_id]
            stats[agent_id] = {
                'total_reward': sum(rewards),
                'avg_reward': np.mean(rewards) if rewards else 0.0,
                'min_reward': min(rewards) if rewards else 0.0,
                'max_reward': max(rewards) if rewards else 0.0,
                'num_steps': len(rewards)
            }
        
        return stats
    
    def seed(self, seed: Optional[int] = None):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
            self.config['RANDOM_SEED'] = seed
            self.digital_twin = DigitalTwin(self.config)


class SingleAgentWrapper(gym.Env):
    """
    Wrapper to expose a single agent's view from the multi-agent environment.
    
    Useful for training individual agents with single-agent RL libraries.
    """
    
    def __init__(self, multi_agent_env: HMARLEnvironment, agent_id: str):
        """
        Initialize single-agent wrapper.
        
        Args:
            multi_agent_env: Multi-agent environment
            agent_id: ID of agent to expose
        """
        super().__init__()
        
        self.env = multi_agent_env
        self.agent_id = agent_id
        
        # Extract single-agent spaces
        self.observation_space = multi_agent_env.observation_space[agent_id]
        self.action_space = multi_agent_env.action_space[agent_id]
    
    def reset(self) -> np.ndarray:
        """Reset and return single agent's observation."""
        multi_obs = self.env.reset()
        return multi_obs[self.agent_id]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Step with single agent's action.
        
        Other agents use their default policies.
        """
        # Get actions from all agents
        actions = {}
        for agent_id, agent in self.env.agents.items():
            if agent_id == self.agent_id:
                actions[agent_id] = action
            else:
                # Use agent's default policy
                obs = agent.last_observation
                if obs is not None:
                    actions[agent_id] = agent.act(obs)
                else:
                    actions[agent_id] = 0
        
        # Step environment
        multi_obs, multi_rewards, done, info = self.env.step(actions)
        
        # Extract single agent's data
        obs = multi_obs[self.agent_id]
        reward = multi_rewards[self.agent_id]
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Delegate to multi-agent environment."""
        self.env.render(mode)
    
    def close(self):
        """Delegate to multi-agent environment."""
        self.env.close()
    
    def seed(self, seed=None):
        """Delegate to multi-agent environment."""
        self.env.seed(seed)
