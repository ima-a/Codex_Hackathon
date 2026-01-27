"""
store_agent.py

Store-level agent for inventory management.

Objective: Maximize service level while minimizing holding cost and stockouts.

Observation space (7D):
- current_inventory
- forecasted_demand
- demand_uncertainty
- days_of_inventory_cover
- last_day_stockout
- warehouse_inventory_ratio
- recent_service_level

Action space (4 discrete):
- 0: no_order
- 1: order_0.5x (0.5 × forecast)
- 2: order_1.0x (1.0 × forecast)
- 3: order_1.5x (1.5 × forecast)

Reward (from reconciliation):
- +10.0 × service_level
- -0.1 × holding_cost
- -5.0 × stockout_penalty
- -0.05 × excess_inventory
"""

from typing import Dict, Any
import numpy as np
from agents.base_agent import BaseAgent


class StoreAgent(BaseAgent):
    """
    Store-level inventory management agent.
    
    Supports both rule-based (s,S) policy and learning-based behavior.
    Designed for PPO training with parameter sharing across store agents.
    """
    
    # Action space definition
    ACTION_SPACE = {
        0: {'name': 'no_order', 'multiplier': 0.0},
        1: {'name': 'order_0.5x', 'multiplier': 0.5},
        2: {'name': 'order_1.0x', 'multiplier': 1.0},
        3: {'name': 'order_1.5x', 'multiplier': 1.5}
    }
    
    OBSERVATION_DIM = 7
    ACTION_DIM = 4
    
    def __init__(self, agent_id: str, store_id: str, config: Dict[str, Any] = None):
        """
        Initialize store agent.
        
        Args:
            agent_id: Unique agent identifier
            store_id: Store entity ID in digital twin
            config: Optional configuration dict
        """
        super().__init__(agent_id, agent_type='store')
        self.store_id = store_id
        self.config = config or {}
        
        # Rule-based policy parameters (s,S)
        self.reorder_point = self.config.get('reorder_point', 300)
        self.order_up_to_level = self.config.get('order_up_to_level', 700)
        
        # Demand forecasting (simple moving average)
        self.demand_history = []
        self.forecast_window = self.config.get('forecast_window', 7)
        
        # Track previous state for observation
        self.prev_inventory = 0
        self.prev_stockout = 0
    
    def observe(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Extract store-specific observation from global state.
        
        Args:
            state: Full environment state
        
        Returns:
            7D observation vector
        """
        # Extract store state
        store_state = state['stores'].get(self.store_id, {})
        warehouse_state = list(state['warehouses'].values())[0] if state['warehouses'] else {}
        
        # 1. Current inventory
        inventory = store_state.get('inventory', {})
        current_inventory = sum(inventory.values()) if inventory else 0
        
        # 2. Forecasted demand (simple moving average)
        demand = store_state.get('daily_demand', {})
        current_demand = sum(demand.values()) if demand else 0
        self.demand_history.append(current_demand)
        if len(self.demand_history) > self.forecast_window:
            self.demand_history.pop(0)
        
        forecasted_demand = np.mean(self.demand_history) if self.demand_history else 100.0
        
        # 3. Demand uncertainty (std dev)
        demand_uncertainty = np.std(self.demand_history) if len(self.demand_history) > 1 else 20.0
        
        # 4. Days of inventory cover
        days_of_inventory_cover = current_inventory / forecasted_demand if forecasted_demand > 0 else 0
        
        # 5. Last day stockout indicator
        lost_sales = store_state.get('daily_lost_sales', {})
        last_day_stockout = 1.0 if sum(lost_sales.values()) > 0 else 0.0
        
        # 6. Warehouse inventory ratio
        warehouse_inventory = sum(warehouse_state.get('inventory', {}).values()) if warehouse_state else 0
        warehouse_capacity = warehouse_state.get('capacity', 5000) if warehouse_state else 5000
        warehouse_inventory_ratio = warehouse_inventory / warehouse_capacity if warehouse_capacity > 0 else 0.5
        
        # 7. Recent service level (from store state if available)
        recent_service_level = store_state.get('service_level', 1.0)
        
        # Construct observation vector
        observation = np.array([
            current_inventory / 1000.0,  # Normalize
            forecasted_demand / 200.0,
            demand_uncertainty / 50.0,
            days_of_inventory_cover / 10.0,
            last_day_stockout,
            warehouse_inventory_ratio,
            recent_service_level
        ], dtype=np.float32)
        
        # Store for next iteration
        self.last_observation = observation
        self.prev_inventory = current_inventory
        self.prev_stockout = last_day_stockout
        
        return observation
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select action using rule-based (s,S) policy.
        
        This is the default behavior. When wrapped in LearningWrapper
        with learning enabled, this method is bypassed.
        
        Args:
            observation: 7D observation vector
        
        Returns:
            Action index (0-3)
        """
        # Extract current inventory from observation (denormalized)
        current_inventory = observation[0] * 1000.0
        forecasted_demand = observation[1] * 200.0
        
        # Rule-based (s,S) policy
        if current_inventory < self.reorder_point:
            # Order up to target level
            order_quantity = self.order_up_to_level - current_inventory
            
            # Map to discrete action
            if order_quantity <= 0:
                action = 0  # no_order
            elif order_quantity < forecasted_demand * 0.75:
                action = 1  # order_0.5x
            elif order_quantity < forecasted_demand * 1.25:
                action = 2  # order_1.0x
            else:
                action = 3  # order_1.5x
        else:
            action = 0  # no_order
        
        self.last_action = action
        return action
    
    def receive_feedback(self, reconciliation_report: Dict[str, Any]) -> float:
        """
        Compute reward from reconciliation metrics.
        
        Reward components:
        - +10.0 × service_level (maximize service)
        - -0.1 × holding_cost (minimize inventory)
        - -5.0 × stockout_penalty (avoid stockouts)
        - -0.05 × excess_inventory (avoid over-ordering)
        
        Args:
            reconciliation_report: Store-specific metrics
        
        Returns:
            Scalar reward
        """
        # Extract metrics
        service_level = reconciliation_report.get('service_level', 0.0)
        holding_cost = reconciliation_report.get('holding_cost', 0.0)
        stockout_penalty = reconciliation_report.get('stockout_penalty', 0.0)
        excess_inventory = reconciliation_report.get('excess_inventory', 0.0)
        
        # Compute reward
        reward = (
            10.0 * service_level
            - 0.1 * holding_cost
            - 5.0 * stockout_penalty
            - 0.05 * excess_inventory
        )
        
        # Track for statistics
        self.episode_rewards.append(reward)
        self.total_reward += reward
        
        return reward
    
    def get_observation_space(self) -> int:
        """Return observation dimensionality."""
        return self.OBSERVATION_DIM
    
    def get_action_space(self) -> int:
        """Return number of discrete actions."""
        return self.ACTION_DIM
    
    def action_to_order_quantity(self, action: int, forecasted_demand: float) -> float:
        """
        Convert discrete action to actual order quantity.
        
        Args:
            action: Action index (0-3)
            forecasted_demand: Forecasted demand for next period
        
        Returns:
            Order quantity in units
        """
        multiplier = self.ACTION_SPACE[action]['multiplier']
        return multiplier * forecasted_demand
