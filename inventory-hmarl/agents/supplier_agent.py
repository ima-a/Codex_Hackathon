"""
supplier_agent.py

Supplier-level agent for order fulfillment.

Objective: Fulfill warehouse demand with lead time constraints.

Observation space (3D):
- warehouse_order_quantity
- current_production_capacity
- days_until_delivery

Action space (3 discrete):
- 0: fulfill_full (ship full order)
- 1: fulfill_partial (ship 80% of order)
- 2: delay (delay shipment by 1 day)

Default behavior: Rule-based (always fulfill full order with 7-day lead time)
Conforms to BaseAgent interface for future extensibility.
"""

from typing import Dict, Any
import numpy as np
from agents.base_agent import BaseAgent


class SupplierAgent(BaseAgent):
    """
    Supplier-level order fulfillment agent.
    
    Initially uses simple rule-based policy (always fulfill).
    Conforms to BaseAgent interface for consistency.
    """
    
    # Action space definition
    ACTION_SPACE = {
        0: {'name': 'fulfill_full', 'fulfillment_rate': 1.0},
        1: {'name': 'fulfill_partial', 'fulfillment_rate': 0.8},
        2: {'name': 'delay', 'fulfillment_rate': 0.0}
    }
    
    OBSERVATION_DIM = 3
    ACTION_DIM = 3
    
    def __init__(self, agent_id: str, supplier_id: str, config: Dict[str, Any] = None):
        """
        Initialize supplier agent.
        
        Args:
            agent_id: Unique agent identifier
            supplier_id: Supplier entity ID in digital twin
            config: Optional configuration dict
        """
        super().__init__(agent_id, agent_type='supplier')
        self.supplier_id = supplier_id
        self.config = config or {}
        
        # Supplier parameters
        self.production_capacity = self.config.get('production_capacity', 10000)
        self.lead_time = self.config.get('lead_time', 7)
        self.reliability = self.config.get('reliability', 1.0)
    
    def observe(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Extract supplier-specific observation from global state.
        
        Args:
            state: Full environment state
        
        Returns:
            3D observation vector
        """
        # Extract supplier state
        supplier_state = state['suppliers'].get(self.supplier_id, {})
        
        # 1. Warehouse order quantity (most recent order)
        pending_orders = supplier_state.get('pending_orders', [])
        warehouse_order_quantity = 0.0
        
        if pending_orders:
            # Get most recent order
            latest_order = pending_orders[-1]
            order_qty = latest_order.get('quantity', {})
            warehouse_order_quantity = sum(order_qty.values()) if order_qty else 0
        
        # 2. Current production capacity (static for now)
        current_production_capacity = self.production_capacity
        
        # 3. Days until delivery (lead time)
        days_until_delivery = self.lead_time
        
        # Construct observation vector (normalized)
        observation = np.array([
            warehouse_order_quantity / 2000.0,
            current_production_capacity / 10000.0,
            days_until_delivery / 10.0
        ], dtype=np.float32)
        
        self.last_observation = observation
        return observation
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select action using rule-based policy.
        
        Default: Always fulfill full order (action 0)
        
        Args:
            observation: 3D observation vector
        
        Returns:
            Action index (0-2)
        """
        # Simple rule: always fulfill full order
        # In future, could consider capacity constraints, reliability, etc.
        
        warehouse_order_quantity = observation[0] * 2000.0
        production_capacity = observation[1] * 10000.0
        
        if warehouse_order_quantity <= production_capacity:
            action = 0  # fulfill_full
        elif warehouse_order_quantity <= production_capacity * 1.25:
            action = 1  # fulfill_partial
        else:
            action = 2  # delay
        
        self.last_action = action
        return action
    
    def receive_feedback(self, reconciliation_report: Dict[str, Any]) -> float:
        """
        Compute reward from reconciliation metrics.
        
        Reward components:
        - +2.0 × fulfillment_rate (reward fulfilling orders)
        - -1.0 × delay_penalty (penalize delays)
        
        Args:
            reconciliation_report: Supplier-specific metrics
        
        Returns:
            Scalar reward
        """
        # Extract metrics
        fulfillment_rate = reconciliation_report.get('fulfillment_rate', 0.0)
        delay_penalty = reconciliation_report.get('delay_penalty', 0.0)
        
        # Compute reward
        reward = (
            2.0 * fulfillment_rate
            - 1.0 * delay_penalty
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
    
    def action_to_fulfillment_rate(self, action: int) -> float:
        """
        Convert discrete action to fulfillment rate.
        
        Args:
            action: Action index (0-2)
        
        Returns:
            Fulfillment rate (0.0 to 1.0)
        """
        return self.ACTION_SPACE[action]['fulfillment_rate']
