"""
warehouse_agent.py

Warehouse-level agent for inventory management.

Objective: Balance inventory holding cost vs store service levels.

Observation space (6D):
- aggregate_store_demand
- current_warehouse_inventory
- inbound_supplier_pipeline
- store_1_inventory
- store_2_inventory
- avg_store_service_level

Action space (4 discrete):
- 0: no_order
- 1: order_low (conservative)
- 2: order_medium (standard)
- 3: order_high (aggressive)

Default behavior: Rule-based (s,S) policy
Designed for future PPO upgrade without refactoring.
"""

from typing import Dict, Any
import numpy as np
from agents.base_agent import BaseAgent


class WarehouseAgent(BaseAgent):
    """
    Warehouse-level inventory management agent.
    
    Initially uses rule-based (s,S) policy.
    Designed to be upgraded to learning-based without refactoring.
    """
    
    # Action space definition
    ACTION_SPACE = {
        0: {'name': 'no_order', 'quantity': 0},
        1: {'name': 'order_low', 'quantity': 500},
        2: {'name': 'order_medium', 'quantity': 1000},
        3: {'name': 'order_high', 'quantity': 1500}
    }
    
    OBSERVATION_DIM = 6
    ACTION_DIM = 4
    
    def __init__(self, agent_id: str, warehouse_id: str, config: Dict[str, Any] = None):
        """
        Initialize warehouse agent.
        
        Args:
            agent_id: Unique agent identifier
            warehouse_id: Warehouse entity ID in digital twin
            config: Optional configuration dict
        """
        super().__init__(agent_id, agent_type='warehouse')
        self.warehouse_id = warehouse_id
        self.config = config or {}
        
        # Rule-based policy parameters
        self.reorder_point = self.config.get('reorder_point', 2000)
        self.order_up_to_level = self.config.get('order_up_to_level', 4000)
        
        # Track demand for forecasting
        self.demand_history = []
        self.forecast_window = self.config.get('forecast_window', 7)
    
    def observe(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Extract warehouse-specific observation from global state.
        
        Args:
            state: Full environment state
        
        Returns:
            6D observation vector
        """
        # Extract warehouse state
        warehouse_state = state['warehouses'].get(self.warehouse_id, {})
        stores_state = state.get('stores', {})
        
        # 1. Aggregate store demand
        aggregate_store_demand = 0.0
        for store_id, store_state in stores_state.items():
            demand = store_state.get('daily_demand', {})
            aggregate_store_demand += sum(demand.values()) if demand else 0
        
        self.demand_history.append(aggregate_store_demand)
        if len(self.demand_history) > self.forecast_window:
            self.demand_history.pop(0)
        
        # 2. Current warehouse inventory
        inventory = warehouse_state.get('inventory', {})
        current_warehouse_inventory = sum(inventory.values()) if inventory else 0
        
        # 3. Inbound supplier pipeline (in-transit inventory)
        inbound_pipeline = warehouse_state.get('inbound_orders', [])
        inbound_supplier_pipeline = sum(
            sum(order.get('quantity', {}).values()) 
            for order in inbound_pipeline
        ) if inbound_pipeline else 0
        
        # 4 & 5. Store inventories
        store_inventories = []
        for store_id in sorted(stores_state.keys()):
            store_inv = stores_state[store_id].get('inventory', {})
            store_inventories.append(sum(store_inv.values()) if store_inv else 0)
        
        # Pad to 2 stores
        while len(store_inventories) < 2:
            store_inventories.append(0)
        
        store_1_inventory = store_inventories[0]
        store_2_inventory = store_inventories[1]
        
        # 6. Average store service level
        service_levels = []
        for store_state in stores_state.values():
            sl = store_state.get('service_level', 1.0)
            service_levels.append(sl)
        
        avg_store_service_level = np.mean(service_levels) if service_levels else 1.0
        
        # Construct observation vector (normalized)
        observation = np.array([
            aggregate_store_demand / 500.0,
            current_warehouse_inventory / 5000.0,
            inbound_supplier_pipeline / 2000.0,
            store_1_inventory / 1000.0,
            store_2_inventory / 1000.0,
            avg_store_service_level
        ], dtype=np.float32)
        
        self.last_observation = observation
        return observation
    
    def act(self, observation: np.ndarray) -> int:
        """
        Select action using rule-based (s,S) policy.
        
        Args:
            observation: 6D observation vector
        
        Returns:
            Action index (0-3)
        """
        # Extract current inventory (denormalized)
        current_inventory = observation[1] * 5000.0
        avg_service_level = observation[5]
        
        # Rule-based (s,S) policy with service level consideration
        if current_inventory < self.reorder_point:
            # Order based on how far below reorder point
            shortage = self.reorder_point - current_inventory
            
            if shortage > 1500:
                action = 3  # order_high
            elif shortage > 800:
                action = 2  # order_medium
            else:
                action = 1  # order_low
        elif avg_service_level < 0.95:
            # Service level is low, order conservatively
            action = 1  # order_low
        else:
            action = 0  # no_order
        
        self.last_action = action
        return action
    
    def receive_feedback(self, reconciliation_report: Dict[str, Any]) -> float:
        """
        Compute reward from reconciliation metrics.
        
        Reward components:
        - +5.0 × avg_store_service_level (support downstream)
        - -0.05 × warehouse_holding_cost (minimize inventory)
        - -3.0 × store_stockout_count (prevent downstream stockouts)
        
        Args:
            reconciliation_report: Warehouse-specific metrics
        
        Returns:
            Scalar reward
        """
        # Extract metrics
        avg_store_service_level = reconciliation_report.get('avg_store_service_level', 0.0)
        warehouse_holding_cost = reconciliation_report.get('holding_cost', 0.0)
        store_stockout_count = reconciliation_report.get('store_stockout_count', 0)
        
        # Compute reward
        reward = (
            5.0 * avg_store_service_level
            - 0.05 * warehouse_holding_cost
            - 3.0 * store_stockout_count
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
    
    def action_to_order_quantity(self, action: int) -> float:
        """
        Convert discrete action to actual order quantity.
        
        Args:
            action: Action index (0-3)
        
        Returns:
            Order quantity in units
        """
        return self.ACTION_SPACE[action]['quantity']
