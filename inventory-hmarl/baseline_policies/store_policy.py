"""
store_policy.py

Deterministic reorder point (ROP) policy for retail stores.

Each store independently:
- Forecasts demand using simple statistical methods
- Calculates reorder point based on lead time and safety stock
- Decides order quantity to reach target inventory level

Formula:
    ROP = (Forecast × Lead Time) + (Z-score × σ × √Lead Time)
    Order Qty = Target Inventory - Current Inventory (if inv <= ROP)
"""

import numpy as np
from typing import Dict, Optional
from baseline_policies.forecasting import create_forecaster, Forecaster


class StorePolicy:
    """
    Deterministic ROP policy for a single store.
    
    State:
    - Current inventory level
    - Forecasted demand
    - Reorder point
    - Safety stock
    
    Action:
    - Order quantity (0 if no order needed)
    """
    
    def __init__(self, store_id: str, sku_id: str, config: dict):
        """
        Initialize store policy.
        
        Args:
            store_id: Store identifier
            sku_id: SKU identifier
            config: Policy configuration dict
        """
        self.store_id = store_id
        self.sku_id = sku_id
        self.config = config
        
        # Create forecaster
        forecast_method = config.get('forecast_method', 'moving_average')
        self.forecaster = create_forecaster(forecast_method, config)
        
        # Policy parameters
        self.z_score = config.get('z_score', 1.65)
        self.lead_time = config.get('lead_time', 2)
        self.days_of_cover = config.get('days_of_cover', 14)
        self.min_order_qty = config.get('min_order_quantity', 0)
        self.max_order_qty = config.get('max_order_quantity', 1000)
        
        # State variables (updated each timestep)
        self.current_inventory = 0.0
        self.current_forecast = self.forecaster.forecast
        self.current_reorder_point = 0.0
        self.current_safety_stock = 0.0
        self.current_demand_std = self.forecaster.std_dev
        
        # Last action  
        self.last_order_quantity = 0.0
        self.last_order_placed = False
    
    def update_state(self, inventory: float):
        """
        Update policy state with current inventory.
        
        Args:
            inventory: Current on-hand inventory
        """
        self.current_inventory = inventory
    
    def update_demand(self, actual_demand: float):
        """
        Update forecaster with realized demand.
        
        Args:
            actual_demand: Actual demand that occurred
        """
        self.forecaster.update(actual_demand)
        
        # Update cached values
        self.current_forecast = self.forecaster.predict(horizon=1)
        self.current_demand_std = self.forecaster.get_std_dev()
    
    def calculate_reorder_point(self) -> float:
        """
        Calculate reorder point using ROP formula.
        
        ROP = Expected Demand during Lead Time + Safety Stock
        
        Where:
            Expected Demand = Forecast × Lead Time
            Safety Stock = Z × σ × √(Lead Time)
        
        Returns:
            Reorder point value
        """
        # Expected demand during lead time
        expected_demand_lt = self.current_forecast * self.lead_time
        
        # Safety stock
        self.current_safety_stock = (
            self.z_score 
            * self.current_demand_std 
            * np.sqrt(self.lead_time)
        )
        
        # Reorder point
        self.current_reorder_point = expected_demand_lt + self.current_safety_stock
        
        return self.current_reorder_point
    
    def calculate_order_quantity(self) -> float:
        """
        Calculate order quantity based on current state.
        
        Logic:
            If inventory <= ROP:
                Target = Days of Cover × Forecast
                Order = Target - Current Inventory
            Else:
                Order = 0
        
        Returns:
            Order quantity (0 if no order needed)
        """
        # Calculate ROP
        rop = self.calculate_reorder_point()
        
        # Check if we need to order
        if self.current_inventory <= rop:
            # Calculate target inventory
            target_inventory = self.days_of_cover * self.current_forecast
            
            # Order quantity
            order_qty = target_inventory - self.current_inventory
            
            # Apply constraints
            order_qty = max(self.min_order_qty, order_qty)
            order_qty = min(self.max_order_qty, order_qty)
            
            # Ensure non-negative
            order_qty = max(0, order_qty)
            
            self.last_order_placed = order_qty > 0
        else:
            order_qty = 0
            self.last_order_placed = False
        
        self.last_order_quantity = order_qty
        
        return order_qty
    
    def decide(self, inventory: float) -> float:
        """
        Make ordering decision for current timestep.
        
        Args:
            inventory: Current inventory level
            
        Returns:
            Order quantity to place
        """
        self.update_state(inventory)
        return self.calculate_order_quantity()
    
    def get_state(self) -> dict:
        """
        Get current policy state for logging.
        
        Returns:
            Dict with all state variables
        """
        return {
            'store_id': self.store_id,
            'sku_id': self.sku_id,
            'inventory': self.current_inventory,
            'forecast': self.current_forecast,
            'demand_std': self.current_demand_std,
            'reorder_point': self.current_reorder_point,
            'safety_stock': self.current_safety_stock,
            'lead_time': self.lead_time,
            'z_score': self.z_score,
            'days_of_cover': self.days_of_cover
        }
    
    def get_action(self) -> dict:
        """
        Get last action taken for logging.
        
        Returns:
            Dict with action information
        """
        return {
            'order_quantity': self.last_order_quantity,
            'order_placed': self.last_order_placed
        }


class MultiSKUStorePolicy:
    """
    Manages policies for multiple SKUs at a single store.
    """
    
    def __init__(self, store_id: str, sku_ids: list, config: dict):
        """
        Initialize multi-SKU store policy.
        
        Args:
            store_id: Store identifier
            sku_ids: List of SKU identifiers
            config: Policy configuration
        """
        self.store_id = store_id
        self.policies = {}
        
        # Create policy for each SKU
        for sku_id in sku_ids:
            self.policies[sku_id] = StorePolicy(store_id, sku_id, config)
    
    def decide(self, inventory: Dict[str, float]) -> Dict[str, float]:
        """
        Make ordering decisions for all SKUs.
        
        Args:
            inventory: Dict mapping SKU -> inventory level
            
        Returns:
            Dict mapping SKU -> order quantity
        """
        orders = {}
        
        for sku_id, policy in self.policies.items():
            inv = inventory.get(sku_id, 0.0)
            orders[sku_id] = policy.decide(inv)
        
        return orders
    
    def update_demand(self, demand: Dict[str, float]):
        """
        Update forecasters with realized demand.
        
        Args:
            demand: Dict mapping SKU -> actual demand
        """
        for sku_id, policy in self.policies.items():
            actual = demand.get(sku_id, 0.0)
            policy.update_demand(actual)
    
    def get_states(self) -> Dict[str, dict]:
        """Get states for all SKUs."""
        return {
            sku_id: policy.get_state()
            for sku_id, policy in self.policies.items()
        }
    
    def get_actions(self) -> Dict[str, dict]:
        """Get actions for all SKUs."""
        return {
            sku_id: policy.get_action()
            for sku_id, policy in self.policies.items()
        }
