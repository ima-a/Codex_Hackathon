"""
warehouse_policy.py

Deterministic reorder point (ROP) policy for warehouse.

The warehouse:
- Aggregates demand from all stores
- Forecasts total demand
- Calculates warehouse-level ROP
- Decides when to reorder from supplier

Formula:
    Total Forecast = sum(store forecasts)
    Total Std = √(sum(store_std²))
    ROP = (Total Forecast × Supplier Lead Time) + (Z × Total Std × √Lead Time)
"""

import numpy as np
from typing import Dict, List, Optional
from baseline_policies.forecasting import create_forecaster, Forecaster


class WarehousePolicy:
    """
    Deterministic ROP policy for warehouse.
    
    Aggregates store-level forecasts and makes supplier ordering decisions.
    
    State:
    - Warehouse inventory
    - In-transit shipments from supplier
    - Backorders to stores
    - Aggregate forecast
    - Reorder point
    
    Action:
    - Order quantity to supplier
    """
    
    def __init__(self, warehouse_id: str, sku_id: str, config: dict):
        """
        Initialize warehouse policy.
        
        Args:
            warehouse_id: Warehouse identifier
            sku_id: SKU identifier
            config: Policy configuration dict
        """
        self.warehouse_id = warehouse_id
        self.sku_id = sku_id
        self.config = config
        
        # Create forecaster for aggregate demand
        forecast_method = config.get('forecast_method', 'moving_average')
        self.forecaster = create_forecaster(forecast_method, config)
        
        # Policy parameters
        self.z_score = config.get('z_score', 1.96)
        self.lead_time = config.get('lead_time', 7)
        self.days_of_cover = config.get('days_of_cover', 30)
        self.min_order_qty = config.get('min_order_quantity', 100)
        self.max_order_qty = config.get('max_order_quantity', 10000)
        
        # State variables
        self.current_inventory = 0.0
        self.current_in_transit = 0.0
        self.current_backorders = 0.0
        self.current_forecast = self.forecaster.forecast
        self.current_reorder_point = 0.0
        self.current_safety_stock = 0.0
        self.current_demand_std = self.forecaster.std_dev
        
        # Last action
        self.last_order_quantity = 0.0
        self.last_order_placed = False
    
    def update_state(
        self,
        inventory: float,
        in_transit: float = 0.0,
        backorders: float = 0.0
    ):
        """
        Update policy state.
        
        Args:
            inventory: Current on-hand inventory
            in_transit: Quantity in transit from supplier
            backorders: Quantity backordered to stores
        """
        self.current_inventory = inventory
        self.current_in_transit = in_transit
        self.current_backorders = backorders
    
    def update_demand(self, actual_demand: float):
        """
        Update forecaster with realized aggregate demand.
        
        Args:
            actual_demand: Total actual demand from all stores
        """
        self.forecaster.update(actual_demand)
        
        # Update cached values
        self.current_forecast = self.forecaster.predict(horizon=1)
        self.current_demand_std = self.forecaster.get_std_dev()
    
    def calculate_inventory_position(self) -> float:
        """
        Calculate inventory position.
        
        Inventory Position = On-Hand + In-Transit - Backorders
        
        This represents true available inventory including pipeline stock.
        
        Returns:
            Inventory position
        """
        return (
            self.current_inventory 
            + self.current_in_transit 
            - self.current_backorders
        )
    
    def calculate_reorder_point(self) -> float:
        """
        Calculate warehouse reorder point.
        
        ROP = Expected Demand during Supplier Lead Time + Safety Stock
        
        Returns:
            Reorder point value
        """
        # Expected demand during supplier lead time
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
        Calculate order quantity to supplier.
        
        Uses inventory position (not just on-hand) to account for in-transit.
        
        Logic:
            Inventory Position = On-Hand + In-Transit - Backorders
            If Inv Position <= ROP:
                Target = Days of Cover × Forecast
                Order = Target - Inv Position
            Else:
                Order = 0
        
        Returns:
            Order quantity
        """
        # Calculate inventory position
        inv_position = self.calculate_inventory_position()
        
        # Calculate ROP
        rop = self.calculate_reorder_point()
        
        # Check if we need to order
        if inv_position <= rop:
            # Calculate target inventory
            target_inventory = self.days_of_cover * self.current_forecast
            
            # Order quantity
            order_qty = target_inventory - inv_position
            
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
    
    def decide(
        self,
        inventory: float,
        in_transit: float = 0.0,
        backorders: float = 0.0
    ) -> float:
        """
        Make ordering decision for current timestep.
        
        Args:
            inventory: Current inventory level
            in_transit: In-transit quantity
            backorders: Backordered quantity
            
        Returns:
            Order quantity to place with supplier
        """
        self.update_state(inventory, in_transit, backorders)
        return self.calculate_order_quantity()
    
    def get_state(self) -> dict:
        """
        Get current policy state for logging.
        
        Returns:
            Dict with all state variables
        """
        return {
            'warehouse_id': self.warehouse_id,
            'sku_id': self.sku_id,
            'inventory': self.current_inventory,
            'in_transit': self.current_in_transit,
            'backorders': self.current_backorders,
            'inventory_position': self.calculate_inventory_position(),
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


class MultiSKUWarehousePolicy:
    """
    Manages policies for multiple SKUs at warehouse.
    """
    
    def __init__(self, warehouse_id: str, sku_ids: list, config: dict):
        """
        Initialize multi-SKU warehouse policy.
        
        Args:
            warehouse_id: Warehouse identifier
            sku_ids: List of SKU identifiers
            config: Policy configuration
        """
        self.warehouse_id = warehouse_id
        self.policies = {}
        
        # Create policy for each SKU
        for sku_id in sku_ids:
            self.policies[sku_id] = WarehousePolicy(warehouse_id, sku_id, config)
    
    def decide(
        self,
        inventory: Dict[str, float],
        in_transit: Dict[str, float],
        backorders: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Make ordering decisions for all SKUs.
        
        Args:
            inventory: Dict mapping SKU -> inventory level
            in_transit: Dict mapping SKU -> in-transit quantity
            backorders: Dict mapping SKU -> backorder quantity
            
        Returns:
            Dict mapping SKU -> order quantity
        """
        orders = {}
        
        for sku_id, policy in self.policies.items():
            inv = inventory.get(sku_id, 0.0)
            transit = in_transit.get(sku_id, 0.0)
            backlog = backorders.get(sku_id, 0.0)
            
            orders[sku_id] = policy.decide(inv, transit, backlog)
        
        return orders
    
    def update_demand(self, demand: Dict[str, float]):
        """
        Update forecasters with realized demand.
        
        Args:
            demand: Dict mapping SKU -> actual aggregate demand
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
