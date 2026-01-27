"""
store.py

Retail store entity for the digital twin simulation.

A store:
- Faces customer demand
- Fulfills demand from inventory
- Tracks sales and stockouts
- Places replenishment orders to warehouse
"""

from typing import Dict, Optional
import numpy as np


class Store:
    """
    Represents a retail store in the supply chain.
    
    Attributes:
        store_id: Unique identifier
        name: Human-readable name
        inventory: Current inventory levels per SKU
        demand_multiplier: Store-specific demand scaling
        config: Store configuration dict
    """
    
    def __init__(
        self,
        store_id: str,
        name: str,
        initial_inventory: Dict[str, float],
        config: dict
    ):
        """
        Initialize store.
        
        Args:
            store_id: Unique store identifier
            name: Store name
            initial_inventory: Dict mapping SKU -> initial quantity
            config: Configuration dict with reorder_point, order_up_to, etc.
        """
        self.store_id = store_id
        self.name = name
        self.config = config
        
        # Inventory state
        self.inventory = initial_inventory.copy()
        
        # Daily tracking
        self.daily_demand = {}
        self.daily_sales = {}
        self.daily_lost_sales = {}
        self.daily_holding_cost = 0.0
        self.daily_stockout_penalty = 0.0
        
        # Orders to warehouse
        self.pending_orders = []  # List of (SKU, quantity, order_day)
        self.in_transit = {}      # SKU -> quantity arriving today
        
        # Historical metrics
        self.history = {
            'demand': [],
            'sales': [],
            'lost_sales': [],
            'inventory': [],
            'holding_cost': [],
            'stockout_penalty': []
        }
    
    def receive_demand(self, demand: Dict[str, float], day: int):
        """
        Receive customer demand for the day.
        
        Args:
            demand: Dict mapping SKU -> demand quantity
            day: Current simulation day
        """
        self.daily_demand = demand.copy()
    
    def fulfill_demand(self, sku_costs: Dict[str, dict]) -> Dict[str, float]:
        """
        Fulfill demand from inventory, track sales and stockouts.
        
        Args:
            sku_costs: Dict mapping SKU -> cost config (holding_cost, stockout_penalty)
            
        Returns:
            Dict with sales and lost_sales per SKU
        """
        self.daily_sales = {}
        self.daily_lost_sales = {}
        self.daily_stockout_penalty = 0.0
        
        for sku, demand_qty in self.daily_demand.items():
            available = self.inventory.get(sku, 0.0)
            
            # Fulfill what we can
            fulfilled = min(demand_qty, available)
            lost = max(0, demand_qty - available)
            
            self.daily_sales[sku] = fulfilled
            self.daily_lost_sales[sku] = lost
            
            # Update inventory
            self.inventory[sku] = available - fulfilled
            
            # Calculate stockout penalty
            if sku in sku_costs:
                penalty_per_unit = sku_costs[sku].get('stockout_penalty', 0.0)
                self.daily_stockout_penalty += lost * penalty_per_unit
        
        return {
            'sales': self.daily_sales,
            'lost_sales': self.daily_lost_sales
        }
    
    def calculate_holding_cost(self, sku_costs: Dict[str, dict]) -> float:
        """
        Calculate holding cost for current inventory.
        
        Args:
            sku_costs: Dict mapping SKU -> cost config
            
        Returns:
            Total holding cost for the day
        """
        self.daily_holding_cost = 0.0
        
        for sku, quantity in self.inventory.items():
            if sku in sku_costs:
                cost_per_unit = sku_costs[sku].get('holding_cost', 0.0)
                self.daily_holding_cost += quantity * cost_per_unit
        
        return self.daily_holding_cost
    
    def check_replenishment(self) -> Dict[str, float]:
        """
        Check if replenishment is needed using (s, S) policy.
        
        Returns:
            Dict mapping SKU -> order quantity (0 if no order)
        """
        orders = {}
        
        for sku, current_inv in self.inventory.items():
            reorder_point = self.config.get('reorder_point', 0)
            order_up_to = self.config.get('order_up_to', 1000)
            
            # (s, S) policy: if inventory <= s, order up to S
            if current_inv <= reorder_point:
                order_qty = order_up_to - current_inv
                orders[sku] = max(0, order_qty)
            else:
                orders[sku] = 0
        
        return orders
    
    def place_order(self, orders: Dict[str, float], day: int):
        """
        Place replenishment order to warehouse.
        
        Args:
            orders: Dict mapping SKU -> order quantity
            day: Current simulation day
        """
        for sku, qty in orders.items():
            if qty > 0:
                self.pending_orders.append({
                    'sku': sku,
                    'quantity': qty,
                    'order_day': day
                })
    
    def receive_shipment(self, shipment: Dict[str, float]):
        """
        Receive shipment from warehouse.
        
        Args:
            shipment: Dict mapping SKU -> received quantity
        """
        for sku, qty in shipment.items():
            self.inventory[sku] = self.inventory.get(sku, 0.0) + qty
    
    def record_day(self):
        """Record daily metrics to history."""
        self.history['demand'].append(sum(self.daily_demand.values()))
        self.history['sales'].append(sum(self.daily_sales.values()))
        self.history['lost_sales'].append(sum(self.daily_lost_sales.values()))
        self.history['inventory'].append(sum(self.inventory.values()))
        self.history['holding_cost'].append(self.daily_holding_cost)
        self.history['stockout_penalty'].append(self.daily_stockout_penalty)
    
    def get_state(self) -> dict:
        """
        Get current state of the store.
        
        Returns:
            State dict with inventory, demand, sales, costs
        """
        return {
            'store_id': self.store_id,
            'name': self.name,
            'inventory': self.inventory.copy(),
            'daily_demand': self.daily_demand.copy(),
            'daily_sales': self.daily_sales.copy(),
            'daily_lost_sales': self.daily_lost_sales.copy(),
            'pending_orders': len(self.pending_orders),
            'holding_cost': self.daily_holding_cost,
            'stockout_penalty': self.daily_stockout_penalty
        }
    
    def get_metrics(self) -> dict:
        """
        Get cumulative metrics.
        
        Returns:
            Dict with total costs, service level, etc.
        """
        total_demand = sum(self.history['demand'])
        total_sales = sum(self.history['sales'])
        total_lost_sales = sum(self.history['lost_sales'])
        
        service_level = total_sales / total_demand if total_demand > 0 else 0.0
        
        return {
            'store_id': self.store_id,
            'total_demand': total_demand,
            'total_sales': total_sales,
            'total_lost_sales': total_lost_sales,
            'service_level': service_level,
            'total_holding_cost': sum(self.history['holding_cost']),
            'total_stockout_penalty': sum(self.history['stockout_penalty']),
            'avg_inventory': np.mean(self.history['inventory']) if self.history['inventory'] else 0.0
        }
    
    def reset(self, initial_inventory: Dict[str, float]):
        """
        Reset store to initial state.
        
        Args:
            initial_inventory: Initial inventory levels
        """
        self.inventory = initial_inventory.copy()
        self.daily_demand = {}
        self.daily_sales = {}
        self.daily_lost_sales = {}
        self.daily_holding_cost = 0.0
        self.daily_stockout_penalty = 0.0
        self.pending_orders = []
        self.in_transit = {}
        
        self.history = {
            'demand': [],
            'sales': [],
            'lost_sales': [],
            'inventory': [],
            'holding_cost': [],
            'stockout_penalty': []
        }
