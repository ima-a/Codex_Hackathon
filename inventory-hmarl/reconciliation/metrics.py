"""
metrics.py

Performance metrics computation for reconciliation.

Computes:
- Service level
- Fill rate
- Cost metrics
- Inventory metrics
- Bullwhip metric
- Policy efficiency score

All metrics are deterministic and explainable.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque


class MetricsCalculator:
    """
    Calculates performance metrics from operational data.
    
    Tracks metrics over time and provides aggregations.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize metrics calculator.
        
        Args:
            window_size: Rolling window for metrics (days)
        """
        self.window_size = window_size
        
        # Time series storage
        self.demand_history = deque(maxlen=window_size)
        self.sales_history = deque(maxlen=window_size)
        self.lost_sales_history = deque(maxlen=window_size)
        self.inventory_history = deque(maxlen=window_size)
        self.holding_cost_history = deque(maxlen=window_size)
        self.stockout_penalty_history = deque(maxlen=window_size)
        self.ordering_cost_history = deque(maxlen=window_size)
        self.order_quantity_history = deque(maxlen=window_size)
        
        # Cumulative totals
        self.total_demand = 0.0
        self.total_sales = 0.0
        self.total_lost_sales = 0.0
        self.total_holding_cost = 0.0
        self.total_stockout_penalty = 0.0
        self.total_ordering_cost = 0.0
        
        # Current values
        self.current_inventory = 0.0
        self.current_service_level = 0.0
    
    def update(
        self,
        demand: float,
        sales: float,
        lost_sales: float,
        inventory: float,
        holding_cost: float,
        stockout_penalty: float,
        ordering_cost: float = 0.0,
        order_quantity: float = 0.0
    ):
        """
        Update metrics with new observation.
        
        Args:
            demand: Total demand
            sales: Fulfilled sales
            lost_sales: Unfulfilled demand
            inventory: Current inventory level
            holding_cost: Holding cost incurred
            stockout_penalty: Stockout penalty incurred
            ordering_cost: Ordering cost incurred
            order_quantity: Quantity ordered
        """
        # Update histories
        self.demand_history.append(demand)
        self.sales_history.append(sales)
        self.lost_sales_history.append(lost_sales)
        self.inventory_history.append(inventory)
        self.holding_cost_history.append(holding_cost)
        self.stockout_penalty_history.append(stockout_penalty)
        self.ordering_cost_history.append(ordering_cost)
        self.order_quantity_history.append(order_quantity)
        
        # Update cumulative
        self.total_demand += demand
        self.total_sales += sales
        self.total_lost_sales += lost_sales
        self.total_holding_cost += holding_cost
        self.total_stockout_penalty += stockout_penalty
        self.total_ordering_cost += ordering_cost
        
        # Update current
        self.current_inventory = inventory
    
    def calculate_service_level(self) -> float:
        """
        Calculate service level (demand fulfillment rate).
        
        Service Level = Fulfilled Demand / Total Demand
        
        Returns:
            Service level (0-1)
        """
        if self.total_demand == 0:
            return 1.0
        
        service_level = self.total_sales / self.total_demand
        self.current_service_level = service_level
        
        return service_level
    
    def calculate_fill_rate(self) -> float:
        """
        Calculate fill rate.
        
        Fill Rate = 1 - (Lost Sales / Total Demand)
        
        Returns:
            Fill rate (0-1)
        """
        if self.total_demand == 0:
            return 1.0
        
        return 1.0 - (self.total_lost_sales / self.total_demand)
    
    def calculate_cost_metrics(self) -> Dict[str, float]:
        """
        Calculate cost-related metrics.
        
        Returns:
            Dict with cost metrics
        """
        total_cost = (
            self.total_holding_cost 
            + self.total_stockout_penalty 
            + self.total_ordering_cost
        )
        
        # Cost per unit sold
        cost_per_unit = total_cost / self.total_sales if self.total_sales > 0 else 0
        
        # Cost breakdown percentages
        holding_pct = (
            100 * self.total_holding_cost / total_cost 
            if total_cost > 0 else 0
        )
        stockout_pct = (
            100 * self.total_stockout_penalty / total_cost 
            if total_cost > 0 else 0
        )
        ordering_pct = (
            100 * self.total_ordering_cost / total_cost 
            if total_cost > 0 else 0
        )
        
        return {
            'total_cost': total_cost,
            'holding_cost': self.total_holding_cost,
            'stockout_penalty': self.total_stockout_penalty,
            'ordering_cost': self.total_ordering_cost,
            'cost_per_unit': cost_per_unit,
            'holding_pct': holding_pct,
            'stockout_pct': stockout_pct,
            'ordering_pct': ordering_pct
        }
    
    def calculate_inventory_metrics(self) -> Dict[str, float]:
        """
        Calculate inventory-related metrics.
        
        Returns:
            Dict with inventory metrics
        """
        if not self.inventory_history:
            return {
                'avg_inventory': 0.0,
                'max_inventory': 0.0,
                'min_inventory': 0.0,
                'inventory_std': 0.0,
                'inventory_turnover': 0.0,
                'days_of_supply': 0.0
            }
        
        avg_inventory = np.mean(self.inventory_history)
        max_inventory = np.max(self.inventory_history)
        min_inventory = np.min(self.inventory_history)
        inventory_std = np.std(self.inventory_history)
        
        # Inventory turnover = sales / avg inventory
        inventory_turnover = (
            self.total_sales / avg_inventory 
            if avg_inventory > 0 else 0
        )
        
        # Days of supply = avg inventory / avg daily demand
        avg_daily_demand = np.mean(self.demand_history) if self.demand_history else 0
        days_of_supply = (
            avg_inventory / avg_daily_demand 
            if avg_daily_demand > 0 else 0
        )
        
        return {
            'avg_inventory': float(avg_inventory),
            'max_inventory': float(max_inventory),
            'min_inventory': float(min_inventory),
            'inventory_std': float(inventory_std),
            'inventory_turnover': inventory_turnover,
            'days_of_supply': days_of_supply
        }
    
    def calculate_bullwhip_metric(self) -> float:
        """
        Calculate bullwhip effect metric (variance amplification).
        
        Bullwhip = Variance(Orders) / Variance(Demand)
        
        A value > 1 indicates order variance amplification (bullwhip effect).
        
        Returns:
            Bullwhip metric
        """
        if len(self.demand_history) < 2 or len(self.order_quantity_history) < 2:
            return 1.0
        
        demand_variance = np.var(self.demand_history)
        order_variance = np.var(self.order_quantity_history)
        
        if demand_variance == 0:
            return 1.0
        
        bullwhip = order_variance / demand_variance
        
        return float(bullwhip)
    
    def calculate_efficiency_score(
        self,
        service_weight: float = 0.5,
        cost_weight: float = 0.3,
        inventory_weight: float = 0.2
    ) -> float:
        """
        Calculate overall policy efficiency score.
        
        Weighted combination of:
        - Service level (higher is better)
        - Cost efficiency (lower is better)
        - Inventory efficiency (moderate is better)
        
        Args:
            service_weight: Weight for service level component
            cost_weight: Weight for cost component
            inventory_weight: Weight for inventory component
            
        Returns:
            Efficiency score (0-100, higher is better)
        """
        # Service component (0-100)
        service_level = self.calculate_service_level()
        service_score = service_level * 100
        
        # Cost component (0-100, normalized)
        # Lower cost per unit is better
        cost_metrics = self.calculate_cost_metrics()
        cost_per_unit = cost_metrics['cost_per_unit']
        
        # Normalize cost (assuming reasonable range 0-10)
        max_acceptable_cost = 10.0
        cost_score = max(0, 100 * (1 - min(cost_per_unit / max_acceptable_cost, 1.0)))
        
        # Inventory component (0-100)
        # Penalize both too high and too low inventory
        inventory_metrics = self.calculate_inventory_metrics()
        days_of_supply = inventory_metrics['days_of_supply']
        
        # Target: 14-30 days of supply
        target_days = 21
        tolerance = 14
        
        if abs(days_of_supply - target_days) <= tolerance:
            inventory_score = 100
        else:
            deviation = abs(days_of_supply - target_days) - tolerance
            inventory_score = max(0, 100 - (deviation * 5))
        
        # Weighted combination
        efficiency = (
            service_weight * service_score 
            + cost_weight * cost_score 
            + inventory_weight * inventory_score
        )
        
        return efficiency
    
    def get_all_metrics(self) -> Dict[str, any]:
        """
        Get all metrics in one dict.
        
        Returns:
            Dict with all computed metrics
        """
        metrics = {
            'service_level': self.calculate_service_level(),
            'fill_rate': self.calculate_fill_rate(),
            'bullwhip': self.calculate_bullwhip_metric(),
            'efficiency_score': self.calculate_efficiency_score()
        }
        
        # Add cost metrics
        metrics.update(self.calculate_cost_metrics())
        
        # Add inventory metrics
        metrics.update(self.calculate_inventory_metrics())
        
        # Add cumulative totals
        metrics.update({
            'total_demand': self.total_demand,
            'total_sales': self.total_sales,
            'total_lost_sales': self.total_lost_sales,
            'current_inventory': self.current_inventory
        })
        
        return metrics
    
    def reset(self):
        """Reset all metrics for new episode."""
        self.demand_history.clear()
        self.sales_history.clear()
        self.lost_sales_history.clear()
        self.inventory_history.clear()
        self.holding_cost_history.clear()
        self.stockout_penalty_history.clear()
        self.ordering_cost_history.clear()
        self.order_quantity_history.clear()
        
        self.total_demand = 0.0
        self.total_sales = 0.0
        self.total_lost_sales = 0.0
        self.total_holding_cost = 0.0
        self.total_stockout_penalty = 0.0
        self.total_ordering_cost = 0.0
        
        self.current_inventory = 0.0
        self.current_service_level = 0.0
