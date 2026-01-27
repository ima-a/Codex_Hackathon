"""
metrics.py

Metrics tracking and calculation for supply chain performance.

Tracks:
- Holding costs
- Stockout penalties  
- Service levels
- Fill rates
- Inventory levels over time
"""

import numpy as np
from typing import Dict, List


class MetricsTracker:
    """
    Tracks and calculates supply chain KPIs.
    
    Attributes:
        warmup_days: Days to exclude from metrics (initialization period)
        history: Time series data
        cumulative: Cumulative totals
    """
    
    def __init__(self, warmup_days: int = 7):
        """
        Initialize metrics tracker.
        
        Args:
            warmup_days: Number of initial days to exclude from calculations
        """
        self.warmup_days = warmup_days
        self.current_day = 0
        
        # Time series history
        self.history = {
            'day': [],
            'total_demand': [],
            'total_sales': [],
            'total_lost_sales': [],
            'total_inventory': [],
            'holding_cost': [],
            'stockout_penalty': [],
            'ordering_cost': [],
            'total_cost': [],
            'service_level': [],
            'fill_rate': []
        }
        
        # Cumulative totals (after warmup)
        self.cumulative = {
            'total_demand': 0.0,
            'total_sales': 0.0,
            'total_lost_sales': 0.0,
            'total_holding_cost': 0.0,
            'total_stockout_penalty': 0.0,
            'total_ordering_cost': 0.0,
            'total_cost': 0.0
        }
    
    def record_day(
        self,
        day: int,
        store_states: List[dict],
        warehouse_states: List[dict],
        ordering_cost: float = 0.0
    ):
        """
        Record metrics for a single day.
        
        Args:
            day: Current simulation day
            store_states: List of store state dicts
            warehouse_states: List of warehouse state dicts
            ordering_cost: Total ordering cost for the day
        """
        self.current_day = day
        
        # Aggregate store metrics
        total_demand = sum(
            sum(store['daily_demand'].values()) 
            for store in store_states
        )
        total_sales = sum(
            sum(store['daily_sales'].values()) 
            for store in store_states
        )
        total_lost_sales = sum(
            sum(store['daily_lost_sales'].values()) 
            for store in store_states
        )
        
        # Aggregate inventory
        store_inventory = sum(
            sum(store['inventory'].values()) 
            for store in store_states
        )
        warehouse_inventory = sum(
            sum(wh['inventory'].values()) 
            for wh in warehouse_states
        )
        total_inventory = store_inventory + warehouse_inventory
        
        # Aggregate costs
        store_holding = sum(store['holding_cost'] for store in store_states)
        warehouse_holding = sum(wh['holding_cost'] for wh in warehouse_states)
        total_holding = store_holding + warehouse_holding
        
        total_stockout = sum(store['stockout_penalty'] for store in store_states)
        
        total_cost = total_holding + total_stockout + ordering_cost
        
        # Calculate service metrics
        service_level = total_sales / total_demand if total_demand > 0 else 1.0
        fill_rate = 1.0 - (total_lost_sales / total_demand) if total_demand > 0 else 1.0
        
        # Record to history
        self.history['day'].append(day)
        self.history['total_demand'].append(total_demand)
        self.history['total_sales'].append(total_sales)
        self.history['total_lost_sales'].append(total_lost_sales)
        self.history['total_inventory'].append(total_inventory)
        self.history['holding_cost'].append(total_holding)
        self.history['stockout_penalty'].append(total_stockout)
        self.history['ordering_cost'].append(ordering_cost)
        self.history['total_cost'].append(total_cost)
        self.history['service_level'].append(service_level)
        self.history['fill_rate'].append(fill_rate)
        
        # Update cumulative (exclude warmup period)
        if day >= self.warmup_days:
            self.cumulative['total_demand'] += total_demand
            self.cumulative['total_sales'] += total_sales
            self.cumulative['total_lost_sales'] += total_lost_sales
            self.cumulative['total_holding_cost'] += total_holding
            self.cumulative['total_stockout_penalty'] += total_stockout
            self.cumulative['total_ordering_cost'] += ordering_cost
            self.cumulative['total_cost'] += total_cost
    
    def get_summary(self) -> dict:
        """
        Get summary statistics (excluding warmup period).
        
        Returns:
            Dict with key performance metrics
        """
        # Filter history after warmup
        valid_days = [i for i, d in enumerate(self.history['day']) if d >= self.warmup_days]
        
        if not valid_days:
            return {
                'total_days': 0,
                'avg_service_level': 0.0,
                'avg_fill_rate': 0.0,
                'total_cost': 0.0,
                'avg_daily_cost': 0.0,
                'total_demand': 0.0,
                'total_lost_sales': 0.0,
                'avg_inventory': 0.0
            }
        
        service_levels = [self.history['service_level'][i] for i in valid_days]
        fill_rates = [self.history['fill_rate'][i] for i in valid_days]
        inventories = [self.history['total_inventory'][i] for i in valid_days]
        
        num_days = len(valid_days)
        
        return {
            'total_days': num_days,
            'warmup_days': self.warmup_days,
            
            # Service metrics
            'avg_service_level': np.mean(service_levels),
            'min_service_level': np.min(service_levels),
            'avg_fill_rate': np.mean(fill_rates),
            
            # Demand metrics
            'total_demand': self.cumulative['total_demand'],
            'total_sales': self.cumulative['total_sales'],
            'total_lost_sales': self.cumulative['total_lost_sales'],
            'lost_sales_rate': (
                self.cumulative['total_lost_sales'] / self.cumulative['total_demand']
                if self.cumulative['total_demand'] > 0 else 0.0
            ),
            
            # Cost metrics
            'total_cost': self.cumulative['total_cost'],
            'avg_daily_cost': self.cumulative['total_cost'] / num_days if num_days > 0 else 0.0,
            'total_holding_cost': self.cumulative['total_holding_cost'],
            'total_stockout_penalty': self.cumulative['total_stockout_penalty'],
            'total_ordering_cost': self.cumulative['total_ordering_cost'],
            
            # Cost breakdown
            'holding_cost_pct': (
                100 * self.cumulative['total_holding_cost'] / self.cumulative['total_cost']
                if self.cumulative['total_cost'] > 0 else 0.0
            ),
            'stockout_cost_pct': (
                100 * self.cumulative['total_stockout_penalty'] / self.cumulative['total_cost']
                if self.cumulative['total_cost'] > 0 else 0.0
            ),
            'ordering_cost_pct': (
                100 * self.cumulative['total_ordering_cost'] / self.cumulative['total_cost']
                if self.cumulative['total_cost'] > 0 else 0.0
            ),
            
            # Inventory metrics
            'avg_inventory': np.mean(inventories),
            'max_inventory': np.max(inventories),
            'min_inventory': np.min(inventories),
            'inventory_std': np.std(inventories)
        }
    
    def get_time_series(self, metric_name: str) -> List[float]:
        """
        Get time series for a specific metric.
        
        Args:
            metric_name: Name of metric from history
            
        Returns:
            List of values over time
        """
        if metric_name in self.history:
            return self.history[metric_name]
        return []
    
    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("SUPPLY CHAIN PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Simulation Days: {summary['total_days']} (warmup: {summary['warmup_days']})")
        print()
        
        print("SERVICE METRICS:")
        print(f"  Average Service Level: {summary['avg_service_level']:.2%}")
        print(f"  Minimum Service Level: {summary['min_service_level']:.2%}")
        print(f"  Average Fill Rate:     {summary['avg_fill_rate']:.2%}")
        print(f"  Lost Sales Rate:       {summary['lost_sales_rate']:.2%}")
        print()
        
        print("DEMAND METRICS:")
        print(f"  Total Demand:     {summary['total_demand']:,.0f} units")
        print(f"  Total Sales:      {summary['total_sales']:,.0f} units")
        print(f"  Total Lost Sales: {summary['total_lost_sales']:,.0f} units")
        print()
        
        print("COST METRICS:")
        print(f"  Total Cost:           ${summary['total_cost']:,.2f}")
        print(f"  Average Daily Cost:   ${summary['avg_daily_cost']:,.2f}")
        print(f"  - Holding Cost:       ${summary['total_holding_cost']:,.2f} ({summary['holding_cost_pct']:.1f}%)")
        print(f"  - Stockout Penalty:   ${summary['total_stockout_penalty']:,.2f} ({summary['stockout_cost_pct']:.1f}%)")
        print(f"  - Ordering Cost:      ${summary['total_ordering_cost']:,.2f} ({summary['ordering_cost_pct']:.1f}%)")
        print()
        
        print("INVENTORY METRICS:")
        print(f"  Average Inventory: {summary['avg_inventory']:,.0f} units")
        print(f"  Max Inventory:     {summary['max_inventory']:,.0f} units")
        print(f"  Min Inventory:     {summary['min_inventory']:,.0f} units")
        print(f"  Std Deviation:     {summary['inventory_std']:,.0f} units")
        print("="*60)
        print()
    
    def reset(self):
        """Reset all metrics to initial state."""
        self.current_day = 0
        
        self.history = {
            'day': [],
            'total_demand': [],
            'total_sales': [],
            'total_lost_sales': [],
            'total_inventory': [],
            'holding_cost': [],
            'stockout_penalty': [],
            'ordering_cost': [],
            'total_cost': [],
            'service_level': [],
            'fill_rate': []
        }
        
        self.cumulative = {
            'total_demand': 0.0,
            'total_sales': 0.0,
            'total_lost_sales': 0.0,
            'total_holding_cost': 0.0,
            'total_stockout_penalty': 0.0,
            'total_ordering_cost': 0.0,
            'total_cost': 0.0
        }
