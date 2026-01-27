"""
policy_logger.py

Comprehensive logging for baseline policy decisions.

Logs state-action-outcome at each timestep for:
- Analysis and debugging
- Reward shaping for RL
- Baseline vs RL comparison
- Operational insights

Output formats:
- JSON (structured logs)
- CSV (time series analysis)
"""

import json
import csv
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


def convert_to_serializable(obj):
    """Convert numpy types and other non-serializable types to Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        # Handle deque and other iterables
        try:
            return [convert_to_serializable(item) for item in obj]
        except:
            return str(obj)
    else:
        return obj


class PolicyLogger:
    """
    Logs complete state-action-outcome trajectories.
    
    For each timestep, logs:
    - State: inventory, forecasts, ROP, safety stock
    - Action: order quantities, order decisions
    - Outcome: actual demand, sales, costs, service level
    """
    
    def __init__(self, log_dir: str = 'outputs/baseline_logs/', verbose: bool = False):
        """
        Initialize policy logger.
        
        Args:
            log_dir: Directory to save logs
            verbose: Print logs to console
        """
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log storage
        self.logs = []
        
        # Metadata
        self.start_time = datetime.now()
        self.metadata = {
            'start_time': self.start_time.isoformat(),
            'log_version': '1.0'
        }
    
    def log_timestep(
        self,
        day: int,
        store_states: List[dict],
        store_actions: List[dict],
        warehouse_states: List[dict],
        warehouse_actions: List[dict],
        outcomes: dict,
        metrics: dict
    ):
        """
        Log complete information for a single timestep.
        
        Args:
            day: Current simulation day
            store_states: List of store policy states
            store_actions: List of store policy actions
            warehouse_states: List of warehouse policy states
            warehouse_actions: List of warehouse policy actions
            outcomes: Realized outcomes (demand, sales, costs)
            metrics: Aggregate metrics (service level, total cost)
        """
        log_entry = {
            'day': day,
            'stores': {},
            'warehouses': {},
            'outcomes': convert_to_serializable(outcomes),
            'metrics': convert_to_serializable(metrics)
        }
        
        # Log store states and actions
        for state, action in zip(store_states, store_actions):
            store_id = state.get('store_id', 'unknown')
            log_entry['stores'][store_id] = {
                'state': convert_to_serializable(state),
                'action': convert_to_serializable(action)
            }
        
        # Log warehouse states and actions
        for state, action in zip(warehouse_states, warehouse_actions):
            wh_id = state.get('warehouse_id', 'unknown')
            log_entry['warehouses'][wh_id] = {
                'state': convert_to_serializable(state),
                'action': convert_to_serializable(action)
            }
        
        self.logs.append(log_entry)
        
        if self.verbose:
            print(f"Day {day}: Logged {len(store_states)} stores, {len(warehouse_states)} warehouses")
    
    def save_json(self, filename: str = None):
        """
        Save logs as JSON file.
        
        Args:
            filename: Output filename (default: timestamped)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_logs_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        output = {
            'metadata': self.metadata,
            'logs': self.logs
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved JSON logs to: {filepath}")
        
        return filepath
    
    def save_csv(self, filename: str = None):
        """
        Save logs as CSV file (flattened time series).
        
        Args:
            filename: Output filename (default: timestamped)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_logs_{timestamp}.csv"
        
        filepath = self.log_dir / filename
        
        # Flatten logs to rows
        rows = []
        
        for log_entry in self.logs:
            day = log_entry['day']
            metrics = log_entry.get('metrics', {})
            
            # Aggregate row
            row = {
                'day': day,
                'total_service_level': metrics.get('avg_service_level', 0),
                'total_cost': metrics.get('total_cost', 0),
                'total_demand': metrics.get('total_demand', 0),
                'total_sales': metrics.get('total_sales', 0),
                'total_lost_sales': metrics.get('total_lost_sales', 0),
                'total_inventory': metrics.get('total_inventory', 0)
            }
            
            # Add store-level data
            for store_id, store_data in log_entry.get('stores', {}).items():
                state = store_data.get('state', {})
                action = store_data.get('action', {})
                
                prefix = f"{store_id}_"
                row[f"{prefix}inventory"] = state.get('inventory', 0)
                row[f"{prefix}forecast"] = state.get('forecast', 0)
                row[f"{prefix}rop"] = state.get('reorder_point', 0)
                row[f"{prefix}safety_stock"] = state.get('safety_stock', 0)
                row[f"{prefix}order_qty"] = action.get('order_quantity', 0)
            
            # Add warehouse-level data
            for wh_id, wh_data in log_entry.get('warehouses', {}).items():
                state = wh_data.get('state', {})
                action = wh_data.get('action', {})
                
                prefix = f"{wh_id}_"
                row[f"{prefix}inventory"] = state.get('inventory', 0)
                row[f"{prefix}forecast"] = state.get('forecast', 0)
                row[f"{prefix}rop"] = state.get('reorder_point', 0)
                row[f"{prefix}in_transit"] = state.get('in_transit', 0)
                row[f"{prefix}backorders"] = state.get('backorders', 0)
                row[f"{prefix}order_qty"] = action.get('order_quantity', 0)
            
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"Saved CSV logs to: {filepath}")
        
        return filepath
    
    def get_summary(self) -> dict:
        """
        Get summary statistics from logged data.
        
        Returns:
            Dict with summary metrics
        """
        if not self.logs:
            return {}
        
        total_days = len(self.logs)
        
        # Extract metrics over time
        service_levels = []
        total_costs = []
        total_demands = []
        
        for log_entry in self.logs:
            metrics = log_entry.get('metrics', {})
            service_levels.append(metrics.get('avg_service_level', 0))
            total_costs.append(metrics.get('total_cost', 0))
            total_demands.append(metrics.get('total_demand', 0))
        
        import numpy as np
        
        return {
            'total_days': total_days,
            'avg_service_level': np.mean(service_levels) if service_levels else 0,
            'total_cost': sum(total_costs),
            'avg_daily_cost': np.mean(total_costs) if total_costs else 0,
            'total_demand': sum(total_demands)
        }
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("BASELINE POLICY SUMMARY")
        print("="*60)
        print(f"Total Days: {summary.get('total_days', 0)}")
        print(f"Average Service Level: {summary.get('avg_service_level', 0):.2%}")
        print(f"Total Cost: ${summary.get('total_cost', 0):,.2f}")
        print(f"Average Daily Cost: ${summary.get('avg_daily_cost', 0):,.2f}")
        print(f"Total Demand: {summary.get('total_demand', 0):,.0f} units")
        print("="*60)
        print()
