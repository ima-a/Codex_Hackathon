"""
baseline_runner.py

Run digital twin simulation with baseline ROP policies.

Integrates:
- Digital twin environment
- Store-level ROP policies
- Warehouse-level ROP policies
- Policy logger

Usage:
    python baseline_policies/baseline_runner.py
    python baseline_policies/baseline_runner.py --days 180 --z-score 2.0
    python baseline_policies/baseline_runner.py --scenario spike --output logs/
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.digital_twin import DigitalTwin
from scenarios.scenarios import get_scenario, SCENARIOS
import config.simulation_config as config

from baseline_policies.store_policy import MultiSKUStorePolicy
from baseline_policies.warehouse_policy import MultiSKUWarehousePolicy
from baseline_policies.policy_logger import PolicyLogger
import baseline_policies.policy_config as policy_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run digital twin with baseline ROP policies'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days to simulate (default: 90)'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='normal',
        choices=list(SCENARIOS.keys()),
        help='Scenario to run (default: normal)'
    )
    
    parser.add_argument(
        '--z-score',
        type=float,
        default=None,
        help='Override Z-score for safety stock (default: from config)'
    )
    
    parser.add_argument(
        '--forecast-method',
        type=str,
        default=None,
        choices=['moving_average', 'exp_smoothing', 'seasonal'],
        help='Override forecasting method'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/baseline_logs/',
        help='Output directory for logs'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    parser.add_argument(
        '--save-json',
        action='store_true',
        default=True,
        help='Save logs as JSON (default: True)'
    )
    
    parser.add_argument(
        '--save-csv',
        action='store_true',
        default=True,
        help='Save logs as CSV (default: True)'
    )
    
    return parser.parse_args()


def main():
    """Main baseline runner."""
    args = parse_args()
    
    # Build simulation configuration
    sim_config = {
        'SIMULATION_DAYS': args.days,
        'WARMUP_DAYS': config.WARMUP_DAYS,
        'RANDOM_SEED': args.seed,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    # Override policy config if specified
    store_policy_cfg = policy_config.STORE_POLICY_CONFIG.copy()
    warehouse_policy_cfg = policy_config.WAREHOUSE_POLICY_CONFIG.copy()
    
    if args.z_score is not None:
        store_policy_cfg['z_score'] = args.z_score
        warehouse_policy_cfg['z_score'] = args.z_score
    
    if args.forecast_method is not None:
        store_policy_cfg['forecast_method'] = args.forecast_method
        warehouse_policy_cfg['forecast_method'] = args.forecast_method
    
    # Print configuration
    print("\n" + "="*70)
    print("BASELINE POLICY SIMULATION")
    print("="*70)
    print(f"Scenario: {args.scenario}")
    print(f"Simulation Days: {args.days}")
    print(f"Random Seed: {args.seed}")
    print(f"\nStore Policy:")
    print(f"  Forecast Method: {store_policy_cfg['forecast_method']}")
    print(f"  Z-Score: {store_policy_cfg['z_score']} (safety stock)")
    print(f"  Days of Cover: {store_policy_cfg['days_of_cover']}")
    print(f"  Lead Time: {store_policy_cfg['lead_time']} days")
    print(f"\nWarehouse Policy:")
    print(f"  Forecast Method: {warehouse_policy_cfg['forecast_method']}")
    print(f"  Z-Score: {warehouse_policy_cfg['z_score']}")
    print(f"  Days of Cover: {warehouse_policy_cfg['days_of_cover']}")
    print(f"  Lead Time: {warehouse_policy_cfg['lead_time']} days")
    print("="*70)
    print()
    
    # Initialize digital twin
    print("Initializing digital twin...")
    env = DigitalTwin(sim_config)
    
    # Load and apply scenario
    scenario = get_scenario(args.scenario)
    print(f"Applying scenario: {scenario.name}")
    print(f"Description: {scenario.description}\n")
    
    env.reset()
    scenario.apply_to_environment(env)
    
    # Initialize baseline policies
    print("Initializing baseline policies...")
    
    sku_ids = list(config.SKU_CONFIG.keys())
    
    # Create store policies
    store_policies = {}
    for store_id in config.STORE_CONFIG.keys():
        store_policies[store_id] = MultiSKUStorePolicy(
            store_id, sku_ids, store_policy_cfg
        )
    
    # Create warehouse policies
    warehouse_policies = {}
    for wh_id in config.WAREHOUSE_CONFIG.keys():
        warehouse_policies[wh_id] = MultiSKUWarehousePolicy(
            wh_id, sku_ids, warehouse_policy_cfg
        )
    
    # Initialize logger
    logger = PolicyLogger(log_dir=args.output, verbose=args.verbose)
    
    print(f"Created {len(store_policies)} store policies")
    print(f"Created {len(warehouse_policies)} warehouse policies")
    print()
    
    # Run simulation
    print(f"Running simulation for {args.days} days...")
    print()
    
    state = env.reset()
    
    for day in range(args.days):
        # Get current state from environment
        env_state = env.get_state()
        
        # Store decisions
        store_orders = {}
        store_states = []
        store_actions = []
        
        for store_id, policy in store_policies.items():
            # Get store inventory from environment
            store_env_state = env_state['stores'][store_id]
            inventory = store_env_state['inventory']
            
            # Policy decides order quantity
            orders = policy.decide(inventory)
            store_orders[store_id] = orders
            
            # Log state and action
            for sku_id in sku_ids:
                sku_policy = policy.policies[sku_id]
                store_states.append(sku_policy.get_state())
                store_actions.append(sku_policy.get_action())
        
        # Warehouse decisions
        warehouse_orders = {}
        warehouse_states = []
        warehouse_actions = []
        
        for wh_id, policy in warehouse_policies.items():
            # Get warehouse state from environment
            wh_env_state = env_state['warehouses'][wh_id]
            inventory = wh_env_state['inventory']
            
            # Calculate in-transit and backorders
            in_transit = {sku: 0.0 for sku in sku_ids}  # Simplified
            backorders = wh_env_state.get('backorders', {})
            
            # Policy decides order quantity
            orders = policy.decide(inventory, in_transit, backorders)
            warehouse_orders[wh_id] = orders
            
            # Log state and action
            for sku_id in sku_ids:
                sku_policy = policy.policies[sku_id]
                warehouse_states.append(sku_policy.get_state())
                warehouse_actions.append(sku_policy.get_action())
        
        # Build actions dict for environment
        actions = {
            'store_orders': store_orders,
            'warehouse_orders': warehouse_orders
        }
        
        # Step environment
        next_state, metrics, done, info = env.step(actions)
        
        # Update policy forecasters with actual demand
        for store_id, policy in store_policies.items():
            store_env_state = env_state['stores'][store_id]
            daily_demand = store_env_state.get('daily_demand', {})
            policy.update_demand(daily_demand)
        
        # Update warehouse forecasters with aggregate demand
        for wh_id, policy in warehouse_policies.items():
            # Aggregate store demands
            total_demand = {}
            for store_id in config.STORE_CONFIG.keys():
                store_env_state = env_state['stores'][store_id]
                daily_demand = store_env_state.get('daily_demand', {})
                for sku, demand in daily_demand.items():
                    total_demand[sku] = total_demand.get(sku, 0.0) + demand
            
            policy.update_demand(total_demand)
        
        # Log timestep
        outcomes = {
            'actual_demand': metrics.get('total_demand', 0),
            'sales': metrics.get('total_sales', 0),
            'lost_sales': metrics.get('total_lost_sales', 0)
        }
        
        logger.log_timestep(
            day=day,
            store_states=store_states,
            store_actions=store_actions,
            warehouse_states=warehouse_states,
            warehouse_actions=warehouse_actions,
            outcomes=outcomes,
            metrics=metrics
        )
        
        # Print progress
        if args.verbose and (day + 1) % 10 == 0:
            svc = metrics.get('avg_service_level', 0)
            print(f"  Day {day + 1}/{args.days} - Service Level: {svc:.2%}")
        
        if done:
            break
    
    print(f"\nSimulation completed: {day + 1} days")
    print()
    
    # Save logs
    if args.save_json:
        logger.save_json()
    
    if args.save_csv:
        logger.save_csv()
    
    # Print summary
    logger.print_summary()
    env.metrics_tracker.print_summary()
    
    return logger, env


if __name__ == '__main__':
    try:
        logger, env = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
