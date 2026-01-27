"""
Simple baseline runner test without full logging.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.digital_twin import DigitalTwin
import config.simulation_config as config
from baseline_policies.store_policy import MultiSKUStorePolicy
from baseline_policies.warehouse_policy import MultiSKUWarehousePolicy
import baseline_policies.policy_config as policy_config


def main():
    """Test baseline policies with digital twin."""
    print("="*70)
    print("BASELINE POLICY INTEGRATION TEST")
    print("="*70)
    
    # Build config
    sim_config = {
        'SIMULATION_DAYS': 20,
        'WARMUP_DAYS': 5,
        'RANDOM_SEED': 42,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    print("\nInitializing digital twin...")
    env = DigitalTwin(sim_config)
    
    # Initialize policies
    print("Initializing baseline policies...")
    sku_ids = list(config.SKU_CONFIG.keys())
    
    store_policies = {}
    for store_id in config.STORE_CONFIG.keys():
        store_policies[store_id] = MultiSKUStorePolicy(
            store_id, sku_ids, policy_config.STORE_POLICY_CONFIG
        )
    
    warehouse_policies = {}
    for wh_id in config.WAREHOUSE_CONFIG.keys():
        warehouse_policies[wh_id] = MultiSKUWarehousePolicy(
            wh_id, sku_ids, policy_config.WAREHOUSE_POLICY_CONFIG
        )
    
    print(f"Created {len(store_policies)} store policies")
    print(f"Created {len(warehouse_policies)} warehouse policies\n")
    
    # Run simulation
    print("Running 20-day simulation...")
    state = env.reset()
    
    for day in range(20):
        # Get state
        env_state = env.get_state()
        
        # Store decisions
        store_orders = {}
        for store_id, policy in store_policies.items():
            store_env_state = env_state['stores'][store_id]
            inventory = store_env_state['inventory']
            orders = policy.decide(inventory)
            store_orders[store_id] = orders
        
        # Warehouse decisions
        warehouse_orders = {}
        for wh_id, policy in warehouse_policies.items():
            wh_env_state = env_state['warehouses'][wh_id]
            inventory = wh_env_state['inventory']
            backorders = wh_env_state.get('backorders', {})
            in_transit = {sku: 0.0 for sku in sku_ids}
            orders = policy.decide(inventory, in_transit, backorders)
            warehouse_orders[wh_id] = orders
        
        # Build actions
        actions = {
            'store_orders': store_orders,
            'warehouse_orders': warehouse_orders
        }
        
        # Step
        next_state, metrics, done, info = env.step(actions)
        
        # Update forecasters
        for store_id, policy in store_policies.items():
            store_env_state = env_state['stores'][store_id]
            daily_demand = store_env_state.get('daily_demand', {})
            policy.update_demand(daily_demand)
        
        if (day + 1) % 5 == 0:
            svc = metrics.get('avg_service_level', 0)
            cost = metrics.get('total_cost', 0)
            print(f"  Day {day + 1}: Service Level = {svc:.2%}, Total Cost = ${cost:,.2f}")
        
        if done:
            break
    
    print(f"\nSimulation completed!\n")
    
    # Print summary
    env.metrics_tracker.print_summary()
    
    print("="*70)
    print("TEST PASSED ✓")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
