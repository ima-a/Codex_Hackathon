"""
Demo script to showcase Digital Twin simulation output.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.digital_twin import DigitalTwin
from scenarios.scenarios import get_scenario
import config.simulation_config as config


def main():
    """Run a simple 30-day simulation and display results."""
    
    print("\n" + "="*70)
    print("DIGITAL TWIN SIMULATION DEMO")
    print("="*70)
    
    # Configuration
    sim_config = {
        'SIMULATION_DAYS': 30,
        'WARMUP_DAYS': 5,
        'RANDOM_SEED': 42,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    print(f"\nConfiguration:")
    print(f"  Days: {sim_config['SIMULATION_DAYS']}")
    print(f"  Stores: {len(config.STORE_CONFIG)}")
    print(f"  Warehouses: {len(config.WAREHOUSE_CONFIG)}")
    print(f"  SKUs: {len(config.SKU_CONFIG)}")
    print(f"  Scenario: Normal Operations")
    
    # Create environment
    print("\nInitializing digital twin...")
    env = DigitalTwin(sim_config)
    
    # Apply normal scenario
    scenario = get_scenario('normal')
    env.reset()
    scenario.apply_to_environment(env)
    
    print("Running simulation...")
    print()
    
    # Run simulation with progress updates
    for day in range(30):
        state, metrics, done, info = env.step()
        
        # Print every 5 days
        if (day + 1) % 5 == 0:
            print(f"Day {day + 1:2d} | "
                  f"Service Level: {metrics.get('avg_service_level', 0):.1%} | "
                  f"Total Demand: {metrics.get('total_demand', 0):6.0f} | "
                  f"Lost Sales: {metrics.get('total_lost_sales', 0):4.0f} | "
                  f"Total Cost: ${metrics.get('total_cost', 0):8,.0f}")
        
        if done:
            break
    
    # Final summary
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    
    final_metrics = env.get_metrics()
    
    print(f"\n📊 SERVICE METRICS:")
    print(f"   Service Level:  {final_metrics.get('avg_service_level', 0):.2%}")
    print(f"   Total Demand:   {final_metrics.get('total_demand', 0):,.0f} units")
    print(f"   Total Sales:    {final_metrics.get('total_sales', 0):,.0f} units")
    print(f"   Lost Sales:     {final_metrics.get('total_lost_sales', 0):,.0f} units")
    
    print(f"\n💰 COST METRICS:")
    print(f"   Total Cost:     ${final_metrics.get('total_cost', 0):,.2f}")
    print(f"   Holding Cost:   ${final_metrics.get('total_holding_cost', 0):,.2f}")
    print(f"   Stockout Cost:  ${final_metrics.get('total_stockout_penalty', 0):,.2f}")
    
    print(f"\n📦 INVENTORY METRICS:")
    print(f"   Avg Inventory:  {final_metrics.get('avg_inventory', 0):,.0f} units")
    
    print("\n" + "="*70)
    print("✓ Simulation completed successfully!")
    print("="*70)
    print()


if __name__ == '__main__':
    main()
