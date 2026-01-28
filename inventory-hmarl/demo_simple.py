"""
Simple Digital Twin Demonstration

This script runs a 30-day simulation and shows the key outputs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from env.digital_twin import DigitalTwin
import config.simulation_config as config


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

# Create and run
env = DigitalTwin(sim_config)
env.reset()

print("\n" + "="*80)
print(" "*25 + "DIGITAL TWIN SIMULATION")
print("="*80)
print(f"\nSimulating {sim_config['SIMULATION_DAYS']} days of retail supply chain operations...\n")

# Track daily metrics
daily_results = []

for day in range(30):
    state, metrics, done, info = env.step()
    daily_results.append(metrics)
    
    if (day + 1) % 10 == 0:
        svc = metrics.get('avg_service_level', 0) * 100
        demand = metrics.get('total_demand', 0)
        lost = metrics.get('total_lost_sales', 0)
        cost = metrics.get('total_cost', 0)
        
        print(f"Day {day + 1:2d}: Service={svc:5.1f}%  Demand={demand:6.0f}  Lost Sales={lost:4.0f}  Cost=${cost:9,.0f}")

# Final summary
final = env.get_metrics()

print("\n" + "="*80)
print(" "*30 + "FINAL RESULTS")
print("="*80)

print(f"\n📊  SERVICE PERFORMANCE:")
print(f"     Service Level:        {final.get('avg_service_level', 0)*100:6.2f}%")
print(f"     Total Demand:         {final.get('total_demand', 0):10,.0f} units")
print(f"     Fulfilled:            {final.get('total_sales', 0):10,.0f} units")
print(f"     Lost Sales:           {final.get('total_lost_sales', 0):10,.0f} units")

print(f"\n💰  COST BREAKDOWN:")
print(f"     Total Cost:           ${final.get('total_cost', 0):10,.2f}")
print(f"     Holding Cost:         ${final.get('total_holding_cost', 0):10,.2f}")
print(f"     Stockout Penalty:     ${final.get('total_stockout_penalty', 0):10,.2f}")

print(f"\n📦  INVENTORY LEVELS:")
print(f"     Average Inventory:    {final.get('avg_inventory', 0):10,.0f} units")

# Store breakdown
state = env.get_state()
print(f"\n🏪  STORE STATUS (End of Simulation):")
for store_id, store_state in state['stores'].items():
    inv = sum(store_state['inventory'].values())
    print(f"     {store_id}: {inv:6.0f} units")

# Warehouse status
print(f"\n🏭  WAREHOUSE STATUS:")
for wh_id, wh_state in state['warehouses'].items():
    inv = sum(wh_state['inventory'].values())
    print(f"     {wh_id}: {inv:6.0f} units")

print("\n" + "="*80)
print(f"✓ Simulation completed!")
print("="*80 + "\n")
