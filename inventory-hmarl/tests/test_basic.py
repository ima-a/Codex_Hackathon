"""
Simple test script to verify digital twin implementation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.digital_twin import DigitalTwin
import config.simulation_config as config

def test_basic_simulation():
    """Test basic simulation functionality."""
    print("="*60)
    print("DIGITAL TWIN TEST")
    print("="*60)
    
    # Build configuration
    sim_config = {
        'SIMULATION_DAYS': 30,
        'WARMUP_DAYS': 7,
        'RANDOM_SEED': 42,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    print(f"\nInitializing digital twin...")
    print(f"  Stores: {len(config.STORE_CONFIG)}")
    print(f"  Warehouses: {len(config.WAREHOUSE_CONFIG)}")
    print(f"  Suppliers: {len(config.SUPPLIER_CONFIG)}")
    print(f"  SKUs: {len(config.SKU_CONFIG)}")
    
    # Create environment
    env = DigitalTwin(sim_config)
    
    print(f"\nResetting environment...")
    state = env.reset()
    print(f"  Initial state has {len(state['stores'])} stores")
    
    print(f"\nRunning simulation for 30 days...")
    for day in range(30):
        state, metrics, done, info = env.step()
        
        if (day + 1) % 10 == 0:
            print(f"  Day {day + 1} completed")
            svc_level = metrics.get('avg_service_level', 0)
            print(f"    Service Level: {svc_level:.2%}")
    
    print(f"\nSimulation complete!")
    print(f"\nFinal Metrics:")
    
    metrics = env.get_metrics()
    print(f"  Total Demand: {metrics['total_demand']:,.0f} units")
    print(f"  Total Sales: {metrics['total_sales']:,.0f} units")
    print(f"  Lost Sales: {metrics['total_lost_sales']:,.0f} units")
    print(f"  Service Level: {metrics['avg_service_level']:.2%}")
    print(f"  Total Cost: ${metrics['total_cost']:,.2f}")
    print(f"  Average Daily Cost: ${metrics['avg_daily_cost']:,.2f}")
    
    print("\n" + "="*60)
    print("TEST PASSED ✓")
    print("="*60)
    
    return metrics

if __name__ == '__main__':
    try:
        test_basic_simulation()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
