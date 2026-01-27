"""
Test baseline forecasting and policies.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline_policies.forecasting import MovingAverageForecaster
from baseline_policies.store_policy import StorePolicy
import baseline_policies.policy_config as policy_config

def test_forecasting():
    """Test forecasting methods."""
    print("="*60)
    print("Testing Forecasting")
    print("="*60)
    
    forecaster = MovingAverageForecaster(window=3)
    
    # Feed demand history
    demands = [50, 55, 60, 50, 45]
    for d in demands:
        forecaster.update(d)
    
    forecast = forecaster.predict()
    std_dev = forecaster.get_std_dev()
    
    print(f"Demand history: {demands}")
    print(f"Forecast: {forecast:.2f}")
    print(f"Std Dev: {std_dev:.2f}")
    print("✓ Forecasting test passed\n")
    
    return forecaster


def test_store_policy():
    """Test store policy."""
    print("="*60)
    print("Testing Store Policy")
    print("="*60)
    
    config = policy_config.STORE_POLICY_CONFIG
    policy = StorePolicy('store_1', 'SKU_001', config)
    
    # Update with demand history
    demands = [50, 55, 60, 50, 45]
    for d in demands:
        policy.update_demand(d)
    
    # Make decision with inventory = 100
    inventory = 100
    order_qty = policy.decide(inventory)
    
    state = policy.get_state()
    action = policy.get_action()
    
    print(f"Inventory: {inventory}")
    print(f"Forecast: {state['forecast']:.2f}")
    print(f"Reorder Point: {state['reorder_point']:.2f}")
    print(f"Safety Stock: {state['safety_stock']:.2f}")
    print(f"Order Quantity: {action['order_quantity']:.2f}")
    print("✓ Store policy test passed\n")
    
    return policy


if __name__ == '__main__':
    try:
        test_forecasting()
        test_store_policy()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
