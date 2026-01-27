"""
Test reconciliation layer components.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reconciliation import (
    DeviationReason,
    ReconciliationEngine,
    ReconciliationReport,
    MetricsCalculator
)
from reconciliation.reason_codes import (
    attribute_demand_deviation,
    attribute_stockout,
    get_primary_reason
)


def test_deviation_attribution():
    """Test deviation attribution rules."""
    print("="*60)
    print("Testing Deviation Attribution")
    print("="*60)
    
    # Test demand spike
    forecast = 50
    actual = 80  # 60% spike
    reasons = attribute_demand_deviation(forecast, actual)
    
    print(f"\nDemand Spike Test:")
    print(f"  Forecast: {forecast}, Actual: {actual}")
    print(f"  Reasons: {[r.value for r in reasons]}")
    
    assert DeviationReason.DEMAND_SPIKE in reasons
    assert DeviationReason.FORECAST_ERROR in reasons
    
    # Test stockout attribution
    safety_stock = 15
    actual_demand = 70
    forecast = 50
    inventory_before = 40  # Lower than forecast to trigger under-ordering
    
    reasons = attribute_stockout(
        safety_stock=safety_stock,
        actual_demand=actual_demand,
        forecast=forecast,
        inventory_before=inventory_before
    )
    
    print(f"\nStockout Test:")
    print(f"  Safety Stock: {safety_stock}, Demand: {actual_demand}, Forecast: {forecast}")
    print(f"  Inventory Before: {inventory_before}")
    print(f"  Reasons: {[r.value for r in reasons]}")
    
    assert DeviationReason.DEMAND_SPIKE in reasons or DeviationReason.POLICY_UNDER_ORDERING in reasons
    
    # Test primary reason
    all_reasons = [
        DeviationReason.FORECAST_ERROR,
        DeviationReason.DEMAND_SPIKE,
        DeviationReason.INSUFFICIENT_SAFETY_STOCK
    ]
    primary = get_primary_reason(all_reasons)
    
    print(f"\nPrimary Reason Test:")
    print(f"  All reasons: {[r.value for r in all_reasons]}")
    print(f"  Primary: {primary.value}")
    
    # Primary should be one of the critical ones
    assert primary in [DeviationReason.DEMAND_SPIKE, DeviationReason.INSUFFICIENT_SAFETY_STOCK]
    
    print("\n✓ Deviation attribution test passed\n")


def test_metrics_calculator():
    """Test metrics calculation."""
    print("="*60)
    print("Testing Metrics Calculator")
    print("="*60)
    
    calc = MetricsCalculator()
    
    # Feed data
    for i in range(10):
        calc.update(
            demand=50 + i,
            sales=48 + i,
            lost_sales=2,
            inventory=100 - i*5,
            holding_cost=10,
            stockout_penalty=5,
            order_quantity=50
        )
    
    # Calculate metrics
    service_level = calc.calculate_service_level()
    fill_rate = calc.calculate_fill_rate()
    cost_metrics = calc.calculate_cost_metrics()
    inv_metrics = calc.calculate_inventory_metrics()
    efficiency = calc.calculate_efficiency_score()
    
    print(f"\nMetrics:")
    print(f"  Service Level: {service_level:.2%}")
    print(f"  Fill Rate: {fill_rate:.2%}")
    print(f"  Total Cost: ${cost_metrics['total_cost']:,.2f}")
    print(f"  Avg Inventory: {inv_metrics['avg_inventory']:.1f}")
    print(f"  Efficiency Score: {efficiency:.1f}/100")
    
    assert 0 <= service_level <= 1
    assert cost_metrics['total_cost'] > 0
    assert inv_metrics['avg_inventory'] > 0
    
    print("\n✓ Metrics calculator test passed\n")


def test_reconciliation_engine():
    """Test reconciliation engine."""
    print("="*60)
    print("Testing Reconciliation Engine")
    print("="*60)
    
    engine = ReconciliationEngine()
    
    # Planned state
    planned_state = {
        'demand_forecast': 50.0,
        'planned_inventory': 200.0,
        'planned_order': 100.0,
        'target_service_level': 0.95,
        'safety_stock': 20.0,
        'planned_holding_cost': 50.0

    }
    
    # Actual state (demand spike scenario)
    actual_state = {
        'store_id': 'store_1',
        'sku_id': 'SKU_001',
        'demand': 75.0,  # 50% higher than forecast
        'fulfilled_demand': 70.0,  # Some stockout
        'lost_sales': 5.0,
        'inventory': 130.0,
        'holding_cost': 45.0,
        'stockout_penalty': 25.0,
        'order_quantity': 100.0
    }
    
    # Observe and reconcile
    engine.observe(planned_state, actual_state)
    report = engine.reconcile()
    
    print(f"\nReconciliation Report:")
    print(f"  Timestep: {report.timestep}")
    print(f"  Demand Delta: {report.demand_delta:.1f}")
    print(f"  Service Level: {report.actual_service_level:.2%}")
    print(f"  Primary Reason: {report.primary_reason.value if report.primary_reason else 'None'}")
    print(f"  All Reasons: {[r.value for r in report.deviation_reasons]}")
    print(f"  Proto-Reward: {report.proto_reward:.2f}")
    print(f"  Efficiency Score: {report.efficiency_score:.1f}/100")
    
    # Verify basic functionality
    assert report.demand_delta > 0  # Demand was higher than forecast
    assert len(report.deviation_reasons) > 0  # Some reasons were attributed
    assert report.primary_reason is not None  # Primary reason identified
    
    print("\n✓ Reconciliation engine test passed\n")


def test_episode_aggregation():
    """Test episode report aggregation."""
    print("="*60)
    print("Testing Episode Aggregation")
    print("="*60)
    
    engine = ReconciliationEngine()
    
    # Run multiple timesteps
    for day in range(10):
        planned = {
            'demand_forecast': 50.0,
            'planned_inventory': 200.0 - day*10,
            'planned_order': 50.0,
            'target_service_level': 0.95,
            'safety_stock': 20.0,
            'planned_holding_cost': 50.0
        }
        
        # Simulate varying demand
        actual_demand = 50.0 + (day % 3) * 10  # Some variance
        fulfilled = min(actual_demand, planned['planned_inventory'])
        
        actual = {
            'store_id': 'store_1',
            'sku_id': 'SKU_001',
            'demand': actual_demand,
            'fulfilled_demand': fulfilled,
            'lost_sales': max(0, actual_demand - fulfilled),
            'inventory': planned['planned_inventory'],
            'holding_cost': 40.0,
            'stockout_penalty': 10.0 if fulfilled < actual_demand else 0.0,
            'order_quantity': 50.0
        }
        
        engine.observe(planned, actual)
        engine.reconcile()
    
    # Get episode report
    episode_report = engine.get_episode_report()
    metrics = episode_report.aggregate_metrics()
    reason_dist = episode_report.get_reason_distribution()
    
    print(f"\nEpisode Metrics:")
    print(f"  Timesteps: {metrics['total_timesteps']}")
    print(f"  Avg Service Level: {metrics['avg_service_level']:.2%}")
    print(f"  Total Cost: ${metrics['total_cost']:,.2f}")
    print(f"  Avg Proto-Reward: {metrics['avg_proto_reward']:.2f}")
    
    print(f"\nReason Distribution:")
    for reason, count in sorted(reason_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {reason}: {count}")
    
    assert metrics['total_timesteps'] == 10
    assert metrics['avg_service_level'] > 0.9
    
    print("\n✓ Episode aggregation test passed\n")


if __name__ == '__main__':
    try:
        test_deviation_attribution()
        test_metrics_calculator()
        test_reconciliation_engine()
        test_episode_aggregation()
        
        print("="*60)
        print("ALL RECONCILIATION TESTS PASSED ✓")
        print("="*60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
