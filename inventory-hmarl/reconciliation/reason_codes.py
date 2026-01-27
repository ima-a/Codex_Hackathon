"""
reason_codes.py

Deviation reason codes and attribution rules for reconciliation.

Provides:
- DeviationReason enum
- Rule-based attribution functions
- Configuration for thresholds

All attribution is deterministic and explainable.
"""

from enum import Enum
from typing import List, Optional


class DeviationReason(Enum):
    """
    Reason codes for plan vs actual deviations.
    
    Each code represents an interpretable cause for why
    actual outcomes differed from planned outcomes.
    """
    
    # Demand-related
    FORECAST_ERROR = "forecast_error"
    DEMAND_SPIKE = "demand_spike"
    DEMAND_DROP = "demand_drop"
    
    # Inventory-related
    INSUFFICIENT_SAFETY_STOCK = "insufficient_safety_stock"
    EXCESS_SAFETY_STOCK = "excess_safety_stock"
    
    # Supply-related
    UPSTREAM_SHORTAGE = "upstream_shortage"
    SUPPLIER_DELAY = "supplier_delay"
    
    # Policy-related
    POLICY_UNDER_ORDERING = "policy_under_ordering"
    POLICY_OVER_ORDERING = "policy_over_ordering"
    
    # Execution-related
    EXECUTION_DELAY = "execution_delay"
    ORDER_NOT_FULFILLED = "order_not_fulfilled"
    
    # Good performance
    PERFECT_EXECUTION = "perfect_execution"
    ON_TARGET = "on_target"


# Attribution configuration
ATTRIBUTION_CONFIG = {
    # Demand thresholds
    'forecast_error_threshold': 0.2,       # 20% deviation triggers forecast error
    'demand_spike_threshold': 0.5,         # 50% above forecast = spike
    'demand_drop_threshold': 0.3,          # 30% below forecast = drop
    
    # Inventory thresholds
    'excess_inventory_threshold': 0.3,     # 30% above target = excess
    'low_inventory_threshold': 0.2,        # 20% below target = too low
    
    # Safety stock thresholds
    'safety_stock_multiplier': 2.0,        # Demand > 2× safety stock = insufficient
    
    # Service level thresholds
    'service_level_target': 0.95,          # Target service level
    'service_level_tolerance': 0.05,       # ±5% tolerance
    
    # Perfect execution thresholds
    'perfect_execution_tolerance': 0.05    # Within 5% = perfect
}


def attribute_demand_deviation(
    forecast: float,
    actual: float,
    config: dict = ATTRIBUTION_CONFIG
) -> List[DeviationReason]:
    """
    Attribute demand deviation to reason codes.
    
    Rules:
    - Large deviation from forecast → FORECAST_ERROR
    - Actual >> forecast → DEMAND_SPIKE
    - Actual << forecast → DEMAND_DROP
    
    Args:
        forecast: Predicted demand
        actual: Realized demand
        config: Attribution configuration
        
    Returns:
        List of applicable reason codes
    """
    reasons = []
    
    if forecast == 0:
        # Handle edge case
        if actual > 0:
            reasons.append(DeviationReason.DEMAND_SPIKE)
        return reasons
    
    # Calculate deviation
    deviation_pct = abs(actual - forecast) / forecast
    
    # General forecast error
    if deviation_pct > config['forecast_error_threshold']:
        reasons.append(DeviationReason.FORECAST_ERROR)
    
    # Demand spike
    if actual > forecast * (1 + config['demand_spike_threshold']):
        reasons.append(DeviationReason.DEMAND_SPIKE)
    
    # Demand drop
    if actual < forecast * (1 - config['demand_drop_threshold']):
        reasons.append(DeviationReason.DEMAND_DROP)
    
    # Perfect forecast
    if deviation_pct <= config['perfect_execution_tolerance']:
        reasons.append(DeviationReason.ON_TARGET)
    
    return reasons


def attribute_stockout(
    safety_stock: float,
    actual_demand: float,
    forecast: float,
    inventory_before: float,
    upstream_available: bool = True,
    config: dict = ATTRIBUTION_CONFIG
) -> List[DeviationReason]:
    """
    Attribute stockout to reason codes.
    
    Rules:
    - Demand spike beyond safety stock → DEMAND_SPIKE + INSUFFICIENT_SAFETY_STOCK
    - Upstream had shortage → UPSTREAM_SHORTAGE
    - Normal demand but low inventory → POLICY_UNDER_ORDERING
    
    Args:
        safety_stock: Configured safety stock
        actual_demand: Realized demand
        forecast: Predicted demand
        inventory_before: Inventory level before fulfillment
        upstream_available: Whether upstream had inventory
        config: Attribution configuration
        
    Returns:
        List of applicable reason codes
    """
    reasons = []
    
    # Check for demand spike
    if actual_demand > forecast * (1 + config['demand_spike_threshold']):
        reasons.append(DeviationReason.DEMAND_SPIKE)
    
    # Check if safety stock was insufficient
    demand_excess = actual_demand - forecast
    if demand_excess > safety_stock * config['safety_stock_multiplier']:
        reasons.append(DeviationReason.INSUFFICIENT_SAFETY_STOCK)
    
    # Check for upstream shortage
    if not upstream_available:
        reasons.append(DeviationReason.UPSTREAM_SHORTAGE)
    
    # Check for policy under-ordering
    if inventory_before < forecast and upstream_available:
        reasons.append(DeviationReason.POLICY_UNDER_ORDERING)
    
    return reasons


def attribute_excess_inventory(
    inventory: float,
    target_inventory: float,
    actual_demand: float,
    forecast: float,
    config: dict = ATTRIBUTION_CONFIG
) -> List[DeviationReason]:
    """
    Attribute excess inventory to reason codes.
    
    Rules:
    - Inventory >> target → POLICY_OVER_ORDERING or EXCESS_SAFETY_STOCK
    - Demand << forecast → DEMAND_DROP
    
    Args:
        inventory: Current inventory level
        target_inventory: Target/planned inventory
        actual_demand: Realized demand
        forecast: Predicted demand
        config: Attribution configuration
        
    Returns:
        List of applicable reason codes
    """
    reasons = []
    
    if target_inventory == 0:
        return reasons
    
    # Calculate excess
    excess_pct = (inventory - target_inventory) / target_inventory
    
    # Excess inventory
    if excess_pct > config['excess_inventory_threshold']:
        # Determine if it's policy or demand
        if actual_demand < forecast * (1 - config['demand_drop_threshold']):
            reasons.append(DeviationReason.DEMAND_DROP)
        else:
            reasons.append(DeviationReason.POLICY_OVER_ORDERING)
        
        reasons.append(DeviationReason.EXCESS_SAFETY_STOCK)
    
    return reasons


def attribute_service_level_deviation(
    service_level: float,
    target_service_level: float,
    reasons_so_far: List[DeviationReason],
    config: dict = ATTRIBUTION_CONFIG
) -> List[DeviationReason]:
    """
    Attribute service level deviation.
    
    This aggregates existing reasons and adds context.
    
    Args:
        service_level: Achieved service level
        target_service_level: Target service level
        reasons_so_far: Reasons already attributed
        config: Attribution configuration
        
    Returns:
        Updated list of reason codes
    """
    reasons = reasons_so_far.copy()
    
    # Perfect execution
    if abs(service_level - target_service_level) <= config['service_level_tolerance']:
        if DeviationReason.ON_TARGET not in reasons:
            reasons.append(DeviationReason.PERFECT_EXECUTION)
    
    return reasons


def get_primary_reason(reasons: List[DeviationReason]) -> Optional[DeviationReason]:
    """
    Determine primary reason from list of reasons.
    
    Priority order (most to least critical):
    1. UPSTREAM_SHORTAGE
    2. DEMAND_SPIKE
    3. INSUFFICIENT_SAFETY_STOCK
    4. POLICY_UNDER_ORDERING
    5. FORECAST_ERROR
    6. Others
    
    Args:
        reasons: List of reason codes
        
    Returns:
        Primary (most critical) reason
    """
    if not reasons:
        return None
    
    # Priority order
    priority = [
        DeviationReason.UPSTREAM_SHORTAGE,
        DeviationReason.SUPPLIER_DELAY,
        DeviationReason.DEMAND_SPIKE,
        DeviationReason.INSUFFICIENT_SAFETY_STOCK,
        DeviationReason.POLICY_UNDER_ORDERING,
        DeviationReason.ORDER_NOT_FULFILLED,
        DeviationReason.FORECAST_ERROR,
        DeviationReason.DEMAND_DROP,
        DeviationReason.POLICY_OVER_ORDERING,
        DeviationReason.EXCESS_SAFETY_STOCK,
        DeviationReason.EXECUTION_DELAY,
        DeviationReason.PERFECT_EXECUTION,
        DeviationReason.ON_TARGET
    ]
    
    for reason in priority:
        if reason in reasons:
            return reason
    
    # Return first if not in priority list
    return reasons[0]


def get_reason_severity(reason: DeviationReason) -> str:
    """
    Get severity level for a reason code.
    
    Args:
        reason: Deviation reason
        
    Returns:
        Severity: 'critical', 'warning', 'info', 'good'
    """
    critical_reasons = {
        DeviationReason.UPSTREAM_SHORTAGE,
        DeviationReason.SUPPLIER_DELAY,
        DeviationReason.DEMAND_SPIKE,
        DeviationReason.INSUFFICIENT_SAFETY_STOCK
    }
    
    warning_reasons = {
        DeviationReason.POLICY_UNDER_ORDERING,
        DeviationReason.FORECAST_ERROR,
        DeviationReason.ORDER_NOT_FULFILLED
    }
    
    info_reasons = {
        DeviationReason.DEMAND_DROP,
        DeviationReason.POLICY_OVER_ORDERING,
        DeviationReason.EXCESS_SAFETY_STOCK,
        DeviationReason.EXECUTION_DELAY
    }
    
    good_reasons = {
        DeviationReason.PERFECT_EXECUTION,
        DeviationReason.ON_TARGET
    }
    
    if reason in critical_reasons:
        return 'critical'
    elif reason in warning_reasons:
        return 'warning'
    elif reason in info_reasons:
        return 'info'
    elif reason in good_reasons:
        return 'good'
    else:
        return 'unknown'
