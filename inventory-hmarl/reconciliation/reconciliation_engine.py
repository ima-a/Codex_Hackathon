"""
reconciliation_engine.py

Main reconciliation engine that compares planned vs actual outcomes,
attributes deviations, computes metrics, and generates structured reports.

This is the core of the reconciliation layer.
"""

from typing import Dict, List, Optional
import numpy as np

from reconciliation.reason_codes import (
    DeviationReason,
    attribute_demand_deviation,
    attribute_stockout,
    attribute_excess_inventory,
    attribute_service_level_deviation,
    get_primary_reason,
    get_reason_severity,
    ATTRIBUTION_CONFIG
)
from reconciliation.metrics import MetricsCalculator
from reconciliation.report import ReconciliationReport, EpisodeReport


class ReconciliationEngine:
    """
    Reconciliation engine for comparing planned vs actual outcomes.
    
    Core responsibilities:
    1. Observe planned and actual states
    2. Compare and compute deltas
    3. Attribute deviations to reason codes
    4.    Compute performance metrics
    5. Generate structured reports
    6. Calculate proto-reward signals
    
    Interface:
        observe(planned_state, actual_state)
        reconcile() -> ReconciliationReport
        generate_report() -> ReconciliationReport
        get_metrics() -> dict
        calculate_proto_reward() -> float
    """
    
    def __init__(
        self,
        attribution_config: dict = None,
        metrics_window: int = 30
    ):
        """
        Initialize reconciliation engine.
        
        Args:
            attribution_config: Configuration for deviation attribution
            metrics_window: Rolling window for metrics calculation
        """
        self.attribution_config = attribution_config or ATTRIBUTION_CONFIG
        self.metrics_calculator = MetricsCalculator(window_size=metrics_window)
        
        # Current state storage
        self.current_planned = None
        self.current_actual = None
        self.current_report = None
        
        # Episode tracking
        self.episode_report = EpisodeReport()
        self.timestep = 0
    
    def observe(self, planned_state: dict, actual_state: dict):
        """
        Observe planned and actual states for current timestep.
        
        Args:
            planned_state: Planned/forecasted state from baseline policy
            actual_state: Actual/realized state from digital twin
        """
        self.current_planned = planned_state
        self.current_actual = actual_state
    
    def _compare_states(self) -> Dict[str, float]:
        """
        Compare planned vs actual states and compute deltas.
        
        Returns:
            Dict with all deltas
        """
        planned = self.current_planned
        actual = self.current_actual
        
        # Demand delta
        demand_delta = actual['demand'] - planned['demand_forecast']
        
        # Inventory delta
        inventory_delta = actual['inventory'] - planned['planned_inventory']
        
        # Service level calculation
        if actual['demand'] > 0:
            actual_service_level = actual['fulfilled_demand'] / actual['demand']
        else:
            actual_service_level = 1.0
        
        service_level_delta = actual_service_level - planned['target_service_level']
        
        # Cost delta
        planned_cost = planned.get('planned_holding_cost', 0.0)
        actual_cost = actual.get('holding_cost', 0.0) + actual.get('stockout_penalty', 0.0)
        cost_delta = actual_cost - planned_cost
        
        return {
            'demand_delta': demand_delta,
            'inventory_delta': inventory_delta,
            'service_level_delta': service_level_delta,
            'cost_delta': cost_delta,
            'actual_service_level': actual_service_level
        }
    
    def _attribute_deviations(self, deltas: dict) -> List[DeviationReason]:
        """
        Attribute deviations to reason codes using rule-based logic.
        
        Args:
            deltas: Computed deltas from comparison
            
        Returns:
            List of applicable deviation reasons
        """
        planned = self.current_planned
        actual = self.current_actual
        reasons = []
        
        # Attribute demand deviation
        demand_reasons = attribute_demand_deviation(
            forecast=planned['demand_forecast'],
            actual=actual['demand'],
            config=self.attribution_config
        )
        reasons.extend(demand_reasons)
        
        # Attribute stockout (if occurred)
        if actual.get('lost_sales', 0) > 0:
            stockout_reasons = attribute_stockout(
                safety_stock=planned.get('safety_stock', 0),
                actual_demand=actual['demand'],
                forecast=planned['demand_forecast'],
                inventory_before=planned['planned_inventory'],
                upstream_available=actual.get('upstream_available', True),
                config=self.attribution_config
            )
            reasons.extend(stockout_reasons)
        
        # Attribute excess inventory (if exists)
        if deltas['inventory_delta'] > 0:
            excess_reasons = attribute_excess_inventory(
                inventory=actual['inventory'],
                target_inventory=planned['planned_inventory'],
                actual_demand=actual['demand'],
                forecast=planned['demand_forecast'],
                config=self.attribution_config
            )
            reasons.extend(excess_reasons)
        
        # Attribute service level deviation
        reasons = attribute_service_level_deviation(
            service_level=deltas['actual_service_level'],
            target_service_level=planned['target_service_level'],
            reasons_so_far=reasons,
            config=self.attribution_config
        )
        
        # Remove duplicates while preserving order
        unique_reasons = []
        seen = set()
        for r in reasons:
            if r not in seen:
                unique_reasons.append(r)
                seen.add(r)
        
        return unique_reasons
    
    def _calculate_proto_reward(
        self,
        report: ReconciliationReport,
        service_weight: float = 100.0,
        holding_weight: float = 1.0,
        stockout_weight: float = 2.0,
        excess_weight: float = 0.5
    ) -> tuple:
        """
        Calculate proto-reward signal (deterministic, not for training yet).
        
        Formula:
            reward = +service_bonus - holding_penalty - stockout_penalty - excess_penalty
        
        Args:
            report: Reconciliation report
            service_weight: Weight for service bonus
            holding_weight: Weight for holding penalty
            stockout_weight: Weight for stockout penalty
            excess_weight: Weight for excess inventory penalty
            
        Returns:
            Tuple of (proto_reward, service_bonus, cost_penalty, inventory_penalty)
        """
        # Service level bonus
        service_level = report.actual_service_level
        
        if service_level >= 0.98:
            service_bonus = 1.0 * service_weight
        elif service_level >= 0.95:
            service_bonus = 0.8 * service_weight
        elif service_level >= 0.90:
            service_bonus = 0.5 * service_weight
        elif service_level >= 0.85:
            service_bonus = 0.3 * service_weight
        else:
            service_bonus = 0.0
        
        # Cost penalties
        holding_penalty = report.actual_holding_cost * holding_weight
        stockout_penalty = report.stockout_penalty * stockout_weight
        cost_penalty = holding_penalty + stockout_penalty
        
        # Excess inventory penalty
        excess_inventory = max(0, report.actual_inventory - report.planned_inventory)
        inventory_penalty = excess_inventory * excess_weight
        
        # Proto-reward
        proto_reward = service_bonus - cost_penalty - inventory_penalty
        
        return proto_reward, service_bonus, cost_penalty, inventory_penalty
    
    def reconcile(self) -> ReconciliationReport:
        """
        Perform reconciliation: compare, attribute, compute metrics, generate report.
        
        Returns:
            ReconciliationReport for current timestep
        """
        if not self.current_planned or not self.current_actual:
            raise ValueError("Must call observe() before reconcile()")
        
        # Compare states
        deltas = self._compare_states()
        
        # Attribute deviations
        deviation_reasons = self._attribute_deviations(deltas)
        primary_reason = get_primary_reason(deviation_reasons)
        reason_severity = get_reason_severity(primary_reason) if primary_reason else "info"
        
        # Update metrics calculator
        self.metrics_calculator.update(
            demand=self.current_actual['demand'],
            sales=self.current_actual['fulfilled_demand'],
            lost_sales=self.current_actual.get('lost_sales', 0),
            inventory=self.current_actual['inventory'],
            holding_cost=self.current_actual.get('holding_cost', 0),
            stockout_penalty=self.current_actual.get('stockout_penalty', 0),
            ordering_cost=self.current_actual.get('ordering_cost', 0),
            order_quantity=self.current_actual.get('order_quantity', 0)
        )
        
        # Get metrics
        service_level = self.metrics_calculator.calculate_service_level()
        fill_rate = self.metrics_calculator.calculate_fill_rate()
        cost_metrics = self.metrics_calculator.calculate_cost_metrics()
        efficiency_score = self.metrics_calculator.calculate_efficiency_score()
        
        # Create report
        report = ReconciliationReport(
            timestep=self.timestep,
            store_id=self.current_actual.get('store_id', 'unknown'),
            sku_id=self.current_actual.get('sku_id', 'unknown'),
            
            # Planned
            planned_demand=self.current_planned['demand_forecast'],
            planned_inventory=self.current_planned['planned_inventory'],
            planned_order=self.current_planned.get('planned_order', 0),
            planned_service_level=self.current_planned['target_service_level'],
            planned_holding_cost=self.current_planned.get('planned_holding_cost', 0),
            
            # Actual
            actual_demand=self.current_actual['demand'],
            actual_inventory=self.current_actual['inventory'],
            actual_order=self.current_actual.get('order_quantity', 0),
            fulfilled_demand=self.current_actual['fulfilled_demand'],
            lost_sales=self.current_actual.get('lost_sales', 0),
            actual_service_level=deltas['actual_service_level'],
            actual_holding_cost=self.current_actual.get('holding_cost', 0),
            stockout_penalty=self.current_actual.get('stockout_penalty', 0),
            
            # Deltas
            demand_delta=deltas['demand_delta'],
            inventory_delta=deltas['inventory_delta'],
            service_level_delta=deltas['service_level_delta'],
            cost_delta=deltas['cost_delta'],
            
            # Attribution
            deviation_reasons=deviation_reasons,
            primary_reason=primary_reason,
            reason_severity=reason_severity,
            
            # Metrics
            service_level=service_level,
            fill_rate=fill_rate,
            total_cost=cost_metrics['total_cost'],
            efficiency_score=efficiency_score
        )
        
        # Calculate proto-reward
        proto_reward, service_bonus, cost_penalty, inv_penalty = self._calculate_proto_reward(report)
        report.proto_reward = proto_reward
        report.service_bonus = service_bonus
        report.cost_penalty = cost_penalty
        report.inventory_penalty = inv_penalty
        
        # Store report
        self.current_report = report
        self.episode_report.add_report(report)
        
        # Increment timestep
        self.timestep += 1
        
        return report
    
    def generate_report(self) -> Optional[ReconciliationReport]:
        """
        Get current reconciliation report.
        
        Returns:
            Most recent ReconciliationReport
        """
        return self.current_report
    
    def get_metrics(self) -> dict:
        """
        Get current metrics.
        
        Returns:
            Dict with all metrics
        """
        return self.metrics_calculator.get_all_metrics()
    
    def get_episode_report(self) -> EpisodeReport:
        """
        Get episode report with aggregations.
        
        Returns:
            EpisodeReport object
        """
        return self.episode_report
    
    def reset(self):
        """Reset engine for new episode."""
        self.current_planned = None
        self.current_actual = None
        self.current_report = None
        self.timestep = 0
        self.metrics_calculator.reset()
        self.episode_report.reset()
