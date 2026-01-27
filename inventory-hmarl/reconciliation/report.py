"""
report.py

Structured report objects for reconciliation output.

Provides:
- ReconciliationReport: Single timestep report
- EpisodeReport: Aggregated episode report
- Utility functions for analysis

All reports are machine-readable for downstream consumption.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json
from reconciliation.reason_codes import DeviationReason


@dataclass
class ReconciliationReport:
    """
    Structured reconciliation report for a single timestep.
    
    Contains planned vs actual comparison, deviation attribution,
    metrics, and proto-reward signals.
    """
    
    # Identifiers
    timestep: int
    store_id: str
    sku_id: str
    
    # Planned state
    planned_demand: float
    planned_inventory: float
    planned_order: float
    planned_service_level: float = 0.95
    planned_holding_cost: float = 0.0
    
    # Actual state
    actual_demand: float = 0.0
    actual_inventory: float = 0.0
    actual_order: float = 0.0
    fulfilled_demand: float = 0.0
    lost_sales: float = 0.0
    actual_service_level: float = 0.0
    actual_holding_cost: float = 0.0
    stockout_penalty: float = 0.0
    
    # Deltas (actual - planned)
    demand_delta: float = 0.0
    inventory_delta: float = 0.0
    service_level_delta: float = 0.0
    cost_delta: float = 0.0
    
    # Attribution
    deviation_reasons: List[DeviationReason] = field(default_factory=list)
    primary_reason: Optional[DeviationReason] = None
    reason_severity: str = "info"
    
    # Metrics
    service_level: float = 0.0
    fill_rate: float = 0.0
    total_cost: float = 0.0
    
    # Signals (proto-rewards, not for RL yet)
    proto_reward: float = 0.0
    service_bonus: float = 0.0
    cost_penalty: float = 0.0
    inventory_penalty: float = 0.0
    efficiency_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        d = asdict(self)
        # Convert enums to strings
        d['deviation_reasons'] = [r.value for r in self.deviation_reasons]
        if self.primary_reason:
            d['primary_reason'] = self.primary_reason.value
        return d
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class EpisodeReport:
    """
    Aggregated report for an entire episode.
    
    Collects and analyzes all timestep reports.
    """
    
    def __init__(self):
        """Initialize episode report."""
        self.timestep_reports: List[ReconciliationReport] = []
    
    def add_report(self, report: ReconciliationReport):
        """Add timestep report to episode."""
        self.timestep_reports.append(report)
    
    def aggregate_metrics(self) -> Dict[str, float]:
        """
        Aggregate metrics across all timesteps.
        
        Returns:
            Dict with aggregated metrics
        """
        if not self.timestep_reports:
            return {}
        
        total_demand = sum(r.actual_demand for r in self.timestep_reports)
        total_sales = sum(r.fulfilled_demand for r in self.timestep_reports)
        total_lost_sales = sum(r.lost_sales for r in self.timestep_reports)
        total_holding_cost = sum(r.actual_holding_cost for r in self.timestep_reports)
        total_stockout = sum(r.stockout_penalty for r in self.timestep_reports)
        total_cost = sum(r.total_cost for r in self.timestep_reports)
        
        avg_service_level = total_sales / total_demand if total_demand > 0 else 0
        avg_fill_rate = 1 - (total_lost_sales / total_demand) if total_demand > 0 else 1
        
        avg_proto_reward = sum(r.proto_reward for r in self.timestep_reports) / len(self.timestep_reports)
        avg_efficiency = sum(r.efficiency_score for r in self.timestep_reports) / len(self.timestep_reports)
        
        return {
            'total_timesteps': len(self.timestep_reports),
            'total_demand': total_demand,
            'total_sales': total_sales,
            'total_lost_sales': total_lost_sales,
            'avg_service_level': avg_service_level,
            'avg_fill_rate': avg_fill_rate,
            'total_holding_cost': total_holding_cost,
            'total_stockout_penalty': total_stockout,
            'total_cost': total_cost,
            'avg_proto_reward': avg_proto_reward,
            'avg_efficiency_score': avg_efficiency
        }
    
    def get_reason_distribution(self) -> Dict[str, int]:
        """
        Get distribution of deviation reasons across episode.
        
        Returns:
            Dict mapping reason -> count
        """
        reason_counts: Dict[str, int] = {}
        
        for report in self.timestep_reports:
            for reason in report.deviation_reasons:
                reason_str = reason.value
                reason_counts[reason_str] = reason_counts.get(reason_str, 0) + 1
        
        return reason_counts
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """
        Get cost breakdown across episode.
        
        Returns:
            Dict with cost components
        """
        total_holding = sum(r.actual_holding_cost for r in self.timestep_reports)
        total_stockout = sum(r.stockout_penalty for r in self.timestep_reports)
        total_cost = total_holding + total_stockout
        
        if total_cost == 0:
            return {
                'holding_cost': 0.0,
                'stockout_penalty': 0.0,
                'holding_pct': 0.0,
                'stockout_pct': 0.0
            }
        
        return {
            'holding_cost': total_holding,
            'stockout_penalty': total_stockout,
            'total_cost': total_cost,
            'holding_pct': 100 * total_holding / total_cost,
            'stockout_pct': 100 * total_stockout / total_cost
        }
    
    def get_service_level_progression(self) -> List[float]:
        """
        Get service level over time.
        
        Returns:
            List of service levels per timestep
        """
        return [r.actual_service_level for r in self.timestep_reports]
    
    def get_proto_reward_progression(self) -> List[float]:
        """
        Get proto-reward signal over time.
        
        Returns:
            List of proto-rewards per timestep
        """
        return [r.proto_reward for r in self.timestep_reports]
    
    def get_primary_reasons_over_time(self) -> List[Optional[str]]:
        """
        Get primary deviation reason for each timestep.
        
        Returns:
            List of primary reasons (as strings)
        """
        return [
            r.primary_reason.value if r.primary_reason else None
            for r in self.timestep_reports
        ]
    
    def print_summary(self):
        """Print episode summary to console."""
        metrics = self.aggregate_metrics()
        reason_dist = self.get_reason_distribution()
        cost_breakdown = self.get_cost_breakdown()
        
        print("\n" + "="*70)
        print("RECONCILIATION EPISODE SUMMARY")
        print("="*70)
        
        print(f"\nTimesteps: {metrics['total_timesteps']}")
        
        print("\nSERVICE METRICS:")
        print(f"  Average Service Level: {metrics['avg_service_level']:.2%}")
        print(f"  Average Fill Rate: {metrics['avg_fill_rate']:.2%}")
        print(f"  Total Lost Sales: {metrics['total_lost_sales']:,.0f} units")
        
        print("\nCOST METRICS:")
        print(f"  Total Cost: ${metrics['total_cost']:,.2f}")
        print(f"  - Holding: ${cost_breakdown['holding_cost']:,.2f} ({cost_breakdown['holding_pct']:.1f}%)")
        print(f"  - Stockout: ${cost_breakdown['stockout_penalty']:,.2f} ({cost_breakdown['stockout_pct']:.1f}%)")
        
        print("\nPERFORMANCE SIGNALS:")
        print(f"  Average Proto-Reward: {metrics['avg_proto_reward']:.2f}")
        print(f"  Average Efficiency Score: {metrics['avg_efficiency_score']:.1f}/100")
        
        print("\nDEVIATION REASONS (Top 5):")
        sorted_reasons = sorted(reason_dist.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_reasons[:5]:
            print(f"  {reason}: {count} occurrences")
        
        print("="*70)
        print()
    
    def to_dict(self) -> dict:
        """Convert episode report to dictionary."""
        return {
            'metrics': self.aggregate_metrics(),
            'reason_distribution': self.get_reason_distribution(),
            'cost_breakdown': self.get_cost_breakdown(),
            'service_level_progression': self.get_service_level_progression(),
            'proto_reward_progression': self.get_proto_reward_progression(),
            'primary_reasons': self.get_primary_reasons_over_time(),
            'timestep_count': len(self.timestep_reports)
        }
    
    def to_json(self) -> str:
        """Convert episode report to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    def reset(self):
        """Reset for new episode."""
        self.timestep_reports.clear()
