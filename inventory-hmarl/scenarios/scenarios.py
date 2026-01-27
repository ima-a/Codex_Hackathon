"""
scenarios.py

Pre-defined simulation scenarios for testing the digital twin.

Each scenario can:
- Modify demand patterns
- Inject demand spikes
- Simulate disruptions
"""

from typing import Optional


class Scenario:
    """Base class for simulation scenarios."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize scenario.
        
        Args:
            name: Scenario name
            description: Human-readable description
        """
        self.name = name
        self.description = description
    
    def apply_to_environment(self, env):
        """
        Apply scenario modifications to digital twin environment.
        
        Args:
            env: DigitalTwin instance
        """
        pass  # Override in subclasses


class NormalOperations(Scenario):
    """Baseline scenario with normal demand patterns."""
    
    def __init__(self):
        super().__init__(
            "Normal Operations",
            "Standard stochastic and seasonal demand, no disruptions"
        )
    
    def apply_to_environment(self, env):
        """No modifications needed for normal operations."""
        pass


class DemandSpike(Scenario):
    """Scenario with sudden demand spikes on specific days."""
    
    def __init__(self, spike_days: Optional[list] = None, multiplier: float = 2.5):
        """
        Initialize demand spike scenario.
        
        Args:
            spike_days: List of days to spike demand (default: [30, 31, 32])
            multiplier: Demand multiplier for spike days
        """
        if spike_days is None:
            spike_days = [30, 31, 32]
        
        super().__init__(
            "Demand Spike",
            f"Demand spike on days {spike_days} (x{multiplier})"
        )
        self.spike_days = spike_days
        self.multiplier = multiplier
    
    def apply_to_environment(self, env):
        """Apply demand spikes to all stores."""
        for store_id, generator in env.demand_generators.items():
            for sku_id in generator.generators.keys():
                generator.set_spike(sku_id, self.spike_days, self.multiplier)


class StrongSeasonality(Scenario):
    """Scenario with amplified seasonal patterns."""
    
    def __init__(self, amplitude: float = 0.5):
        """
        Initialize strong seasonality scenario.
        
        Args:
            amplitude: Seasonal amplitude (0-1, higher = more variation)
        """
        super().__init__(
            "Strong Seasonality",
            f"Amplified seasonal pattern (amplitude: {amplitude})"
        )
        self.amplitude = amplitude
    
    def apply_to_environment(self, env):
        """Increase seasonality amplitude in demand generators."""
        for store_id, generator in env.demand_generators.items():
            for sku_id, demand_gen in generator.generators.items():
                demand_gen.seasonality_amplitude = self.amplitude


class SupplierDelay(Scenario):
    """Scenario simulating supplier delays."""
    
    def __init__(self, additional_lead_time: int = 7):
        """
        Initialize supplier delay scenario.
        
        Args:
            additional_lead_time: Extra days added to supplier lead time
        """
        super().__init__(
            "Supplier Delay",
            f"Supplier lead time increased by {additional_lead_time} days"
        )
        self.additional_lead_time = additional_lead_time
    
    def apply_to_environment(self, env):
        """Increase supplier lead times."""
        for supplier in env.suppliers:
            supplier.lead_time += self.additional_lead_time


class HighVariability(Scenario):
    """Scenario with increased demand variability."""
    
    def __init__(self, std_multiplier: float = 2.0):
        """
        Initialize high variability scenario.
        
        Args:
            std_multiplier: Multiplier for demand standard deviation
        """
        super().__init__(
            "High Variability",
            f"Demand variability increased (x{std_multiplier})"
        )
        self.std_multiplier = std_multiplier
    
    def apply_to_environment(self, env):
        """Increase demand standard deviation."""
        for store_id, generator in env.demand_generators.items():
            for sku_id, demand_gen in generator.generators.items():
                demand_gen.demand_std *= self.std_multiplier


# Scenario registry for easy access
SCENARIOS = {
    'normal': NormalOperations(),
    'spike': DemandSpike(),
    'strong_seasonality': StrongSeasonality(),
    'supplier_delay': SupplierDelay(),
    'high_variability': HighVariability()
}


def get_scenario(scenario_name: str) -> Scenario:
    """
    Get scenario by name.
    
    Args:
        scenario_name: Name of scenario
        
    Returns:
        Scenario instance
        
    Raises:
        ValueError: If scenario not found
    """
    if scenario_name not in SCENARIOS:
        available = ', '.join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    
    return SCENARIOS[scenario_name]
