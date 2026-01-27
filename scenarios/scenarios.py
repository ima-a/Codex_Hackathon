"""
scenarios.py

Pre-defined simulation scenarios for testing the HMARL inventory system.

Each scenario can:
- Modify global config
- Inject special demand events
- Simulate supply chain disruptions
"""

from config.simulation_config import DEMAND_CONFIG, INVENTORY_CONFIG
import numpy as np

class Scenario:
    """Base class for scenarios."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def modify_config(self, config):
        """
        Override to modify simulation config dynamically.
        Return modified config.
        """
        return config

    def modify_demand(self, day: int, demand_generator):
        """
        Override to inject custom demand events.
        Default: no changes.
        """
        return demand_generator.generate(day)


# -----------------------------
# Standard Scenarios
# -----------------------------

class NormalOperations(Scenario):
    """Baseline: normal stochastic and seasonal demand."""
    def __init__(self):
        super().__init__("Normal Operations", "Standard demand patterns, no disruptions")


class DemandSpike(Scenario):
    """Spike in demand on specific days."""
    def __init__(self, spike_days=None, magnitude=2.0):
        if spike_days is None:
            spike_days = [30, 31, 32]  # default spike
        super().__init__("Demand Spike", f"Demand spike on days {spike_days}")
        self.spike_days = spike_days
        self.magnitude = magnitude

    def modify_demand(self, day, demand_generator):
        demand = demand_generator.generate(day)
        if day in self.spike_days:
            # Scale demand for spike
            demand *= self.magnitude
        return demand


class SupplierDelay(Scenario):
    """Simulate supplier delays affecting warehouse inventory."""
    def __init__(self, delay_days=None, delay_duration=7):
        if delay_days is None:
            delay_days = [40, 50]
        super().__init__("Supplier Delay", f"Delays on days {delay_days}")
        self.delay_days = delay_days
        self.delay_duration = delay_duration

    def modify_config(self, config):
        # Increase lead times temporarily
        modified_config = config.copy()
        for day in self.delay_days:
            modified_config['lead_time_adjustment'] = {
                'warehouse_to_retailer': self.delay_duration
            }
        return modified_config


class SeasonalPeak(Scenario):
    """Stronger seasonal effect than normal."""
    def __init__(self, amplitude=0.4, period=7):
        super().__init__("Seasonal Peak", "Higher amplitude seasonality")
        self.amplitude = amplitude
        self.period = period

    def modify_config(self, config):
        modified_config = config.copy()
        modified_config['SEASONAL_AMPLITUDE'] = self.amplitude
        modified_config['SEASONAL_PERIOD'] = self.period
        return modified_config


# -----------------------------
# Scenario Registry
# -----------------------------

SCENARIOS = {
    "normal": NormalOperations(),
    "spike": DemandSpike(),
    "delay": SupplierDelay(),
    "seasonal": SeasonalPeak()
}
