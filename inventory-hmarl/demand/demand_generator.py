"""
demand_generator.py

Stochastic demand generation with seasonal patterns.

Generates daily customer demand with:
- Base demand (configurable mean)
- Seasonal component (sinusoidal pattern)
- Random noise (Gaussian or Poisson)
- Optional demand spikes (scenario-driven)
"""

import numpy as np
from typing import Optional


class DemandGenerator:
    """
    Generates stochastic demand with seasonal patterns.
    
    Formula:
        demand(day) = base_demand 
                      * (1 + seasonality_amplitude * sin(2π * day / period))
                      * demand_multiplier
                      + noise(std_dev)
    
    Attributes:
        base_demand: Average daily demand
        demand_std: Standard deviation for noise
        seasonality_amplitude: Amplitude of seasonal variation (0-1)
        seasonality_period: Period of seasonal cycle in days
        demand_multiplier: Store-specific demand scaling factor
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        base_demand: float = 50.0,
        demand_std: float = 10.0,
        seasonality_amplitude: float = 0.0,
        seasonality_period: int = 30,
        demand_multiplier: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize demand generator.
        
        Args:
            base_demand: Mean daily demand
            demand_std: Standard deviation of demand
            seasonality_amplitude: Seasonal variation amplitude (0 = no seasonality)
            seasonality_period: Seasonal cycle length in days
            demand_multiplier: Scaling factor for this demand stream
            seed: Random seed for reproducibility
        """
        self.base_demand = base_demand
        self.demand_std = demand_std
        self.seasonality_amplitude = seasonality_amplitude
        self.seasonality_period = seasonality_period
        self.demand_multiplier = demand_multiplier
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Spike parameters (controlled by scenarios)
        self.spike_days = set()
        self.spike_multiplier = 1.0
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset the random number generator.
        
        Args:
            seed: New random seed (optional)
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.spike_days = set()
    
    def set_spike(self, days: list, multiplier: float = 2.0):
        """
        Configure demand spike for specific days.
        
        Args:
            days: List of days to spike demand
            multiplier: Demand multiplier for spike days
        """
        self.spike_days = set(days)
        self.spike_multiplier = multiplier
    
    def _seasonal_factor(self, day: int) -> float:
        """
        Calculate seasonal adjustment factor.
        
        Args:
            day: Current simulation day
            
        Returns:
            Seasonal multiplier (1.0 = no adjustment)
        """
        if self.seasonality_amplitude == 0:
            return 1.0
        
        # Sinusoidal pattern
        phase = 2 * np.pi * day / self.seasonality_period
        seasonal_component = self.seasonality_amplitude * np.sin(phase)
        
        return 1.0 + seasonal_component
    
    def generate(self, day: int) -> float:
        """
        Generate demand for a specific day.
        
        Args:
            day: Current simulation day (0-indexed)
            
        Returns:
            Demand value (non-negative)
        """
        # Start with base demand scaled by multiplier
        mean_demand = self.base_demand * self.demand_multiplier
        
        # Apply seasonal adjustment
        seasonal_factor = self._seasonal_factor(day)
        mean_demand *= seasonal_factor
        
        # Add stochastic noise
        noise = self.rng.normal(0, self.demand_std)
        demand = mean_demand + noise
        
        # Apply spike if configured for this day
        if day in self.spike_days:
            demand *= self.spike_multiplier
        
        # Ensure non-negative demand
        demand = max(0, demand)
        
        return demand
    
    def generate_series(self, num_days: int) -> np.ndarray:
        """
        Generate demand for multiple days at once.
        
        Args:
            num_days: Number of days to generate
            
        Returns:
            Array of demand values
        """
        return np.array([self.generate(day) for day in range(num_days)])
    
    def get_expected_demand(self, day: int) -> float:
        """
        Get expected demand (without noise) for a given day.
        Useful for forecasting.
        
        Args:
            day: Simulation day
            
        Returns:
            Expected demand value
        """
        mean_demand = self.base_demand * self.demand_multiplier
        seasonal_factor = self._seasonal_factor(day)
        return mean_demand * seasonal_factor


class MultiSKUDemandGenerator:
    """
    Manages demand generation for multiple SKUs.
    
    Each SKU has its own demand generator with potentially
    different parameters.
    """
    
    def __init__(self, sku_configs: dict, demand_multiplier: float = 1.0, seed: Optional[int] = None):
        """
        Initialize multi-SKU demand generator.
        
        Args:
            sku_configs: Dict mapping SKU_ID -> config dict with demand parameters
            demand_multiplier: Store-level demand multiplier
            seed: Random seed
        """
        self.generators = {}
        
        for sku_id, config in sku_configs.items():
            # Create individual generator for each SKU
            sku_seed = seed + hash(sku_id) % 10000 if seed is not None else None
            
            self.generators[sku_id] = DemandGenerator(
                base_demand=config.get("base_demand", 50.0),
                demand_std=config.get("demand_std", 10.0),
                seasonality_amplitude=config.get("seasonality_amplitude", 0.0),
                seasonality_period=config.get("seasonality_period", 30),
                demand_multiplier=demand_multiplier,
                seed=sku_seed
            )
    
    def generate(self, day: int) -> dict:
        """
        Generate demand for all SKUs for a given day.
        
        Args:
            day: Simulation day
            
        Returns:
            Dict mapping SKU_ID -> demand value
        """
        return {
            sku_id: generator.generate(day)
            for sku_id, generator in self.generators.items()
        }
    
    def set_spike(self, sku_id: str, days: list, multiplier: float = 2.0):
        """
        Set demand spike for a specific SKU.
        
        Args:
            sku_id: SKU identifier
            days: List of spike days
            multiplier: Spike multiplier
        """
        if sku_id in self.generators:
            self.generators[sku_id].set_spike(days, multiplier)
    
    def reset(self, seed: Optional[int] = None):
        """Reset all generators."""
        for sku_id, generator in self.generators.items():
            sku_seed = seed + hash(sku_id) % 10000 if seed is not None else None
            generator.reset(sku_seed)
