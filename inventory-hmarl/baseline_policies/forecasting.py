"""
forecasting.py

Simple statistical demand forecasting methods for baseline policies.

Methods:
- Moving Average: Average of last N days
- Exponential Smoothing: Weighted average with decay
- Seasonal Average: Average for same day-of-cycle

All methods are deterministic and do not involve learning.
"""

import numpy as np
from typing import List, Optional
from collections import deque


class Forecaster:
    """
    Base forecaster class.
    
    All forecasters maintain:
    - Demand history
    - Current forecast
    - Demand standard deviation (for safety stock)
    """
    
    def __init__(self, initial_forecast: float = 50.0, initial_std: float = 10.0):
        """
        Initialize forecaster.
        
        Args:
            initial_forecast: Starting forecast value
            initial_std: Starting standard deviation
        """
        self.forecast = initial_forecast
        self.std_dev = initial_std
        self.history = deque()
    
    def update(self, actual_demand: float):
        """
        Update forecaster with actual demand observation.
        
        Args:
            actual_demand: Realized demand value
        """
        raise NotImplementedError("Subclass must implement update()")
    
    def predict(self, horizon: int = 1) -> float:
        """
        Generate forecast for future period.
        
        Args:
            horizon: Number of periods ahead (default: 1 day)
            
        Returns:
            Forecasted demand
        """
        # For stationary methods, forecast is same for all horizons
        return self.forecast
    
    def get_std_dev(self) -> float:
        """
        Get current estimate of demand standard deviation.
        Used for safety stock calculation.
        
        Returns:
            Standard deviation of demand
        """
        return self.std_dev
    
    def get_history(self) -> List[float]:
        """Get demand history."""
        return list(self.history)


class MovingAverageForecaster(Forecaster):
    """
    Moving average forecaster.
    
    Forecast = average of last N observations
    Std Dev = standard deviation of last N observations
    """
    
    def __init__(
        self,
        window: int = 7,
        initial_forecast: float = 50.0,
        initial_std: float = 10.0
    ):
        """
        Initialize moving average forecaster.
        
        Args:
            window: Number of periods to average
            initial_forecast: Starting forecast
            initial_std: Starting std deviation
        """
        super().__init__(initial_forecast, initial_std)
        self.window = window
        self.history = deque(maxlen=window)
    
    def update(self, actual_demand: float):
        """Update with new demand observation."""
        self.history.append(actual_demand)
        
        if len(self.history) >= 2:
            # Update forecast and std dev
            self.forecast = np.mean(self.history)
            self.std_dev = np.std(self.history)
            
            # Ensure minimum std dev to avoid division by zero
            self.std_dev = max(self.std_dev, 1.0)
    
    def predict(self, horizon: int = 1) -> float:
        """Return current moving average."""
        return self.forecast


class ExponentialSmoothingForecaster(Forecaster):
    """
    Exponential smoothing forecaster.
    
    Forecast[t] = α × Actual[t-1] + (1-α) × Forecast[t-1]
    
    Where α (alpha) controls responsiveness:
    - High α: More responsive to recent changes
    - Low α: More stable, ignores noise
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        initial_forecast: float = 50.0,
        initial_std: float = 10.0,
        history_length: int = 30
    ):
        """
        Initialize exponential smoothing forecaster.
        
        Args:
            alpha: Smoothing parameter (0-1)
            initial_forecast: Starting forecast
            initial_std: Starting std deviation
            history_length: How much history to keep for std calculation
        """
        super().__init__(initial_forecast, initial_std)
        
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        self.alpha = alpha
        self.history = deque(maxlen=history_length)
    
    def update(self, actual_demand: float):
        """Update forecast using exponential smoothing."""
        self.history.append(actual_demand)
        
        # Exponential smoothing update
        self.forecast = self.alpha * actual_demand + (1 - self.alpha) * self.forecast
        
        # Update std dev from history
        if len(self.history) >= 2:
            self.std_dev = np.std(self.history)
            self.std_dev = max(self.std_dev, 1.0)
    
    def predict(self, horizon: int = 1) -> float:
        """Return current smoothed forecast."""
        return self.forecast


class SeasonalAverageForecaster(Forecaster):
    """
    Seasonal average forecaster.
    
    Forecast = average of demand on same day-of-cycle in history
    
    Example: For 7-day weekly seasonality, forecast for Monday
    is the average of all previous Mondays.
    """
    
    def __init__(
        self,
        period: int = 7,
        initial_forecast: float = 50.0,
        initial_std: float = 10.0
    ):
        """
        Initialize seasonal forecaster.
        
        Args:
            period: Seasonal cycle length (e.g., 7 for weekly)
            initial_forecast: Starting forecast
            initial_std: Starting std deviation
        """
        super().__init__(initial_forecast, initial_std)
        self.period = period
        
        # Store demand by day-of-cycle
        # seasonal_data[day_of_cycle] = list of observations
        self.seasonal_data = {i: [] for i in range(period)}
        
        # Track current day in cycle
        self.current_day = 0
        
        # Full history for overall std dev
        self.history = deque(maxlen=period * 10)
    
    def update(self, actual_demand: float):
        """Update with new demand observation."""
        day_of_cycle = self.current_day % self.period
        
        self.seasonal_data[day_of_cycle].append(actual_demand)
        self.history.append(actual_demand)
        
        # Update forecast for current day-of-cycle
        if len(self.seasonal_data[day_of_cycle]) > 0:
            self.forecast = np.mean(self.seasonal_data[day_of_cycle])
        
        # Update std dev from full history
        if len(self.history) >= 2:
            self.std_dev = np.std(self.history)
            self.std_dev = max(self.std_dev, 1.0)
        
        # Advance day counter
        self.current_day += 1
    
    def predict(self, horizon: int = 1) -> float:
        """
        Predict demand for future day.
        
        Args:
            horizon: Days ahead (1 = tomorrow)
            
        Returns:
            Forecast for that day-of-cycle
        """
        future_day = (self.current_day + horizon - 1) % self.period
        
        if len(self.seasonal_data[future_day]) > 0:
            return np.mean(self.seasonal_data[future_day])
        else:
            # No data for this day-of-cycle yet, use overall average
            return self.forecast


def create_forecaster(method: str, config: dict) -> Forecaster:
    """
    Factory function to create forecaster from configuration.
    
    Args:
        method: Forecasting method ('moving_average', 'exp_smoothing', 'seasonal')
        config: Configuration dict with method-specific parameters
        
    Returns:
        Forecaster instance
        
    Raises:
        ValueError: If method is unknown
    """
    initial_forecast = config.get('initial_forecast', 50.0)
    initial_std = config.get('initial_std', 10.0)
    
    if method == 'moving_average':
        window = config.get('forecast_window', 7)
        return MovingAverageForecaster(
            window=window,
            initial_forecast=initial_forecast,
            initial_std=initial_std
        )
    
    elif method == 'exp_smoothing':
        alpha = config.get('exp_smoothing_alpha', 0.3)
        return ExponentialSmoothingForecaster(
            alpha=alpha,
            initial_forecast=initial_forecast,
            initial_std=initial_std
        )
    
    elif method == 'seasonal':
        period = config.get('seasonal_period', 7)
        return SeasonalAverageForecaster(
            period=period,
            initial_forecast=initial_forecast,
            initial_std=initial_std
        )
    
    else:
        raise ValueError(f"Unknown forecasting method: {method}")
