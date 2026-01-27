"""
policy_config.py

Configuration parameters for baseline inventory control policies.

Contains deterministic parameters for:
- Demand forecasting
- Reorder point calculations
- Safety stock levels
- Order quantity policies
"""

# =========================
# STORE-LEVEL POLICY CONFIG
# =========================

STORE_POLICY_CONFIG = {
    # Forecasting parameters
    'forecast_method': 'moving_average',  # 'moving_average', 'exp_smoothing', 'seasonal'
    'forecast_window': 7,                  # Days for moving average
    'exp_smoothing_alpha': 0.3,            # Alpha for exponential smoothing (0-1)
    'seasonal_period': 7,                  # Period for seasonal forecasting
    
    # Reorder point parameters
    'z_score': 1.65,                       # Safety factor for 95% service level
                                           # 1.28 = 90%, 1.65 = 95%, 1.96 = 97.5%, 2.33 = 99%
    'lead_time': 2,                        # Lead time from warehouse (days)
    
    # Order quantity parameters
    'days_of_cover': 14,                   # Target inventory in days
    'min_order_quantity': 0,               # Minimum order size
    'max_order_quantity': 1000,            # Maximum order size (safety constraint)
    
    # Initial values
    'initial_forecast': 50.0,              # Starting forecast if no history
    'initial_std': 10.0                    # Starting standard deviation
}


# =========================
# WAREHOUSE-LEVEL POLICY CONFIG
# =========================

WAREHOUSE_POLICY_CONFIG = {
    # Forecasting parameters (aggregated from stores)
    'forecast_method': 'moving_average',
    'forecast_window': 14,                 # Longer window for stability
    'exp_smoothing_alpha': 0.2,            # More conservative smoothing
    
    # Reorder point parameters
    'z_score': 1.96,                       # Higher service level (97.5%)
    'lead_time': 7,                        # Supplier lead time (days)
    
    # Order quantity parameters
    'days_of_cover': 30,                   # Longer coverage for warehouse
    'min_order_quantity': 100,             # Economies of scale
    'max_order_quantity': 10000,           # Warehouse capacity constraint
    
    # Initial values
    'initial_forecast': 150.0,             # Aggregate of 3 stores
    'initial_std': 30.0
}


# =========================
# LOGGING CONFIGURATION
# =========================

LOGGING_CONFIG = {
    'log_to_console': True,
    'log_to_file': True,
    'log_directory': 'outputs/baseline_logs/',
    'log_formats': ['json', 'csv'],        # Output formats
    'save_frequency': 1,                   # Save every N days (1 = every day)
    'verbose': False                       # Print detailed logs
}


# =========================
# SAFETY STOCK Z-SCORES
# =========================
# Reference table for service level targets

Z_SCORE_TABLE = {
    '90%': 1.28,
    '95%': 1.65,
    '97.5%': 1.96,
    '99%': 2.33,
    '99.5%': 2.58,
    '99.9%': 3.09
}


# =========================
# FORECASTING METHOD INFO
# =========================

FORECAST_METHOD_INFO = {
    'moving_average': {
        'description': 'Simple average of last N days',
        'parameters': ['forecast_window'],
        'best_for': 'Stable demand with low variability'
    },
    'exp_smoothing': {
        'description': 'Exponentially weighted moving average',
        'parameters': ['exp_smoothing_alpha'],
        'best_for': 'Demand with gradual trends'
    },
    'seasonal': {
        'description': 'Average of same day-of-cycle in history',
        'parameters': ['seasonal_period'],
        'best_for': 'Demand with clear weekly/monthly patterns'
    }
}
