"""
simulation_config.py

Central configuration file for the Digital Twin–based
Inventory Optimization Simulation.

This file contains ONLY constants and dictionaries.
No functions, no cross-file imports.
"""

import numpy as np

# =========================
# GLOBAL SIMULATION CONTROL
# =========================

SIMULATION_NAME = "Digital_Twin_Retail_Supply_Chain"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TIME_STEP = "day"          # simulation granularity
SIMULATION_DAYS = 90       # total horizon (can be increased to 180)
WARMUP_DAYS = 7            # initial days ignored for metrics

# =========================
# SUPPLY CHAIN TOPOLOGY
# =========================

NUM_STORES = 3
NUM_WAREHOUSES = 1
NUM_SUPPLIERS = 1
NUM_SKUS = 1  # Start with 1 SKU, can extend to 5

# Network structure: who supplies whom
SUPPLY_CHAIN_GRAPH = {
    "supplier": ["warehouse_1"],
    "warehouse_1": ["store_1", "store_2", "store_3"]
}

# =========================
# SKU DEFINITIONS
# =========================

# For hackathon simplicity, start with 1 SKU
# Can be extended to 5 SKUs as per initial requirements
SKU_CONFIG = {
    "SKU_001": {
        "name": "Premium Coffee Beans",
        "base_demand": 50,           # average daily demand
        "demand_std": 15,            # standard deviation
        "seasonality_amplitude": 0.3,  # ±30% seasonal variation
        "seasonality_period": 30,    # 30-day cycle
        "holding_cost": 0.5,         # cost per unit per day
        "stockout_penalty": 10.0,    # penalty per lost sale
        "ordering_cost": 25.0,       # fixed cost per order
        "unit_cost": 12.50,          # supplier cost
        "unit_price": 24.99          # retail price
    }
}

# =========================
# STORE CONFIGURATION
# =========================

STORE_CONFIG = {
    "store_1": {
        "name": "Downtown Store",
        "demand_multiplier": 1.2,    # higher demand
        "initial_inventory": 700,    # ~14 days of demand
        "max_inventory": 2000,
        "reorder_point": 300,        # (s,S) policy
        "order_up_to": 800
    },
    "store_2": {
        "name": "Suburban Store",
        "demand_multiplier": 1.0,    # baseline demand
        "initial_inventory": 600,
        "max_inventory": 1500,
        "reorder_point": 250,
        "order_up_to": 700
    },
    "store_3": {
        "name": "Mall Store",
        "demand_multiplier": 1.5,    # highest demand
        "initial_inventory": 900,
        "max_inventory": 2500,
        "reorder_point": 400,
        "order_up_to": 1000
    }
}

# =========================
# WAREHOUSE CONFIGURATION
# =========================

WAREHOUSE_CONFIG = {
    "warehouse_1": {
        "name": "Central Distribution Center",
        "initial_inventory": 5000,   # ~30 days of aggregate demand
        "max_inventory": 15000,
        "reorder_point": 2000,
        "order_up_to": 6000,
        "lead_time_to_stores": 2     # days to deliver to stores
    }
}

# =========================
# SUPPLIER CONFIGURATION
# =========================

SUPPLIER_CONFIG = {
    "supplier_1": {
        "name": "Primary Supplier",
        "lead_time": 7,              # days to fulfill orders
        "reliability": 0.98,         # 98% order fulfillment
        "infinite_supply": True      # simplified for hackathon
    }
}

# =========================
# DEMAND GENERATION CONFIG
# =========================

DEMAND_CONFIG = {
    "distribution": "normal",        # normal / poisson
    "noise_type": "gaussian",        # gaussian / none
    "enable_seasonality": True,
    "enable_spikes": False,         # controlled by scenarios
    "spike_probability": 0.0,
    "spike_multiplier": 2.0
}

# =========================
# COST PARAMETERS
# =========================

COST_CONFIG = {
    "holding_cost_weight": 1.0,
    "stockout_penalty_weight": 1.0,
    "ordering_cost_weight": 1.0
}

# =========================
# SERVICE LEVEL TARGETS
# =========================

SERVICE_LEVEL_CONFIG = {
    "target_service_level": 0.95,   # 95% of demand met
    "target_fill_rate": 0.98,       # 98% from stock
    "target_inventory_turns": 12    # annual turns
}

# =========================
# LOGGING & OUTPUT
# =========================

LOGGING_CONFIG = {
    "log_level": "INFO",
    "save_metrics": True,
    "metrics_path": "outputs/metrics/",
    "save_state_history": False,    # save full state history (expensive)
    "print_daily_summary": False,   # print each day (verbose)
    "print_final_summary": True
}

# =========================
# SCENARIO DEFAULTS
# =========================

SCENARIO_CONFIG = {
    "default_scenario": "normal",
    "available_scenarios": [
        "normal",
        "demand_spike",
        "strong_seasonality",
        "supplier_delay",
        "high_variability"
    ]
}
