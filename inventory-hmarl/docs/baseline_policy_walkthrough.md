# Baseline Policy Layer - Complete Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)
6. [Testing & Validation](#testing--validation)

---

## Overview

The Baseline Policy Layer implements **classical deterministic inventory control** using Reorder Point (ROP) and safety stock formulas. This provides:
- **Benchmark performance** for comparison with learning-based agents
- **Training data generation** with state-action-outcome trajectories
- **Explainable decisions** using textbook inventory management
- **Stable baseline** before introducing RL complexity

### Key Characteristics
- **Deterministic**: Same inputs → same outputs
- **Non-learning**: Fixed rules, no adaptation
- **Classical methods**: ROP, safety stock, forecasting
- **Fully logged**: Complete state-action-outcome history

---

## Architecture

### System Components

```
baseline_policies/
├── policy_config.py          # Configuration parameters
├── forecasting.py            # Demand forecasting methods
├── store_policy.py           # Store-level ROP policy
├── warehouse_policy.py       # Warehouse-level ROP policy
├── policy_logger.py          # State-action-outcome logging
└── baseline_runner.py        # Integration runner
```

### Data Flow

```
┌─────────────────────────────────────────────┐
│         Baseline Policy System              │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │      Demand History                  │  │
│  └──────────────┬───────────────────────┘  │
│                 ▼                           │
│  ┌──────────────────────────────────────┐  │
│  │      Forecasters                     │  │
│  │  • Moving Average                    │  │
│  │  • Exponential Smoothing             │  │
│  │  • Seasonal Average                  │  │
│  └──────────────┬───────────────────────┘  │
│                 ▼                           │
│  ┌──────────────────────────────────────┐  │
│  │      ROP Policies                    │  │
│  │  • Store Policies (s,S)              │  │
│  │  • Warehouse Policies (s,S)          │  │
│  └──────────────┬───────────────────────┘  │
│                 ▼                           │
│  ┌──────────────────────────────────────┐  │
│  │      Order Decisions                 │  │
│  └──────────────┬───────────────────────┘  │
│                 ▼                           │
│  ┌──────────────────────────────────────┐  │
│  │      Policy Logger                   │  │
│  │  • State logs                        │  │
│  │  • Action logs                       │  │
│  │  • Outcome logs                      │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

---

## Core Components

### 1. Forecasting Methods
**File**: [forecasting.py](file:///d:/Codex%20Hackathon/Codex_Hackathon/inventory-hmarl/baseline_policies/forecasting.py)

**Purpose**: Simple statistical demand forecasting (no ML)

#### Moving Average Forecaster

**Formula**:
```python
forecast = mean(demand[-window:])
std_dev = std(demand[-window:])
```

**Implementation**:
```python
class MovingAverageForecaster(Forecaster):
    def __init__(self, window=7, initial_forecast=50.0, initial_std=10.0):
        super().__init__(initial_forecast, initial_std)
        self.window = window
        self.history = deque(maxlen=window)
    
    def update(self, actual_demand):
        """Update with new observation."""
        self.history.append(actual_demand)
        
        if len(self.history) >= 2:
            self.forecast = np.mean(self.history)
            self.std_dev = np.std(self.history)
            self.std_dev = max(self.std_dev, 1.0)  # Minimum std
    
    def predict(self, horizon=1):
        """Return current forecast."""
        return self.forecast
```

**Characteristics**:
- Window size: 7 days (default)
- Equal weight to all observations in window
- Good for stable demand
- Quick to respond but can be noisy

---

#### Exponential Smoothing Forecaster

**Formula**:
```python
forecast[t] = α × actual[t-1] + (1-α) × forecast[t-1]
```

**Implementation**:
```python
class ExponentialSmoothingForecaster(Forecaster):
    def __init__(self, alpha=0.3, initial_forecast=50.0):
        super().__init__(initial_forecast)
        self.alpha = alpha  # Smoothing parameter (0-1)
    
    def update(self, actual_demand):
        """Exponential smoothing update."""
        self.history.append(actual_demand)
        
        # Update forecast
        self.forecast = (
            self.alpha * actual_demand + 
            (1 - self.alpha) * self.forecast
        )
        
        # Update std from history
        if len(self.history) >= 2:
            self.std_dev = np.std(self.history)
```

**Parameters**:
- α (alpha): Smoothing factor
  - High α (0.5-0.8): Responsive to changes
  - Low α (0.1-0.3): Stable, ignores noise
- Default: α = 0.3 (balanced)

**Characteristics**:
- More weight to recent observations
- Good for demand with gradual trends
- Smoother than moving average

---

#### Seasonal Average Forecaster

**Formula**:
```python
forecast[day] = mean(demand on same day-of-cycle in history)
```

**Implementation**:
```python
class SeasonalAverageForecaster(Forecaster):
    def __init__(self, period=7):
        super().__init__()
        self.period = period  # Cycle length (e.g., 7 for weekly)
        self.seasonal_data = {i: [] for i in range(period)}
        self.current_day = 0
    
    def update(self, actual_demand):
        """Update seasonal data."""
        day_of_cycle = self.current_day % self.period
        self.seasonal_data[day_of_cycle].append(actual_demand)
        
        # Update forecast for this day-of-cycle
        if len(self.seasonal_data[day_of_cycle]) > 0:
            self.forecast = np.mean(self.seasonal_data[day_of_cycle])
        
        self.current_day += 1
    
    def predict(self, horizon=1):
        """Predict for future day-of-cycle."""
        future_day = (self.current_day + horizon - 1) % self.period
        
        if len(self.seasonal_data[future_day]) > 0:
            return np.mean(self.seasonal_data[future_day])
        return self.forecast
```

**Characteristics**:
- Period: 7 days (weekly), 30 days (monthly)
- Good for clear seasonal patterns
- Requires sufficient history (multiple cycles)

---

### 2. Store-Level ROP Policy
**File**: [store_policy.py](file:///d:/Codex%20Hackathon/Codex_Hackathon/inventory-hmarl/baseline_policies/store_policy.py)

**Purpose**: Deterministic ordering decisions for retail stores

#### ROP Formula

**Reorder Point Calculation**:
```python
# Expected demand during lead time
expected_demand_LT = forecast × lead_time

# Safety stock
safety_stock = z_score × demand_std × sqrt(lead_time)

# Reorder point
ROP = expected_demand_LT + safety_stock
```

**Components Explained**:
- **Expected Demand**: What we expect to sell during replenishment lead time
- **Safety Stock**: Buffer against demand variability
- **Z-score**: Service level target
  - 1.28 → 90% service
  - 1.65 → 95% service
  - 1.96 → 97.5% service
  - 2.33 → 99% service

#### Order Quantity Calculation

**Logic**:
```python
if inventory <= ROP:
    target_inventory = days_of_cover × forecast
    order_quantity = max(0, target - inventory)
else:
    order_quantity = 0
```

**Parameters**:
- **Days of Cover**: How many days of demand to stock (default: 14)
- **Target Inventory**: Desired inventory level after order arrives
- **Order Quantity**: Amount to bring inventory to target

#### Implementation

```python
class StorePolicy:
    def __init__(self, store_id, sku_id, config):
        self.store_id = store_id
        self.sku_id = sku_id
        
        # Create forecaster
        forecast_method = config['forecast_method']
        self.forecaster = create_forecaster(forecast_method, config)
        
        # Policy parameters
        self.z_score = config['z_score']              # 1.65 (95%)
        self.lead_time = config['lead_time']          # 2 days
        self.days_of_cover = config['days_of_cover']  # 14 days
    
    def calculate_reorder_point(self):
        """Calculate ROP using formula."""
        expected_demand_lt = self.current_forecast * self.lead_time
        self.current_safety_stock = (
            self.z_score * 
            self.current_demand_std * 
            np.sqrt(self.lead_time)
        )
        self.current_reorder_point = expected_demand_lt + self.current_safety_stock
        return self.current_reorder_point
    
    def calculate_order_quantity(self):
        """Calculate order using (s,S) logic."""
        rop = self.calculate_reorder_point()
        
        if self.current_inventory <= rop:
            target_inventory = self.days_of_cover * self.current_forecast
            order_qty = target_inventory - self.current_inventory
            order_qty = max(0, min(order_qty, self.max_order_qty))
            return order_qty
        
        return 0
    
    def decide(self, inventory):
        """Make ordering decision."""
        self.update_state(inventory)
        return self.calculate_order_quantity()
```

**Configuration**:
```python
STORE_POLICY_CONFIG = {
    'forecast_method': 'moving_average',
    'forecast_window': 7,
    'z_score': 1.65,           # 95% service level
    'lead_time': 2,            # Days from warehouse
    'days_of_cover': 14,       # Target inventory
    'exp_smoothing_alpha': 0.3
}
```

---

### 3. Warehouse-Level ROP Policy
**File**: [warehouse_policy.py](file:///d:/Codex%20Hackathon/Codex_Hackathon/inventory-hmarl/baseline_policies/warehouse_policy.py)

**Purpose**: Aggregate ordering from supplier

#### Key Differences from Store Policy

1. **Inventory Position** (not just on-hand):
```python
inventory_position = on_hand + in_transit - backorders
```

2. **Aggregate Demand**:
```python
total_forecast = sum(store_forecasts)
total_std = sqrt(sum(store_stds²))
```

3. **Longer Lead Time**: 7 days (supplier) vs 2 days (warehouse-to-store)

4. **Higher Service Level**: Z = 1.96 (97.5%) vs 1.65 (95%)

5. **More Days of Cover**: 30 days vs 14 days

#### Implementation

```python
class WarehousePolicy:
    def calculate_inventory_position(self):
        """Account for pipeline inventory."""
        return (
            self.current_inventory + 
            self.current_in_transit - 
            self.current_backorders
        )
    
    def calculate_reorder_point(self):
        """ROP using supplier lead time."""
        expected_demand_lt = self.current_forecast * self.lead_time  # 7 days
        self.current_safety_stock = (
            self.z_score *  # 1.96 (97.5% service)
            self.current_demand_std * 
            np.sqrt(self.lead_time)
        )
        self.current_reorder_point = expected_demand_lt + self.current_safety_stock
        return self.current_reorder_point
    
    def calculate_order_quantity(self):
        """Order based on inventory position."""
        inv_position = self.calculate_inventory_position()
        rop = self.calculate_reorder_point()
        
        if inv_position <= rop:
            target = self.days_of_cover * self.current_forecast  # 30 days
            order_qty = target - inv_position
            order_qty = max(self.min_order_qty, order_qty)
            return order_qty
        
        return 0
```

**Configuration**:
```python
WAREHOUSE_POLICY_CONFIG = {
    'forecast_method': 'moving_average',
    'forecast_window': 14,     # Longer window for stability
    'z_score': 1.96,           # 97.5% service level
    'lead_time': 7,            # Supplier lead time
    'days_of_cover': 30,       # Longer coverage
    'min_order_quantity': 100  # Economies of scale
}
```

---

### 4. Policy Logger
**File**: [policy_logger.py](file:///d:/Codex%20Hackathon/Codex_Hackathon/inventory-hmarl/baseline_policies/policy_logger.py)

**Purpose**: Log complete state-action-outcome trajectories

#### Log Structure

**Per Timestep**:
```python
{
    'day': 0,
    'stores': {
        'store_1': {
            'state': {
                'inventory': 200.0,
                'forecast': 50.0,
                'reorder_point': 120.5,
                'safety_stock': 20.3,
                'demand_std': 10.0
            },
            'action': {
                'order_quantity': 100.0,
                'order_placed': True
            }
        }
    },
    'warehouses': {...},
    'outcomes': {
        'actual_demand': 55.0,
        'sales': 55.0,
        'lost_sales': 0.0
    },
    'metrics': {
        'total_service_level': 0.95,
        'total_cost': 150.0
    }
}
```

#### Output Formats

1. **JSON**: Structured logs
```python
logger.save_json('baseline_logs_20260128.json')
```

2. **CSV**: Time series
```python
logger.save_csv('baseline_logs_20260128.csv')
```

CSV columns include:
- `day`
- `total_service_level`
- `total_cost`
- `store_1_inventory`, `store_1_forecast`, `store_1_order_qty`
- `warehouse_1_inventory`, `warehouse_1_rop`
- etc.

---

### 5. Baseline Runner
**File**: [baseline_runner.py](file:///d:/Codex%20Hackathon/Codex_Hackathon/inventory-hmarl/baseline_policies/baseline_runner.py)

**Purpose**: Integrate baseline policies with digital twin

#### CLI Interface

```bash
# Basic usage
python baseline_policies/baseline_runner.py

# Custom configuration
python baseline_policies/baseline_runner.py --days 180 --z-score 2.0

# Different scenarios
python baseline_policies/baseline_runner.py --scenario spike

# Override forecasting
python baseline_policies/baseline_runner.py --forecast-method exp_smoothing
```

#### Integration Loop

```python
# Initialize
env = DigitalTwin(sim_config)
store_policies = {store_id: StorePolicy(...) for store_id in stores}
warehouse_policies = {wh_id: WarehousePolicy(...) for wh_id in warehouses}

# Run simulation
for day in range(90):
    # Get current state
    env_state = env.get_state()
    
    # Store decisions
    store_orders = {}
    for store_id, policy in store_policies.items():
        inventory = env_state['stores'][store_id]['inventory']
        orders = policy.decide(inventory)
        store_orders[store_id] = orders
    
    # Warehouse decisions
    warehouse_orders = {}
    for wh_id, policy in warehouse_policies.items():
        inventory = env_state['warehouses'][wh_id]['inventory']
        orders = policy.decide(inventory, in_transit, backorders)
        warehouse_orders[wh_id] = orders
    
    # Step environment
    actions = {
        'store_orders': store_orders,
        'warehouse_orders': warehouse_orders
    }
    next_state, metrics, done, info = env.step(actions)
    
    # Update forecasters
    for store_id, policy in store_policies.items():
        actual_demand = env_state['stores'][store_id]['daily_demand']
        policy.update_demand(actual_demand)
    
    # Log
    logger.log_timestep(day, store_states, store_actions, ...)
```

---

## Implementation Details

### ROP Formula Deep Dive

#### Why Reorder Point?

**Problem**: When should we reorder?

**Answer**: When inventory drops to a level where stockout risk during lead time exceeds our tolerance.

#### Step-by-Step Example

**Given**:
- Forecast: 50 units/day
- Demand std: 10 units
- Lead time: 2 days
- Z-score: 1.65 (95% service)

**Calculate**:

1. **Expected Demand during Lead Time**:
   ```
   Expected = 50 × 2 = 100 units
   ```

2. **Safety Stock**:
   ```
   Safety Stock = 1.65 × 10 × √2
                = 1.65 × 10 × 1.414
                = 23.3 units
   ```

3. **Reorder Point**:
   ```
   ROP = 100 + 23.3 = 123.3 units
   ```

**Interpretation**: Order when inventory ≤ 123 units

#### Order Quantity Example

**Given**:
- Current inventory: 100 units
- ROP: 123 units
- Forecast: 50 units/day
- Days of cover: 14 days

**Calculate**:

1. **Check if order needed**:
   ```
   100 ≤ 123? YES → Place order
   ```

2. **Target inventory**:
   ```
   Target = 14 × 50 = 700 units
   ```

3. **Order quantity**:
   ```
   Order = 700 - 100 = 600 units
   ```

---

### Safety Stock Formula Explained

**Formula**:
```
Safety Stock = Z × σ × √(Lead Time)
```

**Components**:

1. **Z-score**: Controls service level
   - Higher Z → More safety stock → Higher service, higher cost

2. **σ (sigma)**: Demand standard deviation
   - Higher variability → More safety stock needed

3. **√(Lead Time)**: Uncertainty grows with square root of time
   - Longer lead time → More safety stock needed
   - Not linear because averaging effect

**Example**:
- Z = 1.65 (95% service)
- σ = 10 units/day
- LT = 4 days

```
Safety Stock = 1.65 × 10 × √4
             = 1.65 × 10 × 2
             = 33 units
```

---

## Usage Examples

### Example 1: Basic Baseline Simulation

```python
from baseline_policies import StorePolicy, WarehousePolicy
import baseline_policies.policy_config as config

# Create store policy
store_policy = StorePolicy('store_1', 'SKU_001', config.STORE_POLICY_CONFIG)

# Feed historical demand
for demand in [50, 55, 60, 50, 45]:
    store_policy.update_demand(demand)

# Make decision
inventory = 100
order_qty = store_policy.decide(inventory)

# Get state for analysis
state = store_policy.get_state()
print(f"Inventory: {inventory}")
print(f"Forecast: {state['forecast']:.2f}")
print(f"ROP: {state['reorder_point']:.2f}")
print(f"Safety Stock: {state['safety_stock']:.2f}")
print(f"Order: {order_qty:.2f}")
```

---

### Example 2: Compare Forecasting Methods

```python
from baseline_policies.forecasting import (
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    SeasonalAverageForecaster
)

demands = [50, 55, 60, 50, 45, 40, 50, 55, 60]

# Moving Average
ma = MovingAverageForecaster(window=3)
for d in demands:
    ma.update(d)
print(f"MA Forecast: {ma.predict():.2f}")        # 55.0

# Exponential Smoothing
es = ExponentialSmoothingForecaster(alpha=0.3)
for d in demands:
    es.update(d)
print(f"ES Forecast: {es.predict():.2f}")        # ~54.2

# Seasonal (weekly pattern)
seasonal = SeasonalAverageForecaster(period=7)
for d in demands:
    seasonal.update(d)
print(f"Seasonal Forecast: {seasonal.predict():.2f}")
```

---

### Example 3: Run Full Baseline

```bash
cd inventory-hmarl

# Run 90-day baseline simulation
python baseline_policies/baseline_runner.py --days 90 --scenario normal

# Run with different Z-score (higher service level)
python baseline_policies/baseline_runner.py --days 90 --z-score 2.0

# Run with exponential smoothing
python baseline_policies/baseline_runner.py --forecast-method exp_smoothing
```

---

## Testing & Validation

### Test 1: Forecasting Methods
**File**: [test_baseline.py](file:///d:/Codex%20Hackathon/Codex_Hackathon/inventory-hmarl/tests/test_baseline.py)

✅ **Result**: PASSED

**Verified**:
- Moving average calculation: `mean([50, 60, 55]) = 55.0`
- Standard deviation tracking
- Forecast updates with new data
- ROP calculation correctness

---

### Test 2: Store Policy
✅ **Result**: PASSED

**Scenario**:
- Inventory: 100 units
- Forecast: 51.67 units/day
- ROP calculated: 120.54 units
- Order placed: YES

**Verified**:
- ROP formula correct
- Safety stock calculated
- Order quantity determined
- State logging works

---

### Test 3: Integration Test
**File**: [test_baseline_integration.py](file:///d:/Codex%20Hackathon/Codex_Hackathon/inventory-hmarl/tests/test_baseline_integration.py)

✅ **Result**: PASSED

**20-Day Simulation**:
```
Day 5: Service Level = 100.00%
Day 10: Service Level = 98.50%
Day 15: Service Level = 97.80%
Day 20: Service Level = 96.90%
```

**Verified**:
- Policies integrate with digital twin
- Orders placed when inventory ≤ ROP
- Service level stabilizes after warm-up
- System handles demand variability

---

## Performance Characteristics

### Expected Behavior

**Normal Operations**:
- Service Level: 90-98% (depends on Z-score)
- Order Frequency: Every 2-4 days per store
- Inventory Oscillation: ±20% around target

**Demand Spike Scenario**:
- Service Level: Drops 5-10% during spike
- Recovery Time: 3-5 days after spike ends
- Cost Impact: +30-50% during spike period

**Cold Start** (no history):
- Days 1-7: Uses initial forecast, may overstock
- Days 8-14: Forecaster learns, stabilizes
- Days 15+: Normal operation

---

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `baseline_policies/policy_config.py` | 105 | Configuration |
| `baseline_policies/forecasting.py` | 270 | Forecasting methods |
| `baseline_policies/store_policy.py` | 230 | Store ROP policy |
| `baseline_policies/warehouse_policy.py` | 270 | Warehouse ROP policy |
| `baseline_policies/policy_logger.py` | 210 | Logging |
| `baseline_policies/baseline_runner.py` | 330 | Integration |
| `tests/test_baseline.py` | 85 | Unit tests |
| `tests/test_baseline_integration.py` | 130 | Integration test |
| **Total** | **~1,630** | **Complete baseline system** |

---

## Summary

The Baseline Policy Layer successfully provides:
- ✅ **Classical ROP formulas** (textbook inventory management)
- ✅ **Three forecasting methods** (MA, ES, Seasonal)
- ✅ **Deterministic decisions** (reproducible, explainable)
- ✅ **Comprehensive logging** (state-action-outcome)
- ✅ **Full integration** with digital twin
- ✅ **Benchmark ready** for RL comparison

The system generates realistic operational behavior using proven inventory control methods, providing both a performance baseline and training data for future learning-based approaches.
