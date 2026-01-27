# Digital Twin Retail Supply Chain Simulation

A modular, discrete-time digital twin simulation for a retail supply chain, designed for testing inventory management policies and hierarchical multi-agent reinforcement learning (HMARL) frameworks.

## Overview

This digital twin simulates a simplified retail supply chain with:
- **3 Retail Stores** with varying demand patterns
- **1 Central Warehouse** managing distribution
- **1 Upstream Supplier** with configurable lead times
- **Stochastic + Seasonal Demand** with realistic patterns
- **Daily Discrete-Time Operations** over 90-180 day horizon

## Key Features

✅ **Modular Architecture** - Clean separation of entities, demand, metrics, and environment  
✅ **RL-Compatible Interface** - `reset()`, `step()`, `get_state()`, `get_metrics()`  
✅ **Realistic Demand** - Stochastic noise + seasonal patterns + scenario-driven spikes  
✅ **Multiple Scenarios** - Test normal operations, demand spikes, seasonality, supplier delays  
✅ **Comprehensive Metrics** - Service level, fill rate, holding costs, stockout penalties  
✅ **Hackathon-Ready** - No UI, no ML dependencies, pure simulation logic  

## Project Structure

```
inventory-hmarl/
├── config/                 # Simulation configuration
│   └── simulation_config.py
├── entities/              # Supply chain entities
│   ├── store.py          # Retail store
│   ├── warehouse.py      # Distribution center
│   └── supplier.py       # Upstream supplier
├── demand/               # Demand generation
│   └── demand_generator.py
├── env/                  # Digital twin environment
│   └── digital_twin.py   # Main simulation loop
├── evaluation/           # Metrics tracking
│   └── metrics.py
├── scenarios/            # Scenario definitions
│   └── scenarios.py
├── simulation/           # Simulation runner
│   └── run_simulation.py
└── tests/                # Test scripts
    └── test_basic.py
```

## Quick Start

### Basic Usage

```bash
# Run 90-day simulation with normal operations
python simulation/run_simulation.py

# Run 60-day simulation with demand spike scenario
python simulation/run_simulation.py --days 60 --scenario spike

# Run with different random seed
python simulation/run_simulation.py --days 180 --scenario normal --seed 123

# Verbose output
python simulation/run_simulation.py --days 30 --scenario normal --verbose
```

### Programmatic Usage

```python
from env.digital_twin import DigitalTwin
import config.simulation_config as config

# Create digital twin
sim_config = {
    'SIMULATION_DAYS': 90,
    'WARMUP_DAYS': 7,
    'RANDOM_SEED': 42,
    'SKU_CONFIG': config.SKU_CONFIG,
    'STORE_CONFIG': config.STORE_CONFIG,
    'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
    'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
}

env = DigitalTwin(sim_config)

# Reset environment
state = env.reset()

# Step through simulation
for day in range(90):
    state, metrics, done, info = env.step()
    
    if done:
        break

# Get final metrics
final_metrics = env.get_metrics()
print(f"Service Level: {final_metrics['avg_service_level']:.2%}")
print(f"Total Cost: ${final_metrics['total_cost']:,.2f}")
```

## Available Scenarios

| Scenario | Description |
|----------|-------------|
| `normal` | Baseline stochastic and seasonal demand |
| `spike` | Sudden demand spikes on days 30-32 (2.5x multiplier) |
| `strong_seasonality` | Amplified seasonal patterns (50% amplitude) |
| `supplier_delay` | Increased supplier lead times (+7 days) |
| `high_variability` | Doubled demand variability (2x std dev) |

## How It Works

### Daily Simulation Loop

Each simulation day follows this sequence:

1. **Generate Demand** - Stochastic + seasonal customer demand at stores
2. **Fulfill Demand** - Stores serve customers from inventory (track sales & stockouts)
3. **Store Replenishment** - Stores place orders to warehouse using (s,S) policy
4. **Warehouse Fulfillment** - Warehouse ships to stores (or backlogs)
5. **Warehouse Replenishment** - Warehouse orders from supplier
6. **Supplier Delivery** - Supplier fulfills orders (after lead time)
7. **Metrics Update** - Track costs, service levels, inventory

### Demand Generation

Demand is generated using:
```
demand(day) = base_demand 
              × demand_multiplier          # Store-specific
              × (1 + amplitude × sin(...)) # Seasonal
              + noise(std_dev)             # Stochastic
```

### Inventory Policies

Entities use **(s, S) policies**:
- If inventory ≤ reorder_point (s), order up to order_up_to level (S)
- Can be overridden by RL agent actions in `step(actions)`

## Metrics Tracked

### Service Metrics
- **Service Level**: Fulfilled demand / Total demand
- **Fill Rate**: Orders met from stock / Total orders
- **Lost Sales**: Stockout units

### Cost Metrics
- **Holding Cost**: Inventory carrying costs
- **Stockout Penalty**: Lost sales penalties
- **Ordering Cost**: Fixed cost per order
- **Total Cost**: Sum of all costs

### Inventory Metrics
- Average, min, max inventory levels
- Inventory standard deviation

## Future RL Integration

The digital twin is designed for HMARL:
- `step(actions)` accepts agent decisions
- State representation is easily vectorized
- Metrics can be used as reward signals
- Supports hierarchical control (store agents + warehouse agents + coordinator)

## Configuration

Edit `config/simulation_config.py` to customize:
- Simulation horizon
- Number of entities
- SKU parameters (base demand, costs, seasonality)
- Store/warehouse capacities and policies
- Supplier lead times

## Testing

```bash
# Basic functionality test
python tests/test_basic.py

# Test different scenarios
python simulation/run_simulation.py --days 30 --scenario spike
python simulation/run_simulation.py --days 90 --scenario strong_seasonality
```

## Dependencies

- **Python 3.7+**
- **numpy** - Random number generation and array operations
- No heavy ML frameworks required

## License

MIT License - Hackathon Project
