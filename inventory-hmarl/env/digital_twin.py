"""
digital_twin.py

Main digital twin simulation environment.

Orchestrates the entire supply chain:
- Multiple stores
- Warehouse(s)
- Supplier(s)
- Demand generation
- Inventory flows
- Metrics tracking

Provides RL-compatible interface:
- reset() - initialize simulation
- step(actions) - advance one day
- get_state() - return current state
- get_metrics() - return KPIs
"""

import numpy as np
from typing import Dict, Optional, Tuple, List

from entities.store import Store
from entities.warehouse import Warehouse
from entities.supplier import Supplier
from demand.demand_generator import MultiSKUDemandGenerator
from evaluation.metrics import MetricsTracker


class DigitalTwin:
    """
    Digital twin environment for retail supply chain simulation.
    
    Daily simulation loop:
    1. Generate customer demand at stores
    2. Stores fulfill demand from inventory
    3. Stores place replenishment orders to warehouse
    4. Warehouse fulfills store orders
    5. Warehouse places orders to supplier
    6. Supplier delivers orders (after lead time)
    7. Update inventories and metrics
    
    Attributes:
        stores: List of Store entities
        warehouses: List of Warehouse entities
        suppliers: List of Supplier entities
        demand_generators: Dict mapping store_id -> DemandGenerator
        metrics_tracker: MetricsTracker instance
        current_day: Current simulation day
    """
    
    def __init__(self, config: dict):
        """
        Initialize digital twin from configuration.
        
        Args:
            config: Configuration dict with all parameters
        """
        self.config = config
        self.current_day = 0
        self.simulation_days = config.get('SIMULATION_DAYS', 90)
        
        # Extract SKU configuration
        self.sku_config = config.get('SKU_CONFIG', {})
        self.sku_ids = list(self.sku_config.keys())
        
        # Initialize entities
        self.stores = self._create_stores(config)
        self.warehouses = self._create_warehouses(config)
        self.suppliers = self._create_suppliers(config)
        
        # Initialize demand generators (one per store)
        self.demand_generators = self._create_demand_generators(config)
        
        # Initialize metrics tracker
        warmup_days = config.get('WARMUP_DAYS', 7)
        self.metrics_tracker = MetricsTracker(warmup_days=warmup_days)
        
        # Random seed
        seed = config.get('RANDOM_SEED', 42)
        np.random.seed(seed)
    
    def _create_stores(self, config: dict) -> List[Store]:
        """Create store entities from configuration."""
        stores = []
        store_config = config.get('STORE_CONFIG', {})
        
        for store_id, store_params in store_config.items():
            # Initialize inventory for each SKU
            initial_inv = {}
            for sku in self.sku_ids:
                initial_inv[sku] = store_params.get('initial_inventory', 500)
            
            store = Store(
                store_id=store_id,
                name=store_params.get('name', store_id),
                initial_inventory=initial_inv,
                config=store_params
            )
            stores.append(store)
        
        return stores
    
    def _create_warehouses(self, config: dict) -> List[Warehouse]:
        """Create warehouse entities from configuration."""
        warehouses = []
        warehouse_config = config.get('WAREHOUSE_CONFIG', {})
        
        for wh_id, wh_params in warehouse_config.items():
            # Initialize inventory for each SKU
            initial_inv = {}
            for sku in self.sku_ids:
                initial_inv[sku] = wh_params.get('initial_inventory', 3000)
            
            warehouse = Warehouse(
                warehouse_id=wh_id,
                name=wh_params.get('name', wh_id),
                initial_inventory=initial_inv,
                config=wh_params
            )
            warehouses.append(warehouse)
        
        return warehouses
    
    def _create_suppliers(self, config: dict) -> List[Supplier]:
        """Create supplier entities from configuration."""
        suppliers = []
        supplier_config = config.get('SUPPLIER_CONFIG', {})
        
        for sup_id, sup_params in supplier_config.items():
            supplier = Supplier(
                supplier_id=sup_id,
                name=sup_params.get('name', sup_id),
                lead_time=sup_params.get('lead_time', 7),
                reliability=sup_params.get('reliability', 1.0)
            )
            suppliers.append(supplier)
        
        return suppliers
    
    def _create_demand_generators(self, config: dict) -> Dict[str, MultiSKUDemandGenerator]:
        """Create demand generators for each store."""
        generators = {}
        store_config = config.get('STORE_CONFIG', {})
        seed = config.get('RANDOM_SEED', 42)
        
        for i, (store_id, store_params) in enumerate(store_config.items()):
            demand_multiplier = store_params.get('demand_multiplier', 1.0)
            
            # Create generator for this store
            generators[store_id] = MultiSKUDemandGenerator(
                sku_configs=self.sku_config,
                demand_multiplier=demand_multiplier,
                seed=seed + i * 1000  # Different seed per store
            )
        
        return generators
    
    def reset(self) -> dict:
        """
        Reset simulation to initial state.
        
        Returns:
            Initial state dict
        """
        self.current_day = 0
        
        # Reset all entities
        for store in self.stores:
            initial_inv = {sku: store.config.get('initial_inventory', 500) for sku in self.sku_ids}
            store.reset(initial_inv)
        
        for warehouse in self.warehouses:
            initial_inv = {sku: warehouse.config.get('initial_inventory', 3000) for sku in self.sku_ids}
            warehouse.reset(initial_inv)
        
        for supplier in self.suppliers:
            supplier.reset()
        
        # Reset demand generators
        seed = self.config.get('RANDOM_SEED', 42)
        for i, (store_id, generator) in enumerate(self.demand_generators.items()):
            generator.reset(seed + i * 1000)
        
        # Reset metrics
        self.metrics_tracker.reset()
        
        return self.get_state()
    
    def step(self, actions: Optional[dict] = None) -> Tuple[dict, dict, bool, dict]:
        """
        Advance simulation by one day.
        
        Args:
            actions: Optional dict of entity decisions (for RL agents)
                     If None, uses default (s,S) policies
        
        Returns:
            Tuple of (state, metrics, done, info)
        """
        # STEP 1: Generate demand at stores
        for store in self.stores:
            demand = self.demand_generators[store.store_id].generate(self.current_day)
            store.receive_demand(demand, self.current_day)
        
        # STEP 2: Stores fulfill demand from inventory
        for store in self.stores:
            store.fulfill_demand(self.sku_config)
            store.calculate_holding_cost(self.sku_config)
        
        # STEP 3: Stores place replenishment orders to warehouse
        # (Using default (s,S) policy or actions from RL agent)
        for store in self.stores:
            if actions and store.store_id in actions.get('store_orders', {}):
                # Use action from RL agent
                orders = actions['store_orders'][store.store_id]
            else:
                # Use default (s,S) policy
                orders = store.check_replenishment()
            
            store.place_order(orders, self.current_day)
            
            # Send orders to warehouse (assume single warehouse for now)
            if orders and self.warehouses:
                self.warehouses[0].receive_store_order(store.store_id, orders, self.current_day)
        
        # STEP 4: Warehouse fulfills store orders
        for warehouse in self.warehouses:
            shipments = warehouse.fulfill_store_orders(self.sku_config)
            
            # Deliver shipments to stores (with lead time)
            # For simplicity, assume immediate delivery or schedule for future
            for store_id, shipment in shipments.items():
                # Find corresponding store
                for store in self.stores:
                    if store.store_id == store_id:
                        store.receive_shipment(shipment)
                        break
            
            warehouse.calculate_holding_cost(self.sku_config)
        
        # STEP 5: Warehouse places orders to supplier
        ordering_cost = 0.0
        for warehouse in self.warehouses:
            if actions and 'warehouse_orders' in actions:
                # Use action from RL agent
                orders = actions['warehouse_orders'].get(warehouse.warehouse_id, {})
            else:
                # Use default (s,S) policy
                orders = warehouse.check_replenishment()
            
            if any(qty > 0 for qty in orders.values()):
                # Calculate ordering cost
                ordering_cost += sum(
                    self.sku_config[sku].get('ordering_cost', 0) 
                    for sku, qty in orders.items() if qty > 0
                )
                
                # Place order to supplier
                if self.suppliers:
                    supplier_lead_time = self.suppliers[0].lead_time
                    warehouse.place_supplier_order(orders, self.current_day, supplier_lead_time)
                    self.suppliers[0].receive_order(warehouse.warehouse_id, orders, self.current_day)
        
        # STEP 6: Supplier delivers orders (after lead time)
        for supplier in self.suppliers:
            shipments = supplier.get_shipments_for_day(self.current_day)
            
            # Deliver to warehouses
            for wh_id, shipment in shipments.items():
                for warehouse in self.warehouses:
                    if warehouse.warehouse_id == wh_id:
                        warehouse.receive_supplier_shipment(self.current_day)
                        break
        
        # Also check for in-transit deliveries at warehouses
        for warehouse in self.warehouses:
            warehouse.receive_supplier_shipment(self.current_day)
        
        # STEP 7: Record daily metrics
        for store in self.stores:
            store.record_day()
        
        for warehouse in self.warehouses:
            warehouse.record_day()
        
        # Record aggregate metrics
        store_states = [store.get_state() for store in self.stores]
        warehouse_states = [wh.get_state() for wh in self.warehouses]
        
        self.metrics_tracker.record_day(
            self.current_day,
            store_states,
            warehouse_states,
            ordering_cost
        )
        
        # Advance day
        self.current_day += 1
        
        # Check if simulation is done
        done = self.current_day >= self.simulation_days
        
        # Get current state and metrics
        state = self.get_state()
        metrics = self.get_metrics()
        
        info = {
            'day': self.current_day,
            'done': done
        }
        
        return state, metrics, done, info
    
    def get_state(self) -> dict:
        """
        Get current state of entire system.
        
        Returns:
            State dict with all entity states
        """
        return {
            'day': self.current_day,
            'stores': {store.store_id: store.get_state() for store in self.stores},
            'warehouses': {wh.warehouse_id: wh.get_state() for wh in self.warehouses},
            'suppliers': {sup.supplier_id: sup.get_state() for sup in self.suppliers}
        }
    
    def get_metrics(self) -> dict:
        """
        Get current metrics and KPIs.
        
        Returns:
            Metrics dict from MetricsTracker
        """
        return self.metrics_tracker.get_summary()
    
    def run_simulation(self, num_days: Optional[int] = None, verbose: bool = False) -> dict:
        """
        Run complete simulation.
        
        Args:
            num_days: Number of days to simulate (defaults to config)
            verbose: Print daily progress
            
        Returns:
            Final metrics dict
        """
        if num_days is None:
            num_days = self.simulation_days
        
        self.reset()
        
        if verbose:
            print(f"Starting simulation for {num_days} days...")
        
        for day in range(num_days):
            state, metrics, done, info = self.step()
            
            if verbose and (day + 1) % 10 == 0:
                print(f"Day {day + 1}/{num_days} completed")
            
            if done:
                break
        
        if verbose:
            print(f"\nSimulation completed: {self.current_day} days")
            self.metrics_tracker.print_summary()
        
        return self.get_metrics()
