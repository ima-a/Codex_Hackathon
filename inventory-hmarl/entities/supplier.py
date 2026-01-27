"""
supplier.py

Supplier entity for the digital twin simulation.

A supplier:
- Has infinite/large supply (simplified for hackathon)
- Fulfills warehouse orders with configurable lead time
- Optionally models fulfillment reliability
"""

from typing import Dict
from collections import deque


class Supplier:
    """
    Represents an upstream supplier.
    
    For hackathon simplicity:
    - Infinite supply capacity
    - Deterministic lead time
    - High reliability (98%+)
    
    Attributes:
        supplier_id: Unique identifier
        name: Human-readable name
        lead_time: Days to fulfill orders
        reliability: Probability of successful fulfillment (0-1)
    """
    
    def __init__(
        self,
        supplier_id: str,
        name: str,
        lead_time: int = 7,
        reliability: float = 1.0
    ):
        """
        Initialize supplier.
        
        Args:
            supplier_id: Unique identifier
            name: Supplier name
            lead_time: Delivery lead time in days
            reliability: Order fulfillment probability (0-1)
        """
        self.supplier_id = supplier_id
        self.name = name
        self.lead_time = lead_time
        self.reliability = reliability
        
        # Pending orders: queue of (delivery_day, warehouse_id, order_dict)
        self.pending_orders = deque()
        
        # History
        self.total_orders_received = 0
        self.total_orders_fulfilled = 0
        self.total_units_supplied = 0
    
    def receive_order(
        self, 
        warehouse_id: str, 
        order: Dict[str, float], 
        day: int
    ) -> int:
        """
        Receive order from warehouse.
        
        Args:
            warehouse_id: Warehouse placing the order
            order: Dict mapping SKU -> quantity
            day: Current simulation day
            
        Returns:
            Expected delivery day
        """
        self.total_orders_received += 1
        
        delivery_day = day + self.lead_time
        
        # For simplicity, always accept the order
        # In future: could model capacity constraints, order rejections
        self.pending_orders.append({
            'delivery_day': delivery_day,
            'warehouse_id': warehouse_id,
            'order': order.copy(),
            'order_day': day
        })
        
        return delivery_day
    
    def get_shipments_for_day(self, day: int) -> Dict[str, Dict[str, float]]:
        """
        Get all shipments scheduled for delivery on this day.
        
        Args:
            day: Current simulation day
            
        Returns:
            Dict mapping warehouse_id -> shipment (SKU -> quantity)
        """
        shipments = {}
        
        # Process orders due for delivery
        remaining_orders = deque()
        
        for order_info in self.pending_orders:
            if order_info['delivery_day'] <= day:
                # Deliver this order
                warehouse_id = order_info['warehouse_id']
                order = order_info['order']
                
                # Apply reliability (simplified: all-or-nothing)
                import random
                if random.random() <= self.reliability:
                    # Fulfill order
                    if warehouse_id not in shipments:
                        shipments[warehouse_id] = {}
                    
                    for sku, qty in order.items():
                        shipments[warehouse_id][sku] = shipments[warehouse_id].get(sku, 0.0) + qty
                        self.total_units_supplied += qty
                    
                    self.total_orders_fulfilled += 1
                # else: order fails (dropped)
            else:
                # Keep for future delivery
                remaining_orders.append(order_info)
        
        self.pending_orders = remaining_orders
        
        return shipments
    
    def get_state(self) -> dict:
        """Get current supplier state."""
        return {
            'supplier_id': self.supplier_id,
            'name': self.name,
            'lead_time': self.lead_time,
            'reliability': self.reliability,
            'pending_orders': len(self.pending_orders),
            'total_orders_received': self.total_orders_received,
            'total_orders_fulfilled': self.total_orders_fulfilled
        }
    
    def get_metrics(self) -> dict:
        """Get cumulative metrics."""
        fulfillment_rate = (
            self.total_orders_fulfilled / self.total_orders_received
            if self.total_orders_received > 0
            else 0.0
        )
        
        return {
            'supplier_id': self.supplier_id,
            'total_orders_received': self.total_orders_received,
            'total_orders_fulfilled': self.total_orders_fulfilled,
            'total_units_supplied': self.total_units_supplied,
            'fulfillment_rate': fulfillment_rate
        }
    
    def reset(self):
        """Reset supplier to initial state."""
        self.pending_orders = deque()
        self.total_orders_received = 0
        self.total_orders_fulfilled = 0
        self.total_units_supplied = 0
