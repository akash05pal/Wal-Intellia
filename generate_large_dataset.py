import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_returns_data(n_rows=1000):
    """Generate returns data with realistic patterns"""
    return_reasons = ['Defective', 'Not as described', 'Wrong size', 'Wrong color', 'Damaged in transit', 'Late delivery', 'Quality issues']
    items = list(range(1001, 1016))  # 15 items
    
    data = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(n_rows):
        return_id = f"R{i+1:04d}"
        item_id = random.choice(items)
        customer_id = f"C{random.randint(100, 999)}"
        return_reason = random.choice(return_reasons)
        date = start_date + timedelta(days=random.randint(0, 365))
        
        data.append([return_id, item_id, customer_id, return_reason, date.strftime('%Y-%m-%d')])
    
    return pd.DataFrame(data, columns=['return_id', 'item_id', 'customer_id', 'return_reason', 'date'])

def generate_logistics_data(n_rows=1000):
    """Generate logistics data with realistic patterns"""
    carriers = ['CarrierX', 'CarrierY', 'CarrierZ', 'CarrierA', 'CarrierB']
    delay_reasons = ['Weather', 'Customs', 'Traffic', 'Mechanical', 'Driver shortage', 'Warehouse delay', '']
    items = list(range(1001, 1016))
    
    data = []
    
    for i in range(n_rows):
        shipment_id = f"L{i+1:04d}"
        item_id = random.choice(items)
        carrier = random.choice(carriers)
        delayed = random.choice(['yes', 'no'])
        delay_reason = random.choice(delay_reasons) if delayed == 'yes' else ''
        
        data.append([shipment_id, item_id, carrier, delayed, delay_reason])
    
    return pd.DataFrame(data, columns=['shipment_id', 'item_id', 'carrier', 'delayed', 'delay_reason'])

def generate_inventory_data(n_rows=1000):
    """Generate inventory data with realistic patterns"""
    item_names = [
        'Widget A', 'Widget B', 'Gadget C', 'Tool D', 'Device E', 'Component F', 'Material G', 'Part H',
        'Unit I', 'Module J', 'Assembly K', 'System L', 'Kit M', 'Set N', 'Box O', 'Panel P', 'Circuit Q',
        'Sensor R', 'Motor S', 'Battery T', 'Display U', 'Cable V', 'Switch W', 'Relay X', 'Fuse Y'
    ]
    warehouses = ['W1', 'W2', 'W3', 'W4', 'W5']
    
    data = []
    
    for i in range(n_rows):
        item_id = 1001 + i
        item_name = item_names[i % len(item_names)]
        stock = random.randint(0, 500)
        warehouse_id = random.choice(warehouses)
        predicted_demand = random.randint(50, 300)
        
        data.append([item_id, item_name, stock, warehouse_id, predicted_demand])
    
    return pd.DataFrame(data, columns=['item_id', 'item_name', 'stock', 'warehouse_id', 'predicted_demand_next_week'])

def generate_suppliers_data(n_rows=1000):
    """Generate suppliers data with realistic patterns"""
    supplier_names = [
        'Supplier Alpha', 'Supplier Beta', 'Supplier Gamma', 'Supplier Delta', 'Supplier Epsilon',
        'Supplier Zeta', 'Supplier Eta', 'Supplier Theta', 'Supplier Iota', 'Supplier Kappa',
        'Supplier Lambda', 'Supplier Mu', 'Supplier Nu', 'Supplier Xi', 'Supplier Omicron',
        'TechCorp Inc', 'Global Parts Ltd', 'Quality Components', 'Reliable Supply Co', 'Premium Materials',
        'FastTrack Logistics', 'Elite Manufacturing', 'Core Systems', 'Advanced Solutions', 'Prime Vendors'
    ]
    
    data = []
    
    for i in range(n_rows):
        supplier_id = f"S{i+1:03d}"
        supplier_name = supplier_names[i % len(supplier_names)]
        item_id = 1001 + (i % 25)  # Cycle through items
        # Generate realistic rates with some variation
        on_time_rate = round(random.uniform(0.65, 0.98), 2)
        return_rate = round(random.uniform(0.05, 0.45), 2)
        
        data.append([supplier_id, supplier_name, item_id, on_time_rate, return_rate])
    
    return pd.DataFrame(data, columns=['supplier_id', 'supplier_name', 'item_id', 'on_time_rate', 'return_rate'])

def main():
    """Generate all datasets"""
    print("Generating large datasets...")
    
    # Generate data
    returns_df = generate_returns_data(1000)
    logistics_df = generate_logistics_data(1000)
    inventory_df = generate_inventory_data(1000)
    suppliers_df = generate_suppliers_data(1000)
    
    # Save to CSV files
    returns_df.to_csv('data/returns.csv', index=False)
    logistics_df.to_csv('data/logistics.csv', index=False)
    inventory_df.to_csv('data/inventory.csv', index=False)
    suppliers_df.to_csv('data/suppliers.csv', index=False)
    
    print(f"Generated datasets:")
    print(f"- Returns: {len(returns_df)} rows")
    print(f"- Logistics: {len(logistics_df)} rows")
    print(f"- Inventory: {len(inventory_df)} rows")
    print(f"- Suppliers: {len(suppliers_df)} rows")
    print("All files saved to data/ directory")

if __name__ == "__main__":
    main() 