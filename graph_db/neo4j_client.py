from neo4j import GraphDatabase
import pandas as pd
import os

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_graph_from_csvs(self, inventory_path, suppliers_path, logistics_path, returns_path):
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            # Inventory
            inventory = pd.read_csv(inventory_path)
            for _, row in inventory.iterrows():
                session.run(
                    "CREATE (i:Item {item_id: $item_id, name: $name, stock: $stock, warehouse_id: $warehouse_id, predicted_demand: $demand})",
                    {"item_id": str(row['item_id']), "name": row['item_name'], "stock": int(row['stock']), "warehouse_id": row['warehouse_id'], "demand": int(row['predicted_demand_next_week'])}
                )
            # Suppliers
            suppliers = pd.read_csv(suppliers_path)
            for _, row in suppliers.iterrows():
                session.run(
                    "MERGE (s:Supplier {supplier_id: $supplier_id, name: $name, on_time_rate: $on_time, return_rate: $return_rate})",
                    {"supplier_id": row['supplier_id'], "name": row['supplier_name'], "on_time": float(row['on_time_rate']), "return_rate": float(row['return_rate'])}
                )
                session.run(
                    "MATCH (i:Item {item_id: $item_id}), (s:Supplier {supplier_id: $supplier_id}) CREATE (i)-[:SUPPLIED_BY]->(s)",
                    {"item_id": str(row['item_id']), "supplier_id": row['supplier_id']}
                )
            # Logistics
            logistics = pd.read_csv(logistics_path)
            for _, row in logistics.iterrows():
                session.run(
                    "MERGE (c:Carrier {name: $carrier})",
                    {"carrier": row['carrier']}
                )
                session.run(
                    "MATCH (i:Item {item_id: $item_id}), (c:Carrier {name: $carrier}) CREATE (i)-[:SHIPPED_VIA {delayed: $delayed, reason: $reason}]->(c)",
                    {"item_id": str(row['item_id']), "carrier": row['carrier'], "delayed": row['delayed'], "reason": row['delay_reason']}
                )
            # Returns
            returns = pd.read_csv(returns_path)
            for _, row in returns.iterrows():
                session.run(
                    "MERGE (c:Customer {customer_id: $customer_id})",
                    {"customer_id": row['customer_id']}
                )
                session.run(
                    "MATCH (i:Item {item_id: $item_id}), (c:Customer {customer_id: $customer_id}) CREATE (i)-[:RETURNED_BY {reason: $reason, date: $date}]->(c)",
                    {"item_id": str(row['item_id']), "customer_id": row['customer_id'], "reason": row['return_reason'], "date": row['date']}
                )

    def query(self, cypher_query, params=None):
        with self.driver.session() as session:
            result = session.run(cypher_query, params or {})
            return [record.data() for record in result] 