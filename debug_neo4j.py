import os
from dotenv import load_dotenv
from graph_db.neo4j_client import Neo4jClient

# Load environment variables
load_dotenv()

def debug_neo4j_data():
    """Debug Neo4j data structure and relationships"""
    
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USER = os.getenv('NEO4J_USER')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    
    client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    print("🔍 Debugging Neo4j Data Structure")
    print("=" * 50)
    
    # Check all node labels
    print("📋 All Node Labels:")
    labels = client.query("CALL db.labels() YIELD label RETURN label")
    for record in labels:
        print(f"  - {record['label']}")
    
    # Check all relationship types
    print("\n🔗 All Relationship Types:")
    rel_types = client.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    for record in rel_types:
        print(f"  - {record['relationshipType']}")
    
    # Sample nodes from each label
    print("\n📊 Sample Nodes by Label:")
    
    # Items
    items = client.query("MATCH (i:Item) RETURN i.name as name, i.item_id as id LIMIT 3")
    print("Items:")
    for record in items:
        print(f"  - {record['name']} (ID: {record['id']})")
    
    # Suppliers
    suppliers = client.query("MATCH (s:Supplier) RETURN s.name as name, s.supplier_id as id LIMIT 3")
    print("\nSuppliers:")
    for record in suppliers:
        print(f"  - {record['name']} (ID: {record['id']})")
    
    # Carriers
    carriers = client.query("MATCH (c:Carrier) RETURN c.name as name LIMIT 3")
    print("\nCarriers:")
    for record in carriers:
        print(f"  - {record['name']}")
    
    # Customers
    customers = client.query("MATCH (c:Customer) RETURN c.customer_id as id LIMIT 3")
    print("\nCustomers:")
    for record in customers:
        print(f"  - Customer {record['id']}")
    
    # Check for any relationships
    print("\n🔗 Checking for Relationships:")
    
    # All relationships
    all_rels = client.query("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
    print("All relationships:")
    for record in all_rels:
        print(f"  - {record['type']}: {record['count']}")
    
    # Sample relationships with details
    print("\n📋 Sample Relationships with Details:")
    
    # Items to Suppliers
    item_supplier_rels = client.query("""
        MATCH (i:Item)-[r:SUPPLIED_BY]->(s:Supplier)
        RETURN i.name as item, s.name as supplier
        LIMIT 3
    """)
    print("Item -> Supplier relationships:")
    for record in item_supplier_rels:
        print(f"  - {record['item']} -> {record['supplier']}")
    
    # Items to Carriers
    item_carrier_rels = client.query("""
        MATCH (i:Item)-[r:SHIPPED_VIA]->(c:Carrier)
        RETURN i.name as item, c.name as carrier, r.delayed as delayed
        LIMIT 3
    """)
    print("\nItem -> Carrier relationships:")
    for record in item_carrier_rels:
        print(f"  - {record['item']} -> {record['carrier']} (Delayed: {record['delayed']})")
    
    # Items to Customers
    item_customer_rels = client.query("""
        MATCH (i:Item)-[r:RETURNED_BY]->(c:Customer)
        RETURN i.name as item, c.customer_id as customer, r.reason as reason
        LIMIT 3
    """)
    print("\nItem -> Customer relationships:")
    for record in item_customer_rels:
        print(f"  - {record['item']} -> Customer {record['customer']} (Reason: {record['reason']})")
    
    client.close()

if __name__ == "__main__":
    debug_neo4j_data() 