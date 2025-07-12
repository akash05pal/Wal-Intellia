import os
from dotenv import load_dotenv
from graph_db.neo4j_client import Neo4jClient
import pandas as pd

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test Neo4j connection and verify relationships"""
    
    # Get environment variables
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USER = os.getenv('NEO4J_USER')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
    
    print("🔍 Testing Neo4j Connection...")
    print(f"URI: {NEO4J_URI}")
    print(f"User: {NEO4J_USER}")
    print(f"Password: {'*' * len(NEO4J_PASSWORD) if NEO4J_PASSWORD else 'Not set'}")
    
    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        print("❌ Missing Neo4j environment variables!")
        return False
    
    try:
        # Test connection
        client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        print("✅ Neo4j connection successful!")
        
        # Test basic query
        result = client.query("RETURN 1 as test")
        print(f"✅ Basic query test: {result}")
        
        # Check if data exists
        print("\n📊 Checking existing data...")
        
        # Count nodes by type
        node_counts = client.query("""
            MATCH (n)
            RETURN labels(n)[0] as node_type, count(n) as count
            ORDER BY node_type
        """)
        
        print("Node counts:")
        for record in node_counts:
            print(f"  {record['node_type']}: {record['count']}")
        
        # Count relationships by type
        rel_counts = client.query("""
            MATCH ()-[r]->()
            RETURN type(r) as relationship_type, count(r) as count
            ORDER BY relationship_type
        """)
        
        print("\nRelationship counts:")
        for record in rel_counts:
            print(f"  {record['relationship_type']}: {record['count']}")
        
        # Check sample relationships
        print("\n🔗 Sample Relationships:")
        
        # Items and their suppliers
        items_suppliers = client.query("""
            MATCH (i:Item)-[:SUPPLIED_BY]->(s:Supplier)
            RETURN i.name as item, s.name as supplier
            LIMIT 5
        """)
        
        print("Items -> Suppliers:")
        for record in items_suppliers:
            print(f"  {record['item']} -> {record['supplier']}")
        
        # Items and their carriers
        items_carriers = client.query("""
            MATCH (i:Item)-[:SHIPPED_VIA]->(c:Carrier)
            RETURN i.name as item, c.name as carrier
            LIMIT 5
        """)
        
        print("\nItems -> Carriers:")
        for record in items_carriers:
            print(f"  {record['item']} -> {record['carrier']}")
        
        # Items and customers (returns)
        items_customers = client.query("""
            MATCH (i:Item)-[:RETURNED_BY]->(c:Customer)
            RETURN i.name as item, c.customer_id as customer, c.reason as reason
            LIMIT 5
        """)
        
        print("\nItems -> Customers (Returns):")
        for record in items_customers:
            print(f"  {record['item']} -> Customer {record['customer']} (Reason: {record['reason']})")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {str(e)}")
        return False

def check_csv_data():
    """Check if CSV data files exist and have content"""
    print("\n📁 Checking CSV data files...")
    
    data_files = [
        'data/inventory.csv',
        'data/suppliers.csv', 
        'data/logistics.csv',
        'data/returns.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✅ {file_path}: {len(df)} rows")
        else:
            print(f"❌ {file_path}: File not found")

if __name__ == "__main__":
    print("🧪 Neo4j Connection and Relationship Test")
    print("=" * 50)
    
    check_csv_data()
    test_neo4j_connection() 