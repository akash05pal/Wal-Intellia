import os
from dotenv import load_dotenv
from graph_db.neo4j_client import Neo4jClient
from vector_db.vector_store import VectorStore
import pandas as pd
import pickle

load_dotenv()

# Load environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Paths to data
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
INVENTORY_PATH = os.path.join(DATA_DIR, 'inventory.csv')
SUPPLIERS_PATH = os.path.join(DATA_DIR, 'suppliers.csv')
LOGISTICS_PATH = os.path.join(DATA_DIR, 'logistics.csv')
RETURNS_PATH = os.path.join(DATA_DIR, 'returns.csv')

# 1. Load data into Neo4j
graph = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
print('Loading data into Neo4j...')
graph.create_graph_from_csvs(INVENTORY_PATH, SUPPLIERS_PATH, LOGISTICS_PATH, RETURNS_PATH)
graph.close()
print('Neo4j graph created.')

# 2. Build vector index from all text fields in the CSVs
print('Building vector index...')
texts = []
# Inventory
inv = pd.read_csv(INVENTORY_PATH)
for _, row in inv.iterrows():
    texts.append(f"Item: {row['item_name']} (Stock: {row['stock']}, Demand: {row['predicted_demand_next_week']})")
# Suppliers
sup = pd.read_csv(SUPPLIERS_PATH)
for _, row in sup.iterrows():
    texts.append(f"Supplier: {row['supplier_name']} (On-time: {row['on_time_rate']}, Return rate: {row['return_rate']})")
# Logistics
log = pd.read_csv(LOGISTICS_PATH)
for _, row in log.iterrows():
    texts.append(f"Shipment: {row['shipment_id']} for Item {row['item_id']} via {row['carrier']} (Delayed: {row['delayed']}, Reason: {row['delay_reason']})")
# Returns
ret = pd.read_csv(RETURNS_PATH)
for _, row in ret.iterrows():
    texts.append(f"Return: {row['return_id']} for Item {row['item_id']} by Customer {row['customer_id']} (Reason: {row['return_reason']}, Date: {row['date']})")

vector_store = VectorStore()
vector_store.build_index(texts)

# Optionally, save the vector store for later use
with open(os.path.join('vector_db', 'vector_index.pkl'), 'wb') as f:
    pickle.dump({'texts': vector_store.texts, 'embeddings': vector_store.embeddings}, f)

print('Vector index built and saved.') 