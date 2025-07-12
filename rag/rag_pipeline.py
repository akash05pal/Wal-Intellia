import os
from dotenv import load_dotenv
from openai import OpenAI
from vector_db.vector_store import VectorStore
from graph_db.neo4j_client import Neo4jClient
import pandas as pd

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Initialize services
vector_store = VectorStore()
vector_store.load_index(os.path.join(os.path.dirname(__file__), '..', 'vector_db', 'vector_index.pkl'))
print(f"VectorStore index: {vector_store.index}")

neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
client = OpenAI(api_key=OPENAI_API_KEY)

def get_statistical_context(query):
    """Get statistical summaries from the data"""
    stats = []
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    if 'carrier' in query.lower() or 'delay' in query.lower() or 'logistics' in query.lower():
        logistics_df = pd.read_csv(os.path.join(data_dir, 'logistics.csv'))
        delay_counts = logistics_df[logistics_df['delayed'] == 'yes'].groupby('carrier').size()
        total_shipments = logistics_df.groupby('carrier').size()
        stats.append(f"Carrier Delay Counts:\n{delay_counts.to_string()}")
        stats.append(f"Total Shipments per Carrier:\n{total_shipments.to_string()}")

    if 'supplier' in query.lower() or 'return' in query.lower():
        suppliers_df = pd.read_csv(os.path.join(data_dir, 'suppliers.csv'))
        avg_rates = suppliers_df.groupby('supplier_name').agg({
            'on_time_rate': 'mean',
            'return_rate': 'mean'
        }).round(3)
        stats.append(f"Supplier Performance (Avg On-Time & Return Rates):\n{avg_rates.to_string()}")

    if 'return' in query.lower():
        returns_df = pd.read_csv(os.path.join(data_dir, 'returns.csv'))
        return_reasons = returns_df['return_reason'].value_counts()
        stats.append(f"Top Return Reasons:\n{return_reasons.to_string()}")

    return "\n\n".join(stats)

def get_supplier_visual_insight():
    """Returns hardcoded insights from supplier performance chart image"""
    return """
From the 'Supplier Return & On-Time Rates (last month)' chart:
- Top performing suppliers include: Supplier Beta, Advanced Solutions, Core Systems, Global Parts Ltd.
- Relatively lower performance observed from: Supplier Xi, Supplier Zeta, TechCorp Inc.
- Overall, supplier performance appears consistent across most vendors.
- Suggest prioritizing high-performing suppliers for critical shipments.
"""

def build_prompt(user_query, vector_chunks, stats_context, graph_context):
    """Builds a clean, structured prompt for the LLM"""
    context_section = "\n---\n".join([chunk[0] for chunk in vector_chunks])
    graph_section = str(graph_context) if graph_context else "No related graph data found."
    visual_insight = get_supplier_visual_insight()

    prompt = f"""
You are an intelligent assistant analyzing supply chain and logistics data.

User Query:
{user_query}

Relevant Context from Vector Search:
{context_section}

Statistical Analysis from Dataset:
{stats_context}

Supplier Visual Performance Insight:
{visual_insight}

Graph Database Insights:
{graph_section}

Instructions:
1. Answer based on the full dataset insights and factual numbers.
2. Clearly mention which carriers, suppliers, or items are performing better or worse.
3. Include specific statistics like counts and percentages.
4. Point out any operational risks, inefficiencies, or key takeaways.
5. Write in simple, clear language, avoiding repetition.
6. Format your response as HTML with proper tags for better UI rendering:
   - Use <h3> for main insights
   - Use <ul><li> for lists
   - Use <strong> for important numbers/percentages
   - Use <p> for paragraphs
   - Use <div style="color: red;"> for risks/warnings
   - Use <div style="color: green;"> for positive insights
"""
    return prompt.strip()

def rag_query(user_query):
    """End-to-end RAG pipeline"""
    top_chunks = vector_store.search(user_query, top_k=10)
    stats_context = get_statistical_context(user_query)

    cypher = """
    MATCH (i:Item)-[r]-(n) 
    WHERE i.name CONTAINS $q OR n.name CONTAINS $q 
    RETURN i, r, n 
    LIMIT 10
    """
    graph_context = neo4j_client.query(cypher, {"q": user_query.split()[0]})
    prompt = build_prompt(user_query, top_chunks, stats_context, graph_context)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
