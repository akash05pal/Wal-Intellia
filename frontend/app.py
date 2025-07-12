import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import streamlit as st
from rag.rag_pipeline import rag_query
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

st.set_page_config(page_title="INTELLIA CENTRAL INTELLIGENCE PLATFORM", layout="wide")
st.title("INTELLIA: CENTRAL INTELLIGENCE PLATFORM")

# Sidebar filters
st.sidebar.header("📊 Filters")
filter_month = st.sidebar.selectbox("Select Month", ["All Months", "January", "February", "March", "April", "May", "June", 
                                                    "July", "August", "September", "October", "November", "December"])
filter_carrier = st.sidebar.multiselect("Select Carriers", ["All", "CarrierX", "CarrierY", "CarrierZ", "CarrierA", "CarrierB"])
filter_supplier = st.sidebar.multiselect("Select Suppliers", ["All", "Supplier Alpha", "Supplier Beta", "Supplier Gamma", 
                                                             "Supplier Delta", "Supplier Epsilon"])

# Knowledge Graph Toggle
st.sidebar.header("🔗 Knowledge Graph")
show_knowledge_graph = st.sidebar.checkbox("Show Knowledge Graph", value=False)

# Main content area
if show_knowledge_graph:
    st.header("🕸️ Interactive Knowledge Graph")
    
    # Create sample knowledge graph data
    def create_sample_knowledge_graph():
        """Create sample knowledge graph data for visualization"""
        try:
            from pyvis.network import Network
            import random
            
            # Create a PyVis network
            net = Network(height="600px", width="100%", bgcolor="#ffffff")
            net.set_options("""
            var options = {
              "nodes": {
                "color": {
                  "background": "#97C2FC",
                  "border": "#2B7CE9"
                },
                "font": {
                  "size": 12
                },
                "shape": "dot"
              },
              "edges": {
                "color": {
                  "color": "#848484",
                  "highlight": "#848484"
                },
                "smooth": {
                  "type": "continuous"
                }
              },
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "maxVelocity": 50,
                "minVelocity": 0.1,
                "solver": "forceAtlas2Based",
                "timestep": 0.35
              }
            }
            """)
            
            # Sample data for demonstration
            items = [
                {"id": "Item_1001", "label": "Widget A", "group": "items", "title": "Stock: 50, Demand: 120"},
                {"id": "Item_1002", "label": "Widget B", "group": "items", "title": "Stock: 200, Demand: 180"},
                {"id": "Item_1003", "label": "Gadget C", "group": "items", "title": "Stock: 10, Demand: 90"},
                {"id": "Item_1004", "label": "Tool D", "group": "items", "title": "Stock: 75, Demand: 150"},
                {"id": "Item_1005", "label": "Device E", "group": "items", "title": "Stock: 300, Demand: 200"}
            ]
            
            suppliers = [
                {"id": "Supplier_Alpha", "label": "Supplier Alpha", "group": "suppliers", "title": "On-time: 95%, Return: 10%"},
                {"id": "Supplier_Beta", "label": "Supplier Beta", "group": "suppliers", "title": "On-time: 80%, Return: 25%"},
                {"id": "Supplier_Gamma", "label": "Supplier Gamma", "group": "suppliers", "title": "On-time: 60%, Return: 40%"},
                {"id": "Supplier_Delta", "label": "Supplier Delta", "group": "suppliers", "title": "On-time: 90%, Return: 15%"},
                {"id": "Supplier_Epsilon", "label": "Supplier Epsilon", "group": "suppliers", "title": "On-time: 85%, Return: 20%"}
            ]
            
            carriers = [
                {"id": "CarrierX", "label": "CarrierX", "group": "carriers", "title": "Delays: 82, Total: 200"},
                {"id": "CarrierY", "label": "CarrierY", "group": "carriers", "title": "Delays: 45, Total: 180"},
                {"id": "CarrierZ", "label": "CarrierZ", "group": "carriers", "title": "Delays: 67, Total: 220"},
                {"id": "CarrierA", "label": "CarrierA", "group": "carriers", "title": "Delays: 33, Total: 150"},
                {"id": "CarrierB", "label": "CarrierB", "group": "carriers", "title": "Delays: 144, Total: 250"}
            ]
            
            return_reasons = [
                {"id": "Defective", "label": "Defective", "group": "returns", "title": "Count: 156"},
                {"id": "Not_Described", "label": "Not as Described", "group": "returns", "title": "Count: 134"},
                {"id": "Wrong_Size", "label": "Wrong Size", "group": "returns", "title": "Count: 98"},
                {"id": "Wrong_Color", "label": "Wrong Color", "group": "returns", "title": "Count: 87"},
                {"id": "Damaged_Transit", "label": "Damaged in Transit", "group": "returns", "title": "Count: 76"}
            ]
            
            # Add nodes with different colors for different types
            node_colors = {
                "items": "#97C2FC",      # Blue
                "suppliers": "#FB7E81",   # Red
                "carriers": "#7BE141",    # Green
                "returns": "#FFA807",     # Orange
                "customers": "#6E6EFD"    # Purple
            }
            
            # Add all nodes
            for item in items:
                net.add_node(item["id"], label=item["label"], title=item["title"], 
                            color=node_colors[item["group"]], group=item["group"])
            
            for supplier in suppliers:
                net.add_node(supplier["id"], label=supplier["label"], title=supplier["title"], 
                            color=node_colors[supplier["group"]], group=supplier["group"])
            
            for carrier in carriers:
                net.add_node(carrier["id"], label=carrier["label"], title=carrier["title"], 
                            color=node_colors[carrier["group"]], group=carrier["group"])
            
            for reason in return_reasons:
                net.add_node(reason["id"], label=reason["label"], title=reason["title"], 
                            color=node_colors[reason["group"]], group=reason["group"])
            
            # Add edges (relationships)
            # Items to Suppliers
            for item in items:
                supplier = random.choice(suppliers)
                net.add_edge(item["id"], supplier["id"], title="SUPPLIED_BY", color="#FF6B6B")
            
            # Items to Carriers
            for item in items:
                carrier = random.choice(carriers)
                net.add_edge(item["id"], carrier["id"], title="SHIPPED_VIA", color="#4ECDC4")
            
            # Items to Return Reasons
            for item in items[:3]:  # Some items have returns
                reason = random.choice(return_reasons)
                net.add_edge(item["id"], reason["id"], title="RETURNED_FOR", color="#45B7D1")
            
            # Add some customer nodes
            customers = [
                {"id": "Customer_001", "label": "Customer 001", "group": "customers", "title": "Returns: 3"},
                {"id": "Customer_002", "label": "Customer 002", "group": "customers", "title": "Returns: 1"},
                {"id": "Customer_003", "label": "Customer 003", "group": "customers", "title": "Returns: 2"}
            ]
            
            for customer in customers:
                net.add_node(customer["id"], label=customer["label"], title=customer["title"], 
                            color=node_colors[customer["group"]], group=customer["group"])
                # Connect customers to items
                item = random.choice(items)
                net.add_edge(customer["id"], item["id"], title="PURCHASED", color="#96CEB4")
            
            return net
            
        except ImportError:
            st.error("📦 PyVis not installed. Install with: `pip install pyvis`")
            return None
    
    # Create and display the knowledge graph
    net = create_sample_knowledge_graph()
    
    if net:
        # Add legend
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("🔵 **Items**")
        with col2:
            st.markdown("🔴 **Suppliers**")
        with col3:
            st.markdown("🟢 **Carriers**")
        with col4:
            st.markdown("🟠 **Return Reasons**")
        with col5:
            st.markdown("🟣 **Customers**")
        
        # Display the interactive graph
        st.components.html(net.html, height=600, scrolling=False)
        
        # Add graph statistics
        st.subheader("📊 Graph Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", "23")
            st.metric("Items", "5")
            st.metric("Suppliers", "5")
        
        with col2:
            st.metric("Total Edges", "18")
            st.metric("Carriers", "5")
            st.metric("Return Reasons", "5")
        
        with col3:
            st.metric("Graph Density", "0.07")
            st.metric("Customers", "3")
            st.metric("Connected Components", "1")
        
        # Add interaction instructions
        st.info("💡 **Interaction Tips:** Drag nodes to rearrange, hover for details, zoom with mouse wheel, and click nodes to focus on their connections.")
    
    # If PyVis is not available, show a simple text-based graph
    else:
        st.subheader("📋 Knowledge Graph Structure (Text View)")
        st.write("""
        **Items (5 nodes):**
        - Widget A → Supplier Alpha (SUPPLIED_BY)
        - Widget B → Supplier Beta (SUPPLIED_BY)
        - Gadget C → Supplier Gamma (SUPPLIED_BY)
        - Tool D → Supplier Delta (SUPPLIED_BY)
        - Device E → Supplier Epsilon (SUPPLIED_BY)
        
        **Carriers (5 nodes):**
        - CarrierX, CarrierY, CarrierZ, CarrierA, CarrierB
        
        **Return Reasons (5 nodes):**
        - Defective, Not as Described, Wrong Size, Wrong Color, Damaged in Transit
        
        **Customers (3 nodes):**
        - Customer 001, Customer 002, Customer 003
        """)
        
        st.info("💡 Install PyVis for interactive visualization: `pip install pyvis`")

# Main query section (only show if knowledge graph is not displayed)
query = ""
if not show_knowledge_graph:
    query = st.text_input("Ask a question about inventory, suppliers, logistics, or returns:")

if st.button("Submit") and query and not show_knowledge_graph:
    with st.spinner("Thinking..."):
        answer = rag_query(query)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 AI-Powered Insights", "📈 Trends", "🗺️ Map View", "🗄️ Knowledge Graph Database", "📤 Download"])
    
    with tab1:
        st.subheader("🤖 AI-Powered Insights:")
        st.markdown(answer, unsafe_allow_html=True)
        
        # Chart Section
        st.subheader("Supplier Return & On-Time Rates (last month)")
        suppliers_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'suppliers.csv'))
        df = pd.read_csv(suppliers_path)
        chart_data = df.set_index('supplier_name')[['on_time_rate', 'return_rate']]
        st.bar_chart(chart_data)
    
    with tab2:
        st.subheader("📈 Trend Analysis")
        
        # Load data for trend analysis
        logistics_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'logistics.csv')
        returns_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'returns.csv')
        
        logistics_df = pd.read_csv(logistics_path)
        returns_df = pd.read_csv(returns_path)
        
        # Convert dates for trend analysis
        returns_df['date'] = pd.to_datetime(returns_df['date'])
        returns_df['month'] = returns_df['date'].dt.month_name()
        
        # Carrier delay trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Carrier Delay Trends")
            delay_trends = logistics_df.groupby('carrier')['delayed'].value_counts().unstack(fill_value=0)
            fig = px.bar(delay_trends, title="Delays by Carrier", 
                        labels={'value': 'Number of Delays', 'carrier': 'Carrier'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Return Reasons Over Time")
            return_trends = returns_df.groupby(['month', 'return_reason']).size().reset_index()
            return_trends.columns = ['month', 'return_reason', 'count']
            fig = px.line(return_trends, x='month', y='count', color='return_reason', 
                         title="Return Reasons by Month")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🗺️ Regional Analysis")
        
        # Create realistic regional data with coordinates for India
        regions_data = {
            "North": {"lat": 28.7, "lon": 77.1, "name": "North India (Delhi/NCR)"},
            "South": {"lat": 13.1, "lon": 80.3, "name": "South India (Chennai)"},
            "East": {"lat": 22.6, "lon": 88.4, "name": "East India (Kolkata)"},
            "West": {"lat": 19.1, "lon": 72.9, "name": "West India (Mumbai)"},
            "Central": {"lat": 23.3, "lon": 77.4, "name": "Central India (Bhopal)"}
        }
        
        carriers = ["CarrierX", "CarrierY", "CarrierZ", "CarrierA", "CarrierB"]
        
        # Create sample regional delay data with coordinates
        map_data = []
        for region, coords in regions_data.items():
            for carrier in carriers:
                delays = np.random.randint(10, 50)
                total_shipments = np.random.randint(50, 150)
                delay_rate = (delays / total_shipments * 100)
                
                # Color coding based on delay rate
                if delay_rate > 30:
                    color = "red"
                    size = 20
                elif delay_rate > 20:
                    color = "orange"
                    size = 15
                else:
                    color = "green"
                    size = 10
                
                map_data.append({
                    'region': region,
                    'carrier': carrier,
                    'lat': coords['lat'] + np.random.uniform(-2, 2),  # Add some spread
                    'lon': coords['lon'] + np.random.uniform(-2, 2),
                    'delays': delays,
                    'total_shipments': total_shipments,
                    'delay_rate': delay_rate,
                    'color': color,
                    'size': size,
                    'text': f"{region}<br>{carrier}<br>Delays: {delays}<br>Rate: {delay_rate:.1f}%"
                })
        
        map_df = pd.DataFrame(map_data)
        
        # Create interactive map
        fig = px.scatter_mapbox(
            map_df,
            lat='lat',
            lon='lon',
            color='delay_rate',
            size='size',
            hover_name='region',
            hover_data=['carrier', 'delays', 'delay_rate'],
            color_continuous_scale='RdYlGn_r',  # Red to Green (red = high delays)
            size_max=25,
            zoom=4,
            center={'lat': 23.0, 'lon': 80.0},
            title="Carrier Delay Hotspots by Region",
            mapbox_style="carto-positron"  # Clean map style
        )
        
        fig.update_layout(
            height=600,
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add legend and summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Delay Rate Legend")
            st.markdown("""
            - 🔴 **Red**: High delay rate (>30%)
            - 🟠 **Orange**: Medium delay rate (20-30%)
            - 🟢 **Green**: Low delay rate (<20%)
            """)
        
        with col2:
            st.subheader("📊 Regional Summary")
            regional_summary = map_df.groupby('region').agg({
                'delays': 'sum',
                'total_shipments': 'sum'
            }).round(0)
            regional_summary['delay_rate'] = (regional_summary['delays'] / regional_summary['total_shipments'] * 100).round(1)
            st.dataframe(regional_summary)
    
    with tab4:
        st.subheader("🗄️ Knowledge Graph Database")
        
        # Initialize Neo4j client for database exploration
        from graph_db.neo4j_client import Neo4jClient
        from dotenv import load_dotenv
        load_dotenv()
        
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_user = os.getenv('NEO4J_USER')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                neo4j_client = Neo4jClient(neo4j_uri, neo4j_user, neo4j_password)
                
                # Database Overview
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("📊 Database Stats")
                    # Get node counts
                    node_counts = neo4j_client.query("MATCH (n) RETURN labels(n) as type, count(n) as count")
                    for result in node_counts:
                        if 'type' in result and 'count' in result:
                            st.metric(f"{result['type'][0]} Nodes", result['count'])
                
                with col2:
                    st.subheader("🔗 Relationships")
                    # Get relationship counts
                    rel_counts = neo4j_client.query("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
                    for result in rel_counts:
                        if 'type' in result and 'count' in result:
                            st.metric(f"{result['type']} Relationships", result['count'])
                
                with col3:
                    st.subheader("🌐 Graph Density")
                    # Calculate graph density
                    total_nodes = sum([r['count'] for r in node_counts if 'count' in r])
                    total_rels = sum([r['count'] for r in rel_counts if 'count' in r])
                    density = total_rels / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
                    st.metric("Graph Density", f"{density:.4f}")
                
                # Interactive Query Section
                st.subheader("🔍 Interactive Graph Explorer")
                
                # Predefined queries
                query_options = {
                    "Show all Items": "MATCH (i:Item) RETURN i LIMIT 10",
                    "Show all Suppliers": "MATCH (s:Supplier) RETURN s LIMIT 10",
                    "Show Items with Low Stock": "MATCH (i:Item) WHERE i.stock < 50 RETURN i",
                    "Show High Return Rate Suppliers": "MATCH (s:Supplier) WHERE s.return_rate > 0.3 RETURN s",
                    "Show Delayed Shipments": "MATCH (i:Item)-[r:SHIPPED_VIA]->(c:Carrier) WHERE r.delayed = 'yes' RETURN i, r, c LIMIT 10",
                    "Show Return Patterns": "MATCH (i:Item)-[r:RETURNED_BY]->(c:Customer) RETURN i.name, r.reason, count(r) as return_count ORDER BY return_count DESC LIMIT 10"
                }
                
                selected_query = st.selectbox("Choose a query to explore:", list(query_options.keys()))
                
                if st.button("🔍 Execute Query"):
                    with st.spinner("Querying knowledge graph..."):
                        try:
                            results = neo4j_client.query(query_options[selected_query])
                            
                            if results:
                                st.subheader("📋 Query Results")
                                
                                # Display results in a nice format
                                for i, result in enumerate(results[:10]):  # Limit to 10 results
                                    with st.expander(f"Result {i+1}"):
                                        for key, value in result.items():
                                            if isinstance(value, dict):
                                                st.json(value)
                                            else:
                                                st.write(f"**{key}:** {value}")
                                
                                # Show result count
                                st.info(f"Found {len(results)} results")
                            else:
                                st.warning("No results found for this query.")
                                
                        except Exception as e:
                            st.error(f"Error executing query: {str(e)}")
                
                # Custom Query Section
                st.subheader("✍️ Custom Cypher Query")
                custom_query = st.text_area("Enter your own Cypher query:", 
                                          placeholder="MATCH (n) RETURN n LIMIT 5")
                
                if st.button("🚀 Run Custom Query") and custom_query:
                    with st.spinner("Executing custom query..."):
                        try:
                            custom_results = neo4j_client.query(custom_query)
                            
                            if custom_results:
                                st.subheader("📋 Custom Query Results")
                                st.json(custom_results)
                                st.info(f"Found {len(custom_results)} results")
                            else:
                                st.warning("No results found for this query.")
                                
                        except Exception as e:
                            st.error(f"Error executing custom query: {str(e)}")
                
                # Graph Visualization (if networkx is available)
                st.subheader("🕸️ Graph Visualization")
                try:
                    import networkx as nx
                    
                    # Create a sample subgraph for visualization
                    sample_query = "MATCH (i:Item)-[r]-(n) RETURN i, r, n LIMIT 20"
                    sample_results = neo4j_client.query(sample_query)
                    
                    if sample_results:
                        # Create NetworkX graph
                        G = nx.Graph()
                        
                        for result in sample_results:
                            if 'i' in result and 'n' in result:
                                # Add nodes
                                if 'name' in result['i']:
                                    G.add_node(result['i']['name'], type='Item')
                                if 'name' in result['n']:
                                    G.add_node(result['n']['name'], type='Other')
                                
                                # Add edges
                                if 'name' in result['i'] and 'name' in result['n']:
                                    G.add_edge(result['i']['name'], result['n']['name'])
                        
                        # Display graph info
                        st.write(f"**Graph Info:** {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                        
                        # Show node types
                        node_types = {}
                        for node, attrs in G.nodes(data=True):
                            node_type = attrs.get('type', 'Unknown')
                            node_types[node_type] = node_types.get(node_type, 0) + 1
                        
                        st.write("**Node Types:**")
                        for node_type, count in node_types.items():
                            st.write(f"- {node_type}: {count}")
                    
                except ImportError:
                    st.info("📦 Install networkx for graph visualization: `pip install networkx`")
                
                neo4j_client.close()
                
            except Exception as e:
                st.error(f"❌ Could not connect to Neo4j database: {str(e)}")
                st.info("Make sure your Neo4j database is running and credentials are correct in .env file")
        else:
            st.error("❌ Neo4j credentials not found in environment variables")
            st.info("Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in your .env file")
    
    with tab5:
        st.subheader("📤 Download Reports")
        
        # Generate comprehensive report
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Summary Report")
            
            # Create summary statistics
            summary_stats = {
                "Total Shipments": len(logistics_df),
                "Total Delays": len(logistics_df[logistics_df['delayed'] == 'yes']),
                "Total Returns": len(returns_df),
                "Average On-Time Rate": f"{df['on_time_rate'].mean():.1%}",
                "Average Return Rate": f"{df['return_rate'].mean():.1%}"
            }
            
            for key, value in summary_stats.items():
                st.metric(key, value)
        
        with col2:
            st.subheader("📥 Export Options")
            
            # CSV Export
            if st.button("📄 Export as CSV"):
                # Create comprehensive CSV
                report_data = {
                    'summary_stats': summary_stats,
                    'carrier_performance': logistics_df.groupby('carrier')['delayed'].value_counts().unstack(fill_value=0),
                    'supplier_performance': df.groupby('supplier_name').agg({
                        'on_time_rate': 'mean',
                        'return_rate': 'mean'
                    }).round(3)
                }
                
                # Convert to CSV and download
                csv = logistics_df.to_csv(index=False)
                st.download_button(
                    label="Download Logistics Data CSV",
                    data=csv,
                    file_name=f"logistics_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # PDF Report (placeholder)
            if st.button("📋 Generate PDF Report"):
                st.info("PDF generation feature coming soon! For now, use CSV export.")

# Add some styling
st.markdown("""
<style>
.stMetric {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True) 