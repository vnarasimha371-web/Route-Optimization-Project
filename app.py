import streamlit as st
import pandas as pd
import numpy as np
import math
import io
from sklearn.cluster import KMeans
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
import datetime

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Route Optimizer")
st.title("🚚 Route Optimization for Delivery Services")
st.write("This system uses Machine Learning (K-Means) to divide delivery zones and OR-Tools to calculate the most efficient route for each vehicle with Capacity Constraints.")

# --- 1. GEOSPATIAL DATA HANDLING: HAVERSINE FORMULA ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def create_distance_matrix(df):
    matrix = []
    for i in range(len(df)):
        row = []
        for j in range(len(df)):
            # Use Haversine instead of Euclidean
            dist = haversine_distance(
                df.iloc[i]['Latitude'], df.iloc[i]['Longitude'], 
                df.iloc[j]['Latitude'], df.iloc[j]['Longitude']
            )
            # OR-Tools requires integers, so we multiply by 1000 (meters)
            row.append(int(dist * 1000)) 
        matrix.append(row)
    return matrix

# --- GENERATE DATA WITH DEMAND ---
def generate_data():
    depot = {'Stop_ID': 'Depot', 'Latitude': 28.6139, 'Longitude': 77.2090, 'Demand': 0}
    locations = [depot]
    np.random.seed(42) 
    for i in range(1, 21):
        lat = 28.6139 + np.random.uniform(-0.03, 0.03)
        lon = 77.2090 + np.random.uniform(-0.03, 0.03)
        # Add random demand (1 to 3 boxes per stop)
        demand = np.random.randint(1, 4) 
        locations.append({'Stop_ID': f'Stop_{i}', 'Latitude': lat, 'Longitude': lon, 'Demand': demand})
    return pd.DataFrame(locations)

# --- 2. ALGORITHMIC OPTIMIZATION WITH CONSTRAINTS ---
def get_optimized_route(distance_matrix, demands, vehicle_capacity=15, num_vehicles=1, depot=0):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # 1. Distance Callback
    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 2. Capacity Constraint (CVRP)
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [vehicle_capacity], True, 'Capacity'
    )

    # 3. Solve
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    
    route = []
    if solution:
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
    return route

# --- 3. COST-BENEFIT ANALYSIS (BASELINE VS OPTIMIZED) ---
def calculate_baseline_distance(matrix, demands):
    # Simple Nearest Neighbor Baseline
    unvisited = list(range(1, len(matrix)))
    current = 0
    baseline_route = [0]
    while unvisited:
        nearest = min(unvisited, key=lambda x: matrix[current][x])
        baseline_route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    baseline_route.append(0)
    
    total_dist = sum(matrix[baseline_route[i]][baseline_route[i+1]] for i in range(len(baseline_route)-1))
    return total_dist / 1000 # Convert back to KM

# --- PDF GENERATION ---
def create_pdf(df, routes, distances, baseline_dist, optimized_dist):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="Route Optimization Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    
    # Cost-Benefit Analysis
    savings = ((baseline_dist - optimized_dist) / baseline_dist) * 100
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt=f"Cost-Benefit Analysis:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Baseline (Naive) Distance: {round(baseline_dist, 2)} km", ln=True)
    pdf.cell(200, 10, txt=f"Optimized Route Distance: {round(optimized_dist, 2)} km", ln=True)
    pdf.cell(200, 10, txt=f"Total Distance Reduced by: {round(savings, 2)}%", ln=True)
    pdf.ln(5)
    
    for vehicle_id, route in enumerate(routes):
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(200, 10, txt=f"Vehicle {vehicle_id + 1} Route (Total Distance: {distances[vehicle_id]} km)", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(40, 10, "Stop ID", border=1)
        pdf.cell(40, 10, "Lat", border=1)
        pdf.cell(40, 10, "Lon", border=1)
        pdf.cell(40, 10, "Demand", border=1)
        pdf.ln()
        for stop_idx in route:
            pdf.cell(40, 10, str(df.iloc[stop_idx]['Stop_ID']), border=1)
            pdf.cell(40, 10, str(round(df.iloc[stop_idx]['Latitude'], 4)), border=1)
            pdf.cell(40, 10, str(round(df.iloc[stop_idx]['Longitude'], 4)), border=1)
            pdf.cell(40, 10, str(df.iloc[stop_idx]['Demand']), border=1)
            pdf.ln()
        pdf.ln(5)
    
    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode('latin-1')
    return io.BytesIO(pdf_bytes)

# --- STREAMLIT UI ---
if 'optimized_data' not in st.session_state:
    st.session_state.optimized_data = None

if st.button("Generate Data & Optimize Routes"):
    with st.spinner("Running ML & OR-Tools with Capacity Constraints..."):
        try:
            df = generate_data()
            
            # ML Clustering
            depot_row = df.iloc[[0]].copy()
            stops_df = df.iloc[1:].copy()
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            stops_df['Cluster'] = kmeans.fit_predict(stops_df[['Latitude', 'Longitude']])
            df = pd.concat([depot_row, stops_df]).reset_index(drop=True)
            df.loc[0, 'Cluster'] = -1 
            
            all_routes = []
            all_distances = []
            m = folium.Map(location=[df.iloc[0]['Latitude'], df.iloc[0]['Longitude']], zoom_start=13)
            
            for idx, row in df.iterrows():
                color = 'red' if row['Stop_ID'] == 'Depot' else 'blue'
                folium.Marker([row['Latitude'], row['Longitude']], popup=f"{row['Stop_ID']} (Demand: {row['Demand']})", icon=folium.Icon(color=color)).add_to(m)
            
            colors = ['green', 'purple', 'orange']
            total_optimized_dist = 0
            
            for cluster_id in range(3):
                cluster_df = df[df['Cluster'] == cluster_id].copy()
                cluster_df = pd.concat([depot_row, cluster_df]).reset_index(drop=True)
                
                matrix = create_distance_matrix(cluster_df)
                demands = cluster_df['Demand'].tolist()
                
                # Pass capacity constraint (Max 15 boxes per truck)
                route = get_optimized_route(matrix, demands, vehicle_capacity=15)
                
                total_dist = sum(matrix[route[i]][route[i+1]] for i in range(len(route)-1)) / 1000
                all_distances.append(total_dist)
                total_optimized_dist += total_dist
                
                original_indices = [cluster_df.iloc[i].name for i in route]
                all_routes.append(original_indices)
                
                coords = [(cluster_df.iloc[i]['Latitude'], cluster_df.iloc[i]['Longitude']) for i in route]
                folium.PolyLine(coords, color=colors[cluster_id], weight=3, opacity=0.8, tooltip=f"Vehicle {cluster_id+1}").add_to(m)

            # Calculate Baseline for Cost-Benefit Analysis
            baseline_dist = calculate_baseline_distance(create_distance_matrix(df), df['Demand'].tolist())
            
            st.session_state.optimized_data = {
                'df': df,
                'routes': all_routes,
                'distances': all_distances,
                'map': m,
                'baseline': baseline_dist,
                'optimized_total': total_optimized_dist
            }
            st.success("✅ Optimization Complete!")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

if st.session_state.optimized_data is not None:
    data = st.session_state.optimized_data
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Optimized Map")
        st_folium(data['map'], width=700, height=500)
        
    with col2:
        st.subheader("📊 Metrics & Cost-Benefit Analysis")
        for i, dist in enumerate(data['distances']):
            st.metric(label=f"Vehicle {i+1} Distance", value=f"{round(dist, 2)} km")
        
        # Cost Benefit Analysis Metrics
        st.divider()
        st.write("**Cost-Benefit Analysis**")
        savings = ((data['baseline'] - data['optimized_total']) / data['baseline']) * 100
        st.metric(label="Naive Baseline Distance", value=f"{round(data['baseline'], 2)} km")
        st.metric(label="Optimized Fleet Distance", value=f"{round(data['optimized_total'], 2)} km")
        st.metric(label="Total Distance Reduced", value=f"{round(savings, 2)}%", delta=f"-{round(data['baseline'] - data['optimized_total'], 2)} km")
        
        pdf_file = create_pdf(data['df'], data['routes'], [round(d, 2) for d in data['distances']], data['baseline'], data['optimized_total'])
        st.download_button(
            label="📄 Export PDF Report",
            data=pdf_file,
            file_name="Route_Optimization_Report.pdf",
            mime="application/pdf"
        )