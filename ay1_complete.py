import pandas as pd
import numpy as np
import math
import os
import webbrowser
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium

print("🚀 Starting Day 1 Project...")

# --- 1. GENERATE DATA ---
depot = {'Stop_ID': 'Depot', 'Latitude': 28.6139, 'Longitude': 77.2090}
locations = [depot]
np.random.seed(42) 
for i in range(1, 21):
    lat = 28.6139 + np.random.uniform(-0.03, 0.03)
    lon = 77.2090 + np.random.uniform(-0.03, 0.03)
    locations.append({'Stop_ID': f'Stop_{i}', 'Latitude': lat, 'Longitude': lon})
df = pd.DataFrame(locations)
df.to_csv('delivery_data.csv', index=False)
print("✅ Data generated: delivery_data.csv")

# --- 2. DISTANCE MATRIX ---
def calc_dist(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

matrix = []
for i in range(len(df)):
    row = []
    for j in range(len(df)):
        dist = calc_dist(df.iloc[i]['Latitude'], df.iloc[i]['Longitude'], df.iloc[j]['Latitude'], df.iloc[j]['Longitude'])
        row.append(int(dist * 100000)) 
    matrix.append(row)

# --- 3. BASELINE ROUTE (Nearest Neighbor) ---
unvisited = list(range(1, len(df)))
current = 0
baseline_route = [0]
while unvisited:
    nearest = min(unvisited, key=lambda x: matrix[current][x])
    baseline_route.append(nearest)
    unvisited.remove(nearest)
    current = nearest
baseline_route.append(0)

# --- 4. OPTIMIZED ROUTE (OR-Tools) ---
manager = pywrapcp.RoutingIndexManager(len(matrix), 1, 0)
routing = pywrapcp.RoutingModel(manager)
def dist_callback(from_idx, to_idx):
    return matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]
callback_idx = routing.RegisterTransitCallback(dist_callback)
routing.SetArcCostEvaluatorOfAllVehicles(callback_idx)
params = pywrapcp.DefaultRoutingSearchParameters()
params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
solution = routing.SolveWithParameters(params)

optimized_route = []
if solution:
    idx = routing.Start(0)
    while not routing.IsEnd(idx):
        optimized_route.append(manager.IndexToNode(idx))
        idx = solution.Value(routing.NextVar(idx))
    optimized_route.append(manager.IndexToNode(idx))

# --- 5. DRAW MAP ---
m = folium.Map(location=[df.iloc[0]['Latitude'], df.iloc[0]['Longitude']], zoom_start=13)
for idx, row in df.iterrows():
    color = 'red' if row['Stop_ID'] == 'Depot' else 'blue'
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Stop_ID'], icon=folium.Icon(color=color)).add_to(m)

base_coords = [(df.iloc[i]['Latitude'], df.iloc[i]['Longitude']) for i in baseline_route]
folium.PolyLine(base_coords, color="red", weight=2.5, opacity=0.8, tooltip="Baseline").add_to(m)

opt_coords = [(df.iloc[i]['Latitude'], df.iloc[i]['Longitude']) for i in optimized_route]
folium.PolyLine(opt_coords, color="green", weight=2.5, opacity=0.8, tooltip="Optimized").add_to(m)

m.save("route_comparison.html")

# THIS IS THE LINE THAT OPENS THE BROWSER
webbrowser.open("route_comparison.html") 

# --- 6. RESULTS ---
print("\n🎉 DAY 1 COMPLETE!")
print(f"Baseline Route: {baseline_route}")
print(f"Optimized Route: {optimized_route}")
print(f"📍 Map saved! Find 'route_comparison.html' in your folder: {os.getcwd()}")