# Route Optimization for Delivery Services

## 📌 Problem Statement
Last-mile delivery is expensive and time-consuming. Delivery companies waste fuel and time due to inefficient routing. 

## 🧠 Methodology
This project solves the problem using a hybrid approach:
1. **Machine Learning (K-Means Clustering):** Divides the city into 3 distinct zones, simulating 3 different delivery vehicles.
2. **Optimization (Google OR-Tools):** Solves the Traveling Salesperson Problem (TSP) for each zone to find the shortest possible path.
3. **Visualization:** Displays the optimized routes on an interactive map using Folium.
4. **Export:** Generates a PDF report of the optimized routes using FPDF.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Scikit-Learn (ML), Google OR-Tools (Optimization), Streamlit (Web UI), Folium (Maps), FPDF (PDF Generation)

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn ortools streamlit streamlit-folium fpdf2