import streamlit as st
import folium
import requests
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from singlestoredb import connect
from singlestoredb.exceptions import DatabaseError
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

google_maps_api_key = st.secrets["api_key"]["google_maps_api_key"]

st.markdown("""
    <style>
    .block-container {
        padding-top: 0px !important;
        margin-top: 0px !important;
    }
          
    .css-18e3th9 { 
        padding-top: 0rem; 	
        padding-bottom: 0rem;    
    }
    body {
        margin: 0;
        padding: 0;
        background-color: #F4A300 !important;
    }
    .navbar {
        position: fixed;
        background-color: #00A9A5 !important;
        padding: 15px;
        text-align: center;
        top: 50px !important;
        left: 0;
        right: 0;
        width: 100%;
        z-index: 10000 !important; 
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);

    }
            
    .content {
        margin-top: 30px;
    }

    .navbar a {
        color: white !important;
        padding: 12px 20px !important;
        text-decoration: none !important; 
        font-size: 20px !important;
        font-weight: bold;
        display: inline-block !important;
        transition: all 0.3s ease;
    }
    .navbar a:hover {
        background-color: #99E1D9;
        color: black !important;
        border-radius: 5px;
    }
    .content {
        margin-top: 70px !important;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="navbar">
        <a href="?page=Home">🏠 Home</a>
        <a href="?page=resources">📚 Resources</a>
    </div>
""", unsafe_allow_html=True)


def predict_flooding_year(altitude_mm, model, future_X, base_sea_level, start_year, max_years=500):
    """
    Predict the year when a specific altitude will be flooded.
    altitude_mm: altitude in millimeters above current sea level
    max_years: maximum number of years to predict into the future
    """
    # Extend prediction range if needed
    years_needed = np.arange(start_year + 1, start_year + max_years + 1)

    nweights =  future_X['year'] - np.min(future_X['year']) + 1
     
    # Project CO2 emissions using exponential model
    log_emissions = np.log(future_X['Emissions'])
    exp_model = np.polyfit(future_X['year'], log_emissions, 1, w=nweights)
    future_emissions = np.exp(np.polyval(exp_model, years_needed))
    
    # Create extended prediction data
    extended_X = pd.DataFrame({
        'year': years_needed,
        'Emissions': future_emissions
    })
    
    # Make predictions
    future_levels = model.predict(poly.transform(extended_X))
    
    # Find when sea level reaches the altitude
    sea_level_rise = future_levels - base_sea_level
    flooding_levels = sea_level_rise >= altitude_mm
    
    if not any(flooding_levels):
        return None, None  # Location won't flood within max_years
    
    flooding_year = years_needed[flooding_levels][0]
    years_until_flooding = flooding_year - start_year
    
    return flooding_year, years_until_flooding

try:
    # Connect to SingleStore
    conn = connect(
        host='svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com',
        port=3333,
        user='jdelpego',
        password='fTVuFI26cwOVwAB7WVWybNqcBTrUP9KE',
        database='db_luke_503d4'
    )

    # Query sea level data - verify the year range
    sea_level_query = """
        SELECT year, `mmfrom1993-2008average` as sea_level
        FROM 1880sealevel
        ORDER BY year
    """
    sea_level_df = pd.read_sql(sea_level_query, conn)
    
    co2_query = """
        SELECT Year as year, Emissions
        FROM GlobalCO2Emissions
        ORDER BY Year
    """
    co2_df = pd.read_sql(co2_query, conn)
    
    # Convert columns to numeric types
    sea_level_df['sea_level'] = pd.to_numeric(sea_level_df['sea_level'], errors='coerce')
    co2_df['Emissions'] = pd.to_numeric(co2_df['Emissions'], errors='coerce')

except DatabaseError as e:
    print(f"Failed to connect to database or execute query: {e}")
    sys.exit(1)
finally:
    conn.close()


try:
    # Verify data coverage before merge
    print("\nVerifying data coverage:")
    sea_level_years = set(sea_level_df['year'])
    co2_years = set(co2_df['year'])
    overlap_years = sea_level_years.intersection(co2_years)

    # Merge sea level and CO2 data
    merged_df = pd.merge(sea_level_df, co2_df, on='year', how='inner')
    
    # Prepare features and target.
    X = merged_df[['year', 'Emissions']]
    y = merged_df['sea_level']
    
    # Create quadratic features (degree=2) using PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Train the quadratic model using LinearRegression
    model = LinearRegression()
    model.fit(X_poly, y)
    current_year = merged_df['year'].max()
    current_sea_level = merged_df['sea_level'].iloc[-1]
except Exception as e:
    print(f"Error during analysis: {e}")
    print(f"Error details: {str(e)}")

# Function to get elevation
def get_elevation(lat, lon):
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={google_maps_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json()
        if result["status"] == "OK":
            return result["results"][0]["elevation"]
    return None

@st.cache_data
def create_map(lat, lon, zoom=5):
    m = folium.Map(location=[lat, lon], zoom_start=zoom)
    folium.Marker(location=[lat, lon], popup="Selected Location").add_to(m)
    m.add_child(folium.ClickForMarker())

    folium.Marker(location=[lat, lon]).add_to(m)
    return m

# Get current page
query_params = st.query_params
tab = query_params.get("page", ["Home"])[0]

if tab == "Home":
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown("""
        <h1 style='text-align: center; color: #333;'>🌊 Sea Level Predictor</h1>
        <p style='text-align: center;'>Prediction Model</p>
    """, unsafe_allow_html=True)

    lat, lon = 34.4356, -119.8276

    st.markdown('<div class="map-container">', unsafe_allow_html=True)

    m = create_map(lat, lon)

    map_result = st_folium(m, width="100%", height=500)  

    st.markdown('</div>', unsafe_allow_html=True)
    elevation = 0;
    if map_result and "last_clicked" in map_result:
        clicked_location = map_result["last_clicked"]

        if clicked_location and "lat" in clicked_location and "lng" in clicked_location:
            lat = clicked_location["lat"]
            lon = clicked_location["lng"]
            elevation = get_elevation(lat, lon)
            flooding_year, years_until = predict_flooding_year(
                elevation*100, model, X, current_sea_level, current_year
            )

    st.markdown(f"""
    <div style="text-align: center; font-size: 18px;">
        <p><strong>📍Latitude:</strong> {lat}</p>
        <p><strong>📍Longitude:</strong> {lon}</p>
        <p><strong>📏Altitude:</strong> {elevation:.2f} meters</p>
    </div>
""", unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align: center;'>Please click on the map.</p>
        <p style='text-align: center;'>This model predicts when a place will sink due to rising sea levels.</p>
        <div style='text-align: center; font-size: 14px; color: gray;'>
            <p>©Inspire, Inc.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="content">', unsafe_allow_html=True)

elif tab == "resources":
    st.write("Resources page content here.")
