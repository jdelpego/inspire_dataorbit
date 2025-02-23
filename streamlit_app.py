import streamlit as st
import folium
import requests
from streamlit_folium import st_folium

google_maps_api_key = st.secrets["api_key"]["google_maps_api_key"]

# Navigation bar
st.markdown("""
    <style>
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
        padding: 5px;
        text-align: center;
        top: 0;
        left: 0;
        right: 0;
        width: 100%;
        z-index: 9999 !important;
    }
    .navbar a {
        color: white !important;
        padding: 7px 10px !important;
        text-decoration: none !important; 
        font-size: 18px !important;
        display: inline-block !important;
    }
    .navbar a:hover {
        background-color: #99E1D9;
        color: black !important;
        transition: 0.3s ease-in !important;
    }
    .content {
        margin-top: 50px !important;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div class="navbar">
        <a href="?page=Home">Home</a>
        <a href="?page=resources">Resources</a>
    </div>
""", unsafe_allow_html=True)

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
query_params = st.query_params()
tab = query_params.get("page", ["Home"])[0]

if tab == "Home":
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown("""
        <h1 style='text-align: center'>Predictor</h1>
        <p style='text-align: center;'>Prediction Model</p>
    """, unsafe_allow_html=True)
    
    lat, lon = 34.4356, -119.8276

    st.markdown('<div class="map-container">', unsafe_allow_html=True)

    m = create_map(lat, lon)

    map_result = st_folium(m, width="100%", height=500)  

    st.markdown('</div>', unsafe_allow_html=True)

    if map_result and "last_clicked" in map_result:
        clicked_location = map_result["last_clicked"]

        if clicked_location and "lat" in clicked_location and "lng" in clicked_location:
            lat = clicked_location["lat"]
            lon = clicked_location["lng"]

    elevation = get_elevation(lat, lon)
    st.markdown(f"""
    <div style="text-align: center; font-size: 18px;">
        <p><strong>Latitude:</strong> {lat}</p>
        <p><strong>Longitude:</strong> {lon}</p>
        <p><strong>Altitude:</strong> {elevation:.2f} meters</p>
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
