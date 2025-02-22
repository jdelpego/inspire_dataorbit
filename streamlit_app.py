import streamlit as st
import folium
from streamlit_folium import st_folium
import requests

#navigation bar
st.markdown("""
    <style>
    .css-18e3th9 { 
        padding-top: 0rem; 	
        padding-bottom: 0rem;    
    }
    body {
        margin: 0;
        padding: 0;
    }
    .navbar {
        background-color: #00A9A5;
        padding: 5px;
        text-align: center;
        top: 0;
        margin-top: 0px;
        margin-bottom: 20px;
        width: 100%;
        z-index: 1000;
    }

    .navbar a {
        color: white;
        padding: 7px 10px;
        text-decoration: none;
        font-size: 18px;
        display: inline-block;
    }

    .navbar a:hover {
        background-color: #99E1D9;
        color: black;
        transition: 0.3s ease-in;
    }
    .content {
        margin-top: 60px; /* Add margin to push content below the navbar */
    }

    .footer {
        padding: 5px 0;
        bottom: 0;
        margin-bottom: 0;
        margin-top: 50px;
        background-color: #005F60;
    }

    .footer a {
        color: #F4E1D2; 
        font-size: 20px; 
        text-decoration: none;
    }

    .footer a:hover {
        color: #99E1D9;
        text-decoration: underline;
    }
    h1, h3 {
        color: #005F60; 
        font-family: 'Roboto', sans-serif;
    }

    p {
        color: #3E5C5B; 
        font-family: 'Roboto', sans-serif;
    }

    .map-container {
        margin: 0 auto;
        padding-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

#tabs (links)
st.markdown("""
    <div class="navbar">
        <a href="?page=Home" class="{% if page == 'Home' %}active{% endif %}">Home</a>
        <a href="?page=resources" class="{% if page == 'resources' %}active{% endif %}">Resources</a>
    </div>
""", unsafe_allow_html=True)

# Get query params from URL
query_params = st.experimental_get_query_params()

#current page from URL query param
tab = query_params.get("page", ["Home"])[0]  # Default to "home" if no parameter

if tab == "Home":
    # Click
    st.markdown("""
        <h1 style='font-family: Roboto; font-size: 50px; text-align: center'>Predictor</h1>
        <p style='font-family: Roboto; font-size: 20px; text-align: center; padding-right: 10px;'>&nbsp;Prediction Model</p>
    """, unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown('<div class="map-container">', unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; font-family: 'Roboto''>Click on the map to select a location:</h3>", unsafe_allow_html=True)

    st.markdown('<div class="map-container">', unsafe_allow_html=True)

    def city_from_coords(lat, long):
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        response = requests.get(url)
        data = response.json()

        if 'address' in data:
            city = data['address'].get('city', 'unknown') 
            return city
        return "Unknown Location"

    def create_map(lat, lon, zoom=5):
        m = folium.Map(location=[40.0, -120.0], zoom_start=5)
        
        geojson_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
        geojson_data = requests.get(geojson_url).json()

        folium.GeoJson(geojson_data, name="countries").add_to(m)

        def on_click(event):
            lat_click = event.latlng.lat
            long_click = event.latlng.lng

            city = city_from_coords(lat_click, long_click)

            folium.Popup(f"{city}").add_to(folium.Marker([lat_click, long_click]).add_to(m))

        m.on_click(on_click)
        return m

    # Initialize map
    m = create_map(40.0, -120.0)
   
    # Render map 
    map_result = st_folium(m, width=700)

    st.markdown('</div>', unsafe_allow_html=True)

    if map_result and "last_clicked" in map_result: 
        clicked_location = map_result["last_clicked"]

        if clicked_location and "lat" in clicked_location and "lng" in clicked_location:
            latitude = clicked_location["lat"]
            longitude = clicked_location["lng"]
                        
    # Display latitude and longitude separately
            st.markdown(
                f"""
                <p style='text-align: center;'>
                    <strong>Latitude:</strong> {latitude:.6f} <br>
                    <strong>Longitude:</strong> {longitude:.6f}
                </p>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<p style='text-align: center;'>Please click on the map.</p>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; '>This model predicts the likelihood of coastal shrinkage and its effects on house pricing.</p>", unsafe_allow_html=True)


    # Footer
    st.markdown("""
        <div style="text-align: center; font-size: 14px; color: gray;">
            <p>©Inspire, Inc.</p>
        </div>
    """, unsafe_allow_html=True)


elif tab == "resources": 
    st.write('')
