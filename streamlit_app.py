import streamlit as st
import pydeck as pdk
from geopy.geocoders import Nominatim

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

    geolocator = Nominatim(user_agent="inspire-app")

    def cityzip_from_coords(lat, lon):
        try:
            location = geolocator.reverse((lat, lon), language="en", exactly_one=True)
            if location:
                address = location.raw.get("address", {})
                city = address.get("city", "Unknown")
                zipcode = address.get("postcode", "Unknown")
                return city, zipcode
            return "Unknown Location", "Unknown Zipcode"
        except Exception as e:
            st.error(f"Error during geocoding: {e}")
            return "Error", "Error"


    def create_map(lat, lon, zoom=5):
        return pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=40.7128,  
                longitude=-74.0060,
                zoom=5,
                pitch=0
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    [],
                    get_position="[longitude, latitude]",
                    get_radius=20000,
                    get_fill_color=[255, 0, 0],
                    pickable=True,
                    opacity=0.5
                ),
            ]
        )

    st.title("Sea level Rise Prediction")
    st.markdown("""
    <style>
    .map-container {
        margin: 0 auto;
        padding-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)
    
    map = create_map()
    st.pydeck_chart(map)
   
    if st.session_state.get("last_clicked", None): 
       lat, lon = st.session_state["last_clicked"]

    city, zipcode = cityzip_from_coords(lat, lon)

    st.markdown(f"""
        <p style='text-align: center;'>
            <strong>Latitude:</strong> {lat:.6f} <br>
            <strong>Longitude:</strong> {lon:.6f} <br>
            <strong>City:</strong> {city} <br>
            <strong>Zipcode:</strong> {zipcode}
        </p>
        """,
        unsafe_allow_html=True,
    )

    def on_click(event):
        lat = event["latitude"]
        lon = event["longitude"]

        st.session_state["last_clicked"] = (lat, lon)

        st.experimental_rerun()

    map.on_click(on_click)

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
