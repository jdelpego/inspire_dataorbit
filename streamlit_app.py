import streamlit as st
import folium
from streamlit_folium import st_folium

#navigation bar
st.markdown("""
    <style>
    .css-18e3th9 { 
        padding-top: 0rem; 	
        padding-bottom: 0rem; 
        background-color: #F4E1D2;    
    }
    body {
        margin: 0;
        padding: 0;
        background-color: #F4E1D2;  
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
        font-family: 'Tahoma', sans-serif;
    }

    p {
        color: #003B2D; 
        font-family: 'Tahoma', sans-serif;
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
        <h1 style='font-family: Tahoma; font-size: 50px;'>Predictor</h1>
        <p style='font-family: Tahoma; font-size: 20px; text-align: center; padding-right: 10px;'>&nbsp;Prediction Model</p>
    """, unsafe_allow_html=True)
    st.write('')
    st.write('')
    st.markdown('<div class="map-container">', unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; font-family: 'Tahoma''>Click on the map to select a location:</h3>", unsafe_allow_html=True)

    st.markdown('<div class="map-container">', unsafe_allow_html=True)

    def create_map(lat, lon, zoom=5):
        m = folium.Map(location=[40.0, -120.0], zoom_start=5)
        folium.Marker(location=[40.0, -120.0]).add_to(m) 
        m.add_child(folium.ClickForMarker()) # Add click functionality to the map
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
