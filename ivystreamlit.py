import streamlit as st
import urllib.parse

query_params = st.query_params
lat = query_params.get("lat", [""])[0]
lon = query_params.get("lon", [""])[0]
elevation = query_params.get("elevation", [""])[0]
flood_year = query_params.get("flood_year", [""])[0]
years_until = query_params.get("years_until", [""])[0] 

st.markdown(
    """
    <style>
        /* Set the whole page background to light blue */
        [data-testid="stAppViewContainer"] {
            background-color: #8AD4EB !important;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: transparent;
        }
        .header {
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 20px;
            background-color: #B3E5FC;
            padding: 20px;
            width: 100%;
            border-radius: 5px;
            text-align: center;
            color: #00796B;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 80px;
        }
        .details {
            font-size: 24px;
            margin: 10px 0;
        }
        .sea-level {
            color: #00796B;
            font-size: 25px;
            background-color: #B3E5FC;
            padding: 10px 20px;
            border-radius: 5px;
            display: inline-block;
        }
        .icon {
            width: 50px;
            height: 50px;
        }
        .info-box {
            background-color: #B3E5FC;
            border-radius: 5px;
            margin-top: 20px;
        }
        .info-item {
            background-color: #B3E5FC;
            padding: 10px;
            margin-top: 20px;
            text-align: center;
            font-size: 30px;
            font-weight: bold;
        }
        .info-value {
            background-color: #B3E5FC;
            padding: 10px;
            font-size: 35px;
            color: #00796B;
            font-weight: bold;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# UI Layout
st.markdown('<div class="header">Location Details</div>', unsafe_allow_html=True)
st.markdown('<div class="location-container">', unsafe_allow_html=True)

# Columns for displaying image and details side by side
col1, col2 = st.columns([1, 3])
with col1:
    st.image("city.jpg", width=200)
with col2:
    st.markdown(f'<div class="details"><span style="color: #00796B; font-weight: bold;">Coordinates: Latitude: {lat}, Longitude: {lon}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sea-level">Sea Level: {elevation}</div>', unsafe_allow_html=True)

# Info box with Time to Sink and Time Remaining in columns
with st.container():
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    
    # Creating two columns inside the box for side-by-side layout
    info_col1, info_col2 = st.columns([1, 1])
    
    with info_col1:
        st.markdown('<div class="info-item">Submersion Year</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-value">{flood_year}</div>', unsafe_allow_html=True)
    
    with info_col2:
        st.markdown('<div class="info-item">Time Remaining</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-value">{years_until}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)