import streamlit as st
import openai
import os
import tempfile
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
from folium import MacroElement
from jinja2 import Template
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.linear_model import Ridge
from dotenv import load_dotenv
import groq

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
        padding: 5px 0;
        z-index: 10000 !important; 
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);

    }
            
    .content {
        margin-top: 20px;
    }

    .navbar a {
        color: white !important;
        padding: 12px 20px !important;
        text-decoration: none !important; 
        font-size: 20px !important;
        font-weight: bold;
        display: inline-block !important;
        transition: all 0.3s ease;
        border-radius: 3px
    }
    .navbar a:hover {
        background-color: #99E1D9;
        color: black !important;
        transition: 0.3s ease-in !important;
        border-radius: 2px !important; /* Less rounding on hover */
        padding: 6px 10px !important; 
    }
    .content {
        margin-top: 70px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="navbar">
        <a target="_self" href="?page=Home">🏠 Home</a>
        <a target="_self" href="?page=resources">📚 Resources</a>
        <a target="_self" href="?page=chatbot">🤖 Chat Bot</a>
    </div>
""", unsafe_allow_html=True)

class ClearMarkerOnClick(MacroElement):
    _template = Template(u"""
        {% macro script(this, kwargs) %}
        // Create a layer for the click marker and add it to the map
        var markerLayer = L.layerGroup().addTo({{ this._parent.get_name() }});
        // Listen for clicks on the map
        {{ this._parent.get_name() }}.on('click', function(e) {
            // Remove any existing marker
            markerLayer.clearLayers();
            // Add a new marker at the click location
            L.marker(e.latlng).addTo(markerLayer);
        });
        {% endmacro %}
    """)

load_dotenv()

google_maps_api_key = st.secrets["api_key"]["google_maps_api_key"]
groqapi_key = st.secrets["api_key"]["groqapi_key"]
groq_client = groq.Groq(api_key=groqapi_key)

# Function to get chatbot response from Groq API
def get_groq_response(user_input):
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are AI assistant helping users understand the impact of rising sea levels. Please only provide information related to this topic."}, 
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content if response.choices else "No response."
        
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
    #future_levels = model.predict(poly.transform(extended_X))
    #future_levels = model.predict(poly.transform(extended_X)) * 1.2

    extended_X['year^2'] = extended_X['year']**2
    # Keep Emissions as-is (no Emissions^2, no interaction).
    future_levels = model.predict(extended_X[['year', 'year^2', 'Emissions']])

    # Find when sea level reaches the altitude
    sea_level_rise = future_levels - base_sea_level
    flooding_levels = sea_level_rise >= altitude_mm*1000

    if not any(flooding_levels):
        return None, None  # Location won't flood within max_years

    flooding_year = years_needed[flooding_levels][0]
    years_until_flooding = flooding_year - start_year

    return flooding_year, years_until_flooding


def is_land(lat, lon):
    elevation = get_elevation(lat, lon)
    return elevation > 0

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
    #poly = PolynomialFeatures(degree=2, include_bias=False)
    #X_poly = poly.fit_transform(X)

    # Train the quadratic model using LinearRegression
    #model = LinearRegression()
    model = Ridge(alpha=40.0)  # alpha > 0 means more regularization
    
    X['year^2'] = X['year'] ** 2
    X_poly = X[['year', 'year^2', 'Emissions']]

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
        else:
            raise Exception("Status not OK getting elevation" + str(result))
    else:
        raise Exception("Error getting elevation 200 response")
        return None

@st.cache_data
def create_map(lat, lon, zoom=5):
    m = folium.Map(location=[lat, lon], zoom_start=zoom)
    #folium.Marker(location=[lat, lon], popup="Selected Location").add_to(m)
    m.add_child(ClearMarkerOnClick())

    #m.add_child(folium.ClickForMarker())

    #folium.Marker(location=[lat, lon]).add_to(m)
    return m

# Get current page
query_params = st.query_params
tab = query_params.get("page")

if tab == "Home" or tab == None:
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown("""
        <h1 style='text-align: center; margin-top: 50px; color: #60a7f7;'>🌊 When Will Your House Sink?</h1>
        <p style='text-align: center;'>A Rising Sea Level Prediction Model</p>
    """, unsafe_allow_html=True)

    lat, lon = None, None
    elevation = None
    flooding_year, years_until = None, None

    st.markdown('<div class="map-container">', unsafe_allow_html=True)

    m = create_map(34.4356 if not lat else lat, -119.8276 if not lon else lon)

    map_result = st_folium(m, width="100%", height=500)

    st.markdown('</div>', unsafe_allow_html=True)
    if map_result and "last_clicked" in map_result:
        clicked_location = map_result["last_clicked"]

        if clicked_location and "lat" in clicked_location and "lng" in clicked_location:
            lat = clicked_location["lat"]
            lon = clicked_location["lng"]

            if is_land(lat, lon):
                elevation = get_elevation(lat, lon)
                flooding_year, years_until = predict_flooding_year(
                    elevation, model, X, current_sea_level, current_year
                )
            else:
                st.warning("🌊 Please select a location on land")
                flooding_year, years_until = None, None

            #m = create_map(lat, lon)
            #map_result = st_folium(m, width="100%", height=500)

   # st.markdown(f"""
  #  <div style="text-align: center; font-size: 18px;" {'hidden' if not (lat and lon and elevation) else ''}>
  #      <p><strong>📍Latitude:</strong> {'N/A' if not lat else lat}</p>
  #      <p><strong>📍Longitude:</strong> {'N/A' if not lon else lon}</p>
  #      <p><strong>📏Altitude:</strong> {'N/A' if not elevation else (f'{elevation:.2f} meters')}</p>
  #      <p><strong> flooding year:</strong> {'N/A' if not flooding_year else flooding_year}</p>
   #     <p><strong> years until:</strong> {'N/A' if not years_until else years_until}</p>
    #</div>
#""", unsafe_allow_html=True)

    # Ensure values are valid before sending them to the next page
    if flooding_year and years_until:
        predict_url = f"https://inspiredataorbit-pskuarfzyis9iwg26brttt.streamlit.app?lat={lat}&lon={lon}&elevation={elevation:.2f}&flooding_year={flooding_year}&years_until={years_until}"
    else:
        predict_url = None

    st.markdown("""
        <style>
        .predict-button {
            display: flex;
            justify-content: center;
            margin-top: 2px;
        }
        .predict-button a {
            background-color: #007BFF;
            color: white;
            padding: 12px 24px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            text-decoration: none;
            transition: 0.3s ease-in-out;
        }
        .predict-button a:hover {
            background-color: #0056b3;
        }
        </style>
    """, unsafe_allow_html=True)

    if predict_url:
        st.markdown(f"""
            <div class="predict-button" {'hidden' if not (lat and lon and elevation) else ''}>
                <a href="{predict_url}" target="">🔮 Predict</a>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please select a valid location to enable predictions.")

    st.markdown("""
        <p style='text-align: center; margin-top: 40px;'>Please click on the map.</p>
        <p style='text-align: center; color: #007BFF;'>This model predicts when a place will sink due to rising sea levels.</p>
        <div style='text-align: center; font-size: 14px; color: gray;'>
            <p>©Inspire, Inc.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="content">', unsafe_allow_html=True)
elif tab == "resources":
    st.title("Help Mitigate Rising Sea Levels!")

    # Add the interactive graph
    st.write(
        """
        ## Interactive Sea Level Rise Simulator
        This graph shows the rise in sea levels over time. 
        You can hover over a point on the line to see the exact sea level at a given year.
        """
    )

    # Path to the CSV file (make sure to use the correct path for your dataset)
    csv_file_path = 'seaData.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the necessary columns exist in the CSV file
    if 'Time' in df.columns and 'GMSL' in df.columns:
        # Convert 'Time' to datetime and extract the year
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df['Year'] = df['Time'].dt.year

        # Ensure 'GMSL' is numeric
        df['GMSL'] = pd.to_numeric(df['GMSL'], errors='coerce')

        # Drop rows with missing values
        df = df.dropna(subset=['Year', 'GMSL'])

        years = df['Year']
        sea_level = df['GMSL']

        # Create a ColumnDataSource for Bokeh
        source = ColumnDataSource(data=dict(years=years, sea_level=sea_level))

        # Create a Bokeh figure
        p = figure(title="Projected Sea Level Rise", x_axis_label='Year', y_axis_label='Sea Level (meters)', height=400, width=700)

        # Add the line to the figure
        p.line('years', 'sea_level', source=source, line_width=2, color="blue", legend_label="Sea Level Rise")

        # Add a circle (point) that we can drag on the line
        p.circle('years', 'sea_level', source=source, size=8, color="red", alpha=0.6)

        # Add a hover tool to show the year and sea level at that point
        hover_tool = HoverTool()
        hover_tool.tooltips = [("Year", "@years"), ("Sea Level", "@sea_level")]
        p.add_tools(hover_tool)

        # Display the plot in Streamlit
        st.bokeh_chart(p)
    else:
        st.error("The CSV file must contain 'Time' and 'GMSL' columns.")

    # Now display other resources information below the graph
    st.write(
        """
        ## Reduce Climate Change Through Carbon Emissions
        The primary driver of sea level rise is climate change, which causes thermal expansion of water and melting of polar ice caps.
        To curb sea level rise, we need to reduce global greenhouse gas emissions.

        **Some ways to do this at home:**
        - Limit your driving time
        - Switch to an electric vehicle
        - Upgrade to energy-efficient appliances
        - Shift to renewable energy sources

        ## Protecting and Restoring Natural Ecosystems
        Wetlands and salt marshes act as natural barriers, absorbing floodwaters and reducing storm surges.
        Protecting and restoring these ecosystems can help buffer coastal areas from rising seas.

        Additionally, planting trees and restoring forests can help absorb carbon dioxide and reduce the greenhouse effect, slowing climate change.

        ### Some Further Resources:
        - [National Oceanic and Atmospheric Administration (NOAA)](https://www.noaa.gov/): Provides data, resources, and research on sea level rise and its impacts.
        - [Intergovernmental Panel on Climate Change (IPCC)](https://www.ipcc.ch/): Publishes regular reports on climate science, including the expected impacts of sea level rise.
        - [The Nature Conservancy](https://www.nature.org/): Focuses on protecting natural ecosystems like mangroves and wetlands to mitigate sea level rise.
        - [UN Environment Programme](https://www.unep.org/): Offers guidelines and frameworks for managing climate change and its effects, including sea level rise.
        """
    )

elif tab == "chatbot":
    st.title("Sea Level Rise Chatbot")

    st.markdown("<h3 style='text-align: center;'>Ask about rising sea levels, its causes, and impacts!</h3>", unsafe_allow_html=True)

    user_input = st.text_input("Enter your question about sea level rise:")

    if st.button("Get Response"):
        if user_input:
            with st.spinner("Fetching response..."):
                response = get_groq_response(user_input)
            
            st.subheader("Chatbot's Response:")
            st.write(response)
        else:
            st.warning("Please enter a question.")