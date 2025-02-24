from flask import Flask, render_template, jsonify, request, redirect
import requests
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from dotenv import load_dotenv
from singlestoredb import connect, DatabaseError
import sys
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.resources import INLINE
from bokeh.models import ColumnDataSource, HoverTool
import groq

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get environment variables
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '3333')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not all([GOOGLE_MAPS_API_KEY, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, GROQ_API_KEY]):
    raise ValueError("Missing required environment variables")

GOOGLE_MAPS_API_KEY = GOOGLE_MAPS_API_KEY.strip('"\'')

print(f"API Key loaded: {GOOGLE_MAPS_API_KEY[:10]}...")

def get_elevation(lat, lng):
    """Get elevation data from Google Maps API"""
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json()
        if result["status"] == "OK":
            return result["results"][0]["elevation"]
        else:
            raise Exception(f"Status not OK getting elevation: {result}")
    else:
        raise Exception("Error getting elevation: non-200 response")

CUTOFF_YEAR = 2065  # Year when sea level transitions to constant linear rise

def calculate_cutoff_slope(model, cutoff_year, last_emission, emissions_growth_rate, last_year):
    """Calculate the slope at the cutoff year for linear extrapolation"""
    years = np.array([cutoff_year - 1, cutoff_year, cutoff_year + 1])
    
    # Project emissions up to cutoff (exponential growth)
    emissions = last_emission * np.exp(emissions_growth_rate * (years - last_year))
    
    # Predict sea levels
    X_cutoff = pd.DataFrame({'year': years, 'Emissions': emissions})
    predictions = model.predict(X_cutoff)
    
    # Calculate slope (mm/year) at cutoff
    slope = (predictions[1] - predictions[0])  # Rate just before cutoff
    return slope

def predict_flooding_year(altitude_mm, model, future_X, base_sea_level, start_year, start_emission, max_years=1000):
    """
    Predict the year when a specific altitude will be flooded.
    altitude_mm: altitude in millimeters above current sea level
    max_years: maximum number of years to predict into the future
    """
    # Extend prediction range
    years_needed = np.arange(start_year + 1, start_year + max_years + 1)
    
    nweights = future_X['year'] - np.min(future_X['year']) + 1
    
    # Project CO2 emissions using exponential model
    log_emissions = np.log(future_X['Emissions'])
    exp_model = np.polyfit(future_X['year'], log_emissions, 1, w=nweights)
    
    growth_rate_reduction = 1  # Original growth rate
    adjusted_slope = exp_model[0] * growth_rate_reduction
    
    log_emissions_future = np.log(start_emission) + adjusted_slope * (years_needed - start_year)
    future_emissions = np.exp(log_emissions_future)
    
    # Create extended prediction data
    extended_X = pd.DataFrame({
        'year': years_needed,
        'Emissions': future_emissions
    })
    
    # Calculate slope at cutoff for linear extrapolation
    slope_at_cutoff = calculate_cutoff_slope(model, CUTOFF_YEAR, start_emission, adjusted_slope, start_year)
    
    # Split predictions into pre-cutoff and post-cutoff
    pre_cutoff_mask = extended_X['year'] <= CUTOFF_YEAR
    post_cutoff_mask = extended_X['year'] > CUTOFF_YEAR
    
    # Predict normally up to cutoff
    future_predictions_pre = model.predict(extended_X[pre_cutoff_mask])
    
    # For post-cutoff, apply constant linear growth
    cutoff_sea_level = future_predictions_pre[-1] if len(future_predictions_pre) > 0 else base_sea_level
    post_cutoff_years = years_needed[post_cutoff_mask]
    future_predictions_post = cutoff_sea_level + slope_at_cutoff * (post_cutoff_years - CUTOFF_YEAR)
    
    # Combine predictions
    future_predictions = np.concatenate([future_predictions_pre, future_predictions_post])
    
    # Find when sea level reaches the altitude
    sea_level_rise = future_predictions - base_sea_level
    flooding_levels = sea_level_rise >= altitude_mm
    
    if not any(flooding_levels):
        return None, None
    
    flooding_year = int(years_needed[flooding_levels][0])
    years_until_flooding = int((flooding_year - start_year))  # Multiply by 2.5x to match Streamlit
    
    return flooding_year, years_until_flooding

# Initialize model and data
try:
    # Connect to SingleStore
    conn = connect(
        host=DB_HOST,
        port=int(DB_PORT),
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

    # Query sea level and CO2 data
    sea_level_df = pd.read_sql("""
        SELECT year, `mmfrom1993-2008average` as sea_level
        FROM 1880sealevel
        ORDER BY year
    """, conn)
    
    co2_df = pd.read_sql("""
        SELECT Year as year, Emissions
        FROM GlobalCO2Emissions
        ORDER BY Year
    """, conn)
    
    # Convert columns to numeric types
    sea_level_df['sea_level'] = pd.to_numeric(sea_level_df['sea_level'], errors='coerce')
    co2_df['Emissions'] = pd.to_numeric(co2_df['Emissions'], errors='coerce')

    # Merge data and prepare model
    merged_df = pd.merge(sea_level_df, co2_df, on='year', how='inner')
    X = merged_df[['year', 'Emissions']]
    y = merged_df['sea_level']
    
    # Train model using LinearRegression (matching Streamlit)
    model = LinearRegression()
    model.fit(X, y)
    
    # Store current state
    START_YEAR = merged_df['year'].max()
    BASE_SEA_LEVEL = merged_df['sea_level'].iloc[-1]
    START_EMISSION = merged_df['Emissions'].iloc[-1]
    future_X = X

except Exception as e:
    print(f"Error initializing model: {e}")
    sys.exit(1)
finally:
    conn.close()

def is_land(lat, lng):
    """Check if the location is on land by checking elevation"""
    elevation = get_elevation(lat, lng)
    return elevation > 0

@app.route('/')
def index():
    return render_template('index.html', api_key=GOOGLE_MAPS_API_KEY)

@app.route('/get_elevation', methods=['POST'])
def get_elevation_route():
    try:
        data = request.json
        lat = data.get('lat')
        lng = data.get('lng')
        
        if not lat or not lng:
            return jsonify({"error": "Missing coordinates"}), 400
            
        if not is_land(lat, lng):
            return jsonify({"error": "🌊 Please select a location on land"}), 400
        
        elevation = get_elevation(lat, lng)
        # Convert elevation to mm for prediction
        flooding_year, years_until_flooding = predict_flooding_year(
            elevation * 1000,  # Convert elevation from meters to mm
            model,
            future_X,
            BASE_SEA_LEVEL,
            START_YEAR,
            START_EMISSION
        )
        
        return jsonify({
            "elevation": float(elevation),
            "flooding_year": int(flooding_year) if flooding_year is not None else 0,
            "years_until_flooding": int(years_until_flooding) if years_until_flooding is not None else 0
        })
            
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict')
def predict():
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        elevation = float(request.args.get('elevation'))
        
        if not is_land(lat, lng):
            return redirect('/')
        
        # Convert elevation to mm for prediction
        flooding_year, years_until_flooding = predict_flooding_year(
            elevation * 1000,  # Convert elevation from meters to mm
            model,
            future_X,
            BASE_SEA_LEVEL,
            START_YEAR,
            START_EMISSION
        )
        
        return render_template('predict.html',
                             lat=lat,
                             lng=lng,
                             elevation=elevation,
                             flooding_year=flooding_year if flooding_year is not None else 0,
                             years_until_flooding=years_until_flooding if years_until_flooding is not None else 0)
    except Exception as e:
        print(f"Error processing prediction: {str(e)}")
        return redirect('/')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/chart')
def chart():
    try:
        # Read the CSV file
        df = pd.read_csv('seaData.csv')
        
        # Convert 'Time' to datetime and extract the year
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df['Year'] = df['Time'].dt.year
        
        # Ensure 'GMSL' is numeric
        df['GMSL'] = pd.to_numeric(df['GMSL'], errors='coerce')
        
        # Drop rows with missing values
        df = df.dropna(subset=['Year', 'GMSL'])
        
        # Create Bokeh figure with a dark theme
        p = figure(
            title="Projected Sea Level Rise",
            x_axis_label='Year',
            y_axis_label='Sea Level (meters)',
            height=600,
            width=1000,
            background_fill_color='#000033',
            border_fill_alpha=0,
            outline_line_color=None,
            toolbar_location='right'
        )
        
        # Style the plot
        p.title.text_color = '#ffffff'
        p.title.text_font_size = '24px'
        p.title.text_font = 'Inter'
        p.title.text_font_style = 'normal'
        p.title.align = 'center'
        
        p.xaxis.axis_label_text_color = '#ffffff'
        p.yaxis.axis_label_text_color = '#ffffff'
        p.xaxis.major_label_text_color = '#ffffff'
        p.yaxis.major_label_text_color = '#ffffff'
        p.xaxis.axis_label_text_font_size = '16px'
        p.yaxis.axis_label_text_font_size = '16px'
        p.xaxis.major_label_text_font_size = '12px'
        p.yaxis.major_label_text_font_size = '12px'
        
        # Style the grid
        p.grid.grid_line_color = '#333333'
        p.grid.grid_line_alpha = 0.3
        
        # Create a ColumnDataSource
        source = ColumnDataSource(df)
        
        # Add scatter points
        p.scatter('Year', 'GMSL', source=source, size=8, color='#ff4444', alpha=0.6)
        
        # Add the line to the figure
        p.line('Year', 'GMSL', source=source, line_width=2, color="#4a90e2", line_alpha=0.8)
        
        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ('Year', '@Year'),
                ('Sea Level', '@GMSL{0.0} mm')
            ],
            mode='vline'
        )
        p.add_tools(hover)
        
        # Get the components
        script, div = components(p)
        
        # Include Bokeh resources
        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()
        
        return render_template(
            'chart.html',
            chart_html=div + script,
            js_resources=js_resources,
            css_resources=css_resources
        )
        
    except Exception as e:
        print(f"Error generating chart: {str(e)}")
        return redirect('/')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat_message():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({"error": "No message provided"}), 400

        print(f"Making API call to Groq with key: {GROQ_API_KEY[:10]}...")
        
        # Initialize Groq client with API key
        client = groq.Groq(api_key=GROQ_API_KEY)
        
        print("Sending message to Groq API...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specializing in sea level rise and climate change. Provide clear, concise responses."
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=300
        )
        
        print("Response received from Groq")
        if not chat_completion.choices:
            print("No choices in response")
            return jsonify({"error": "No response from AI service"}), 500
            
        response_text = chat_completion.choices[0].message.content
        if not response_text:
            print("Empty response text")
            return jsonify({"error": "Empty response from AI service"}), 500
            
        print("Successfully received response")
        return jsonify({"response": response_text})

    except Exception as e:
        print(f"Error in chat_message: {str(e)}")
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
