from flask import Flask, render_template, jsonify, request, redirect
import requests
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
from singlestoredb import connect, DatabaseError
import sys

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get API key from environment variable
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
if not GOOGLE_MAPS_API_KEY:
    raise ValueError("Missing Google Maps API Key")
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

# Initialize model and data
try:
    # Connect to SingleStore
    conn = connect(
        host='svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com',
        port=3333,
        user='jdelpego',
        password='fTVuFI26cwOVwAB7WVWybNqcBTrUP9KE',
        database='db_luke_503d4'
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
    
    # Create and train model
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Store current state
    START_YEAR = merged_df['year'].max()
    BASE_SEA_LEVEL = merged_df['sea_level'].iloc[-1]
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

def predict_flooding_year(altitude_mm, model=model, future_X=future_X, base_sea_level=BASE_SEA_LEVEL, start_year=START_YEAR, max_years=500):
    """
    Predict the year when a specific altitude will be flooded.
    altitude_mm: altitude in millimeters above current sea level
    """
    try:
        years_needed = np.arange(start_year + 1, start_year + max_years + 1)
        nweights = future_X['year'] - np.min(future_X['year']) + 1
        
        # Project CO2 emissions using exponential model
        log_emissions = np.log(future_X['Emissions'])
        exp_model = np.polyfit(future_X['year'], log_emissions, 1, w=nweights)
        future_emissions = np.exp(np.polyval(exp_model, years_needed))
        
        # Create prediction data
        extended_X = pd.DataFrame({
            'year': years_needed,
            'Emissions': future_emissions
        })
        
        # Make predictions
        future_levels = model.predict(poly.transform(extended_X))
        sea_level_rise = future_levels - base_sea_level
        flooding_levels = sea_level_rise >= altitude_mm
        
        if not any(flooding_levels):
            return None, None
        
        flooding_year = int(years_needed[flooding_levels][0])
        years_until_flooding = int(flooding_year - start_year)
        
        return flooding_year, years_until_flooding
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, None

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
        flooding_year, years_until_flooding = predict_flooding_year(elevation * 1000)  # Convert to mm
        
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
        
        flooding_year, years_until_flooding = predict_flooding_year(elevation * 1000)  # Convert to mm
        
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

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except OSError as e:
        print(f"Error: Port 5001 is in use. Please stop any other running Flask instances. Error: {e}")
        exit(1)
