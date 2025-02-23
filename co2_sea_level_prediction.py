import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from singlestoredb import connect
import sys
from singlestoredb.exceptions import DatabaseError
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error


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
    
    print("\nSea Level Data Range:")
    print(f"From {sea_level_df['year'].min()} to {sea_level_df['year'].max()}")
    print(f"Number of sea level measurements: {len(sea_level_df)}")

    # Query CO2 emissions data
    co2_query = """
        SELECT Year as year, Emissions
        FROM GlobalCO2Emissions
        ORDER BY Year
    """
    co2_df = pd.read_sql(co2_query, conn)
    
    print("\nCO2 Data Range:")
    print(f"From {co2_df['year'].min()} to {co2_df['year'].max()}")
    print(f"Number of CO2 measurements: {len(co2_df)}")

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
    
    print(f"Years with sea level data: {len(sea_level_years)}")
    print(f"Years with CO2 data: {len(co2_years)}")
    print(f"Years with both measurements: {len(overlap_years)}")
    
    if len(sea_level_years - overlap_years) > 0:
        print("\nSea level years without CO2 data:", sorted(sea_level_years - overlap_years))
    if len(co2_years - overlap_years) > 0:
        print("\nCO2 years without sea level data:", sorted(co2_years - overlap_years))

    # Merge sea level and CO2 data
    merged_df = pd.merge(sea_level_df, co2_df, on='year', how='inner')
    print(f"\nFinal number of years after merge: {len(merged_df)}")
    
    # Prepare features and target.
    # We'll scale down CO2 values by multiplying by 0.3
    merged_df['Emissions'] = merged_df['Emissions']
    X = merged_df[['year', 'Emissions']]
    y = merged_df['sea_level']
    
    # Create quadratic features (degree=2) using PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Train the quadratic model using LinearRegression
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Evaluate model performance on the training data
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Generate future years for prediction
    last_year = merged_df['year'].max()
    future_years = np.arange(last_year + 1, last_year + 75)
    
    weights =  co2_df['year'] - np.min(co2_df['year']) + 1

    # Project future CO2 emissions using an exponential model on all available CO2 data.
    # (Here we use the full CO2 dataset, without scaling, then scale the predictions as before.)
    log_emissions = np.log(co2_df['Emissions'])  # Note: co2_df's Emissions are still unscaled here.
    years_for_exp = co2_df['year']
    exp_model = np.polyfit(years_for_exp, log_emissions, 1, w=weights)
    future_emissions = np.exp(np.polyval(exp_model, future_years))
    # Scale future emissions by the same factor (0.3) used in training:
    future_emissions_scaled = future_emissions
    
    # Create future feature DataFrame for predictions
    future_X = pd.DataFrame({
        'year': future_years,
        'Emissions': future_emissions_scaled
    })
    future_X_poly = poly.transform(future_X)
    future_predictions = model.predict(future_X_poly)    



    # -------------------------------\n    # Visualization\n    # -------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Get current (last known) sea level
    current_year = merged_df['year'].max()
    current_sea_level = merged_df['sea_level'].iloc[-1]
    
    # Example altitudes to test (in mm above current sea level)
    test_altitudes = [1000, 2500, 5000, 10000]  # 1m, 2.5m, 5m, 10m
    
    # Plot flooding predictions for different altitudes
    for altitude in test_altitudes:
        flooding_year, years_until = predict_flooding_year(
            altitude, model, X, current_sea_level, current_year
        )
        
        if flooding_year is not None:
            ax2.axhline(y=altitude + current_sea_level, 
                       color='r', linestyle='--', alpha=0.3)
            ax2.text(current_year, altitude + current_sea_level, 
                    f'{altitude/1000:.2f}m - Year {flooding_year}', 
                    verticalalignment='bottom')

    # Plot 1: CO2 Emissions Projection (Graphing the emissions model)
    # For historical data, rescale the emissions back to original by dividing by 0.1.
    ax1.scatter(co2_df['year'], co2_df['Emissions'], 
                color='green', alpha=0.5, label='Historical Emissions')
    # For future emissions, plot the unscaled predicted values.
    ax1.plot(future_years, future_emissions, 
             color='darkgreen', linestyle='--', label='Projected Emissions (Exponential)')
    ax1.set_title('CO2 Emissions: Historical and Projected')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO2 Emissions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add emissions model info
    emissions_growth_rate = (np.exp(exp_model[0]) - 1) * 100
    ax1.text(0.05, 0.95,
             f'Exponential Growth Rate: {emissions_growth_rate:.2f}% per year',
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Add data coverage info
    coverage_text = (
        f'Data Coverage:\n'
        f'Sea Level: {min(sea_level_years)}-{max(sea_level_years)}\n'
        f'CO2: {min(co2_years)}-{max(co2_years)}'
    )
    ax1.text(0.05, 0.85,
             coverage_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 2: Sea Level Predictions (Quadratic Model with Emissions and Year)
    ax2.scatter(sea_level_df['year'], sea_level_df['sea_level'], 
                color='blue', alpha=0.5, label='Historical Sea Level')
    ax2.plot(future_years, future_predictions, 
             color='red', linestyle='--', label='Predicted Sea Level')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3, label='1993-2008 Average')
    
    # Confidence intervals using training error as a proxy:
    std_err = np.sqrt(np.sum((y - y_pred)**2) / (len(y) - X_poly.shape[1]))
    ax2.fill_between(future_years,
                     future_predictions - 2*std_err,
                     future_predictions + 2*std_err,
                     color='red', alpha=0.1,
                     label='95% Confidence Interval')
    
    ax2.set_title('Sea Level Rise Prediction (Relative to 1993-2008 Average)\n(Quadratic Model with Year and Emissions)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Sea Level Change (mm relative to 1993-2008 average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model performance metrics
    metrics_text = (
        f'R² Score: {r2:.3f}\n'
        f'(Quadratic Model with Year and Emissions)'
    )
    ax2.text(0.05, 0.95, 
             metrics_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('co2_sea_level_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print predictions with clear reference to baseline
    print("\nSea Level Rise Predictions (relative to 1993-2008 average):")
    print(f"Current sea level (Year {last_year}): {merged_df['sea_level'].iloc[-1]:.2f} mm")
    print(f"Predicted sea level in {last_year + 5}: {future_predictions[-1]:.2f} mm")
    print(f"Predicted rise over 5 years: {future_predictions[-1] - merged_df['sea_level'].iloc[-1]:.2f} mm")
    
except Exception as e:
    print(f"Error during analysis: {e}")
    print(f"Error details: {str(e)}")