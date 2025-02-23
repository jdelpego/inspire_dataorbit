import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from singlestoredb import connect
import sys
from singlestoredb.exceptions import DatabaseError

def predict_flooding_year(altitude_mm, model, future_X, base_sea_level, start_year, max_years=200):
    """
    Predict the year when a specific altitude will be flooded.
    altitude_mm: altitude in millimeters above current sea level
    max_years: maximum number of years to predict into the future
    """
    # Extend prediction range if needed
    years_needed = np.arange(start_year + 1, start_year + max_years + 1)
    
    # Project CO2 emissions using exponential model
    log_emissions = np.log(merged_df['Emissions'])
    exp_model = np.polyfit(merged_df['Year'], log_emissions, 1)
    future_emissions = np.exp(np.polyval(exp_model, years_needed))
    
    # Create extended prediction data
    extended_X = pd.DataFrame({
        'Year': years_needed,
        'Emissions': future_emissions
    })
    
    # Make predictions
    future_levels = model.predict(extended_X)
    
    # Find when sea level reaches the altitude
    sea_level_rise = future_levels - base_sea_level
    flooding_levels = sea_level_rise >= altitude_mm
    
    if not any(flooding_levels):
        return None, None  # Location won't flood within max_years
    
    flooding_year = years_needed[flooding_levels][0]
    years_until_flooding = flooding_year - start_year
    
    return flooding_year, years_until_flooding

try:
    # Connect to SingleStore and get data
    conn = connect(
        host='svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com',
        port=3333,
        user='jdelpego',
        password='fTVuFI26cwOVwAB7WVWybNqcBTrUP9KE',
        database='db_luke_503d4'
    )

    # Query sea level and CO2 data
    sea_level_query = "SELECT Year, SmoothedGSML_GIA_sigremoved as sea_level FROM sealevel"
    co2_query = "SELECT Year, Emissions FROM GlobalCO2Emissions"
    
    sea_level_df = pd.read_sql(sea_level_query, conn)
    co2_df = pd.read_sql(co2_query, conn)

except DatabaseError as e:
    print(f"Failed to connect to database or execute query: {e}")
    sys.exit(1)
finally:
    conn.close()

try:
    # Merge and prepare data
    merged_df = pd.merge(sea_level_df, co2_df, on='Year', how='inner')
    
    # Create and train the model
    X = merged_df[['Year', 'Emissions']]
    y = merged_df['sea_level']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Get current (last known) sea level
    current_year = merged_df['Year'].max()
    current_sea_level = merged_df['sea_level'].iloc[-1]
    
    # Example altitudes to test (in mm above current sea level)
    test_altitudes = [1000, 2500, 5000, 10000]  # 1m, 2.5m, 5m, 10m
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot flooding predictions for different altitudes
    for altitude in test_altitudes:
        flooding_year, years_until = predict_flooding_year(
            altitude, model, X, current_sea_level, current_year
        )
        
        if flooding_year is not None:
            plt.axhline(y=altitude + current_sea_level, 
                       color='r', linestyle='--', alpha=0.3)
            plt.text(current_year, altitude + current_sea_level, 
                    f'{altitude/1000:.2f}m - Year {flooding_year}', 
                    verticalalignment='bottom')
    
    # Plot historical and predicted sea levels
    plt.scatter(merged_df['Year'], merged_df['sea_level'], 
                color='blue', alpha=0.5, label='Historical Data')
    
    # Calculate exponential model for CO2 emissions
    log_emissions = np.log(merged_df['Emissions'])
    exp_model = np.polyfit(merged_df['Year'], log_emissions, 1)
    
    # Plot future prediction line
    future_years = np.arange(current_year + 1, current_year + 101)
    future_X = pd.DataFrame({
        'Year': future_years,
        'Emissions': np.exp(np.polyval(exp_model, future_years))
    })
    future_predictions = model.predict(future_X)
    
    plt.plot(future_years, future_predictions, 
             color='red', linestyle='--', label='Predicted Sea Level')
    
    # Set y-axis limits to focus on relevant range
    max_altitude = max(test_altitudes) + current_sea_level
    min_altitude = min(merged_df['sea_level'])
    plt.ylim(min_altitude - 50, max_altitude + 50)  # Add some padding
    
    plt.title('Sea Level Rise and Flooding Predictions by Altitude')
    plt.xlabel('Year')
    plt.ylabel('Sea Level (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('flooding_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print predictions for various altitudes
    print("\nFlooding Predictions for Different Altitudes:")
    print("(Assuming exponential CO2 emissions growth)")
    print("\nCurrent sea level (Year {}): {:.2f} mm".format(
        current_year, current_sea_level))
    
    for altitude in test_altitudes:
        flooding_year, years_until = predict_flooding_year(
            altitude, model, X, current_sea_level, current_year
        )
        
        if flooding_year is not None:
            print(f"\nAltitude: {altitude/1000:.1f}m above current sea level:")
            print(f"- Predicted flooding year: {flooding_year}")
            print(f"- Years until flooding: {years_until}")
        else:
            print(f"\nAltitude: {altitude/1000:.1f}m above current sea level:")
            print("- Will not flood within 200 years")

    print("\nPlot has been saved to 'flooding_predictions.png'")

except Exception as e:
    print(f"Error during analysis: {e}")
    print(f"Error details: {str(e)}") 