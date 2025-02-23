import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from singlestoredb import connect
import sys
from singlestoredb.exceptions import DatabaseError
import seaborn as sns
from scipy import stats

try:
    # Connect to SingleStore
    conn = connect(
        host='svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com',
        port=3333,
        user='jdelpego',
        password='fTVuFI26cwOVwAB7WVWybNqcBTrUP9KE',
        database='db_luke_503d4'
    )

    # Query sea level data
    sea_level_query = """
        SELECT Year, SmoothedGSML_GIA_sigremoved as sea_level
        FROM sealevel
    """
    sea_level_df = pd.read_sql(sea_level_query, conn)

    # Query CO2 emissions data
    co2_query = """
        SELECT Year, Emissions
        FROM GlobalCO2Emissions
    """
    co2_df = pd.read_sql(co2_query, conn)

except DatabaseError as e:
    print(f"Failed to connect to database or execute query: {e}")
    sys.exit(1)
finally:
    conn.close()

try:
    # Merge sea level and CO2 data
    merged_df = pd.merge(sea_level_df, co2_df, on='Year', how='inner')
    
    # Create and train the model
    X = merged_df[['Year', 'Emissions']]
    y = merged_df['sea_level']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future years for prediction
    last_year = merged_df['Year'].max()
    future_years = np.arange(last_year + 1, last_year + 31)
    
    # Project CO2 emissions using exponential model
    log_emissions = np.log(merged_df['Emissions'])
    exp_model = np.polyfit(merged_df['Year'], log_emissions, 1)
    future_emissions = np.exp(np.polyval(exp_model, future_years))
    
    # Create prediction data
    future_X = pd.DataFrame({
        'Year': future_years,
        'Emissions': future_emissions
    })
    
    # Make predictions
    future_predictions = model.predict(future_X)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot 1: CO2 Emissions Projection
    ax1.scatter(merged_df['Year'], merged_df['Emissions'], 
                color='green', alpha=0.5, label='Historical Emissions')
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
    
    # Plot 2: Sea Level Predictions
    ax2.scatter(merged_df['Year'], merged_df['sea_level'], 
                color='blue', alpha=0.5, label='Historical Sea Level')
    ax2.plot(future_years, future_predictions, 
             color='red', linestyle='--', label='Predicted Sea Level')
    
    # Add confidence intervals
    y_pred_hist = model.predict(X)
    std_err = np.sqrt(np.sum((y - y_pred_hist)**2) / (len(y) - 2))
    
    ax2.fill_between(future_years,
                     future_predictions - 2*std_err,
                     future_predictions + 2*std_err,
                     color='red', alpha=0.1,
                     label='95% Confidence Interval')
    
    ax2.set_title('Sea Level Rise Prediction (Based on Exponential CO2 Emissions)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Sea Level (mm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add model performance metrics
    r2_score = model.score(X, y)
    ax2.text(0.05, 0.95, 
             f'R² Score: {r2_score:.3f}\nCO2 Coefficient: {model.coef_[1]:.3f}\nYear Coefficient: {model.coef_[0]:.3f}', 
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('sea_level_predictions_exp.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print predictions and growth rates
    print("\nCO2 Emissions Projections:")
    print(f"Current emissions (Year {last_year}): {merged_df['Emissions'].iloc[-1]:.2f}")
    print(f"Projected emissions in {last_year + 30}: {future_emissions[-1]:.2f}")
    print(f"Emissions growth rate: {emissions_growth_rate:.2f}% per year")

    print("\nSea Level Rise Predictions:")
    print(f"Current sea level (Year {last_year}): {merged_df['sea_level'].iloc[-1]:.2f} mm")
    print(f"Predicted sea level in {last_year + 30}: {future_predictions[-1]:.2f} mm")
    print(f"Predicted rise over 30 years: {future_predictions[-1] - merged_df['sea_level'].iloc[-1]:.2f} mm")
    
    # Print model coefficients
    print("\nModel Coefficients:")
    print(f"Year effect: {model.coef_[0]:.4f} mm/year")
    print(f"CO2 effect: {model.coef_[1]:.4f} mm/unit of emissions")
    print(f"R² Score: {r2_score:.4f}")
    
    print("\nPlot has been saved to 'sea_level_predictions_exp.png'")

except Exception as e:
    print(f"Error during analysis: {e}")
    print(f"Error details: {str(e)}") 