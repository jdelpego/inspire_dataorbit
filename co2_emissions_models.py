import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from singlestoredb import connect
import sys
from singlestoredb.exceptions import DatabaseError
from sklearn.metrics import r2_score

try:
    # Connect to SingleStore
    conn = connect(
        host='svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com',
        port=3333,
        user='jdelpego',
        password='fTVuFI26cwOVwAB7WVWybNqcBTrUP9KE',
        database='db_luke_503d4'
    )

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
    # Generate future years for prediction
    last_year = co2_df['Year'].max()
    future_years = np.arange(last_year + 1, last_year + 31)
    
    # Linear Model
    linear_model = np.polyfit(co2_df['Year'], co2_df['Emissions'], 1)
    linear_predictions = np.polyval(linear_model, future_years)
    linear_historical = np.polyval(linear_model, co2_df['Year'])
    linear_r2 = r2_score(co2_df['Emissions'], linear_historical)
    
    # Exponential Model
    log_emissions = np.log(co2_df['Emissions'])
    exp_model = np.polyfit(co2_df['Year'], log_emissions, 1)
    exp_predictions = np.exp(np.polyval(exp_model, future_years))
    exp_historical = np.exp(np.polyval(exp_model, co2_df['Year']))
    exp_r2 = r2_score(co2_df['Emissions'], exp_historical)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot historical data
    plt.scatter(co2_df['Year'], co2_df['Emissions'], 
                color='blue', alpha=0.5, label='Historical Data')
    
    # Plot linear model
    plt.plot(future_years, linear_predictions, 
             color='red', linestyle='--', label='Linear Projection')
    
    # Plot exponential model
    plt.plot(future_years, exp_predictions, 
             color='green', linestyle='--', label='Exponential Projection')
    
    plt.title('CO2 Emissions: Historical Data and Projections')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add model information
    info_text = (
        f'Linear Model:\n'
        f'  Growth rate: {linear_model[0]:.2f} units/year\n'
        f'  R² Score: {linear_r2:.3f}\n\n'
        f'Exponential Model:\n'
        f'  Growth rate: {(np.exp(exp_model[0]) - 1)*100:.2f}% per year\n'
        f'  R² Score: {exp_r2:.3f}'
    )
    plt.text(0.05, 0.95, info_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('co2_emissions_projections.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print detailed statistics
    print("\nLinear Model Statistics:")
    print(f"Annual increase: {linear_model[0]:.2f} units per year")
    print(f"R² Score: {linear_r2:.4f}")
    print(f"Projected emissions in {last_year + 30}: {linear_predictions[-1]:.2f}")
    
    print("\nExponential Model Statistics:")
    print(f"Annual growth rate: {(np.exp(exp_model[0]) - 1)*100:.2f}%")
    print(f"R² Score: {exp_r2:.4f}")
    print(f"Projected emissions in {last_year + 30}: {exp_predictions[-1]:.2f}")
    
    print("\nCurrent emissions (Year {last_year}): {co2_df['Emissions'].iloc[-1]:.2f}")
    print("\nPlot has been saved to 'co2_emissions_projections.png'")

except Exception as e:
    print(f"Error during analysis: {e}")
    print(f"Error details: {str(e)}") 