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
        SELECT year as Year, `mmfrom1993-2008average` as sea_level
        FROM 1880sealevel
    """
    sea_level_df = pd.read_sql(sea_level_query, conn)
    sea_level_df['sea_level'] = pd.to_numeric(sea_level_df['sea_level'], errors='coerce')
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
    
    print(f"\nFinal number of years after merging: {len(merged_df)}")
    print("Years in analysis:", sorted(merged_df['Year'].unique()))

    # Create correlation plot
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with regression line
    sns.regplot(data=merged_df, 
                x='Emissions', 
                y='sea_level',
                scatter_kws={'alpha':0.5})
    
    # Calculate correlation coefficient
    correlation = stats.pearsonr(merged_df['Emissions'], merged_df['sea_level'])
    
    plt.title('Global CO2 Emissions vs Sea Level Rise')
    plt.xlabel('Global CO2 Emissions')
    plt.ylabel('Sea Level (mm)')
    
    # Add correlation information to plot
    plt.text(0.05, 0.95, 
             f'Correlation: {correlation[0]:.3f}\np-value: {correlation[1]:.3f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top')

    # Add year labels to some points
    for idx, row in merged_df.iloc[::5].iterrows():  # Label every 5th point
        plt.annotate(str(int(row['Year'])), 
                    (row['Emissions'], row['sea_level']),
                    xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('co2_sea_level_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print("\nCorrelation Analysis:")
    print(f"CO2 Emissions vs Sea Level:")
    print(f"- Correlation coefficient: {correlation[0]:.3f}")
    print(f"- P-value: {correlation[1]:.3f}")
    
    # Calculate rate of change
    years = merged_df['Year']
    co2_rate = np.polyfit(years, merged_df['Emissions'], 1)[0]
    sea_level_rate = np.polyfit(years, merged_df['sea_level'], 1)[0]
    
    print("\nRates of Change:")
    print(f"CO2 Emissions: {co2_rate:.2f} units per year")
    print(f"Sea Level: {sea_level_rate:.2f} mm per year")
    
    print("\nPlot has been saved to 'co2_sea_level_correlation.png'")

except Exception as e:
    print(f"Error during analysis: {e}")
    print(f"Error details: {str(e)}") 