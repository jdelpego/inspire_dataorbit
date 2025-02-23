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

    # Query housing data
    housing_query = """
        SELECT *
        FROM zip_code_price_time_series
    """
    housing_df = pd.read_sql(housing_query, conn)

    # Query CPI data (assuming you have a CPI table)
    cpi_query = """
        SELECT year, cpi_value
        FROM cpi_data
    """
    try:
        cpi_df = pd.read_sql(cpi_query, conn)
        has_cpi_data = True
    except:
        print("Warning: CPI data not available, proceeding without inflation adjustment")
        has_cpi_data = False

except DatabaseError as e:
    print(f"Failed to connect to database or execute query: {e}")
    sys.exit(1)
finally:
    conn.close()

try:
    # Reshape housing data from wide to long format
    id_cols = ['RegionName', 'distance_to_coast_km', 'RegionID', 'SizeRank', 'RegionType', 'StateName', 'State', 'City', 'Metro', 'CountyName', 'Longitude', 'Latitude']
    
    # Print all columns to debug
    print("All columns:", housing_df.columns.tolist()[:10])
    
    # Filter for date columns
    date_cols = []
    for col in housing_df.columns:
        if col not in id_cols:
            try:
                # Use %y for 2-digit year format
                pd.to_datetime(col, format='%m/%d/%y')
                date_cols.append(col)
            except:
                print(f"Non-date column found: {col}")
                continue
    
    print(f"\nNumber of date columns found: {len(date_cols)}")
    print("Sample date columns:", date_cols[:5])
    
    housing_melted = housing_df.melt(id_vars=id_cols,
                                    value_vars=date_cols,
                                    var_name='date',
                                    value_name='price_index')

    # Print initial data shape
    print(f"\nInitial data shape: {housing_melted.shape}")
    print(f"Number of null values: {housing_melted['price_index'].isnull().sum()}")

    # Convert housing dates to years for matching with sea level data
    # Also use %y format here
    housing_melted['year'] = pd.to_datetime(housing_melted['date'], format='%m/%d/%y').dt.year
    
    # Remove null values and print information about the removal
    null_count_before = len(housing_melted)
    housing_melted = housing_melted.dropna(subset=['price_index'])
    null_count_after = len(housing_melted)
    
    print(f"\nRows removed due to null values: {null_count_before - null_count_after}")
    print(f"Percentage of data retained: {(null_count_after/null_count_before)*100:.2f}%")

    # Calculate average price index per year (excluding null values)
    yearly_prices = housing_melted.groupby('year')['price_index'].mean().reset_index()
    
    print("\nYears covered in housing data after cleaning:")
    print(sorted(yearly_prices['year'].unique()))

    if has_cpi_data:
        # Adjust for inflation
        # Normalize to 2022 dollars
        base_year = 2022
        base_year_cpi = cpi_df[cpi_df['year'] == base_year]['cpi_value'].iloc[0]
        
        # Merge CPI data with yearly prices
        yearly_prices = pd.merge(yearly_prices, cpi_df, on='year', how='left')
        
        # Calculate inflation-adjusted prices
        yearly_prices['price_index_adjusted'] = yearly_prices['price_index'] * (base_year_cpi / yearly_prices['cpi_value'])
        
        # Use adjusted prices for analysis
        price_column = 'price_index_adjusted'
        price_label = 'Inflation-Adjusted Price Index (2022 dollars)'
    else:
        price_column = 'price_index'
        price_label = 'Price Index (nominal)'

    # Rename the year column in sea_level_df to match housing data
    sea_level_df = sea_level_df.rename(columns={'Year': 'year'})

    # Merge sea level and housing price data
    merged_df = pd.merge(sea_level_df, yearly_prices, on='year', how='inner')
    
    print(f"\nFinal number of years after merging with sea level data: {len(merged_df)}")
    print("Years in final analysis:", sorted(merged_df['year'].unique()))

    # Calculate average price by coastal distance
    distance_prices = housing_melted.groupby('distance_to_coast_km')['price_index'].mean().reset_index()
    
    if has_cpi_data:
        # Adjust distance prices for inflation (using the most recent CPI value)
        distance_prices['price_index'] = distance_prices['price_index'] * (base_year_cpi / cpi_df['cpi_value'].iloc[-1])

    # Create correlation plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Sea Level vs Price
    sns.regplot(data=merged_df, x='sea_level', y=price_column, ax=ax1)
    ax1.set_title('Sea Level vs Housing Price Index')
    ax1.set_xlabel('Sea Level (mm)')
    ax1.set_ylabel(price_label)

    # Calculate correlation coefficient for sea level vs price
    correlation_sl = stats.pearsonr(merged_df['sea_level'], merged_df[price_column])
    ax1.text(0.05, 0.95, f'Correlation: {correlation_sl[0]:.3f}\np-value: {correlation_sl[1]:.3f}', 
             transform=ax1.transAxes, verticalalignment='top')

    # Plot 2: Coastal Distance vs Price
    sns.regplot(data=distance_prices, x='distance_to_coast_km', y='price_index', ax=ax2)
    ax2.set_title('Coastal Distance vs Housing Price Index')
    ax2.set_xlabel('Distance from Coast (km)')
    ax2.set_ylabel(price_label)

    # Calculate correlation coefficient for distance vs price
    correlation_dist = stats.pearsonr(distance_prices['distance_to_coast_km'], distance_prices['price_index'])
    ax2.text(0.05, 0.95, f'Correlation: {correlation_dist[0]:.3f}\np-value: {correlation_dist[1]:.3f}', 
             transform=ax2.transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.savefig('correlation_plots_inflation_adjusted.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print additional statistics
    print("\nCorrelation Analysis (Inflation Adjusted):")
    print(f"Sea Level vs Price Index:")
    print(f"- Correlation coefficient: {correlation_sl[0]:.3f}")
    print(f"- P-value: {correlation_sl[1]:.3f}")
    print(f"\nCoastal Distance vs Price Index:")
    print(f"- Correlation coefficient: {correlation_dist[0]:.3f}")
    print(f"- P-value: {correlation_dist[1]:.3f}")
    
    print("\nPlots have been saved to 'correlation_plots_inflation_adjusted.png'")

except Exception as e:
    print(f"Error during analysis: {e}") 