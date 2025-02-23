import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from singlestoredb import connect
import sys
from singlestoredb.exceptions import DatabaseError
from sklearn.exceptions import NotFittedError

try:
    # Connect to SingleStore
    conn = connect(
        host='svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com',
        port=3333,
        user='jdelpego',
        password='fTVuFI26cwOVwAB7WVWybNqcBTrUP9KE',
        database='db_luke_503d4'
    )
except DatabaseError as e:
    print(f"Failed to connect to database: {e}")
    sys.exit(1)

try:
    # Load the data from SingleStore
    # Adjust this query based on your actual column names
    query = """
        SELECT *
        FROM housing_data
    """
    df = pd.read_sql(query, conn)

    # Check if we got any data
    if df.empty:
        raise ValueError("No data retrieved from the database")

    # Reshape data from wide to long format
    # Assuming 'zipcode' and 'coastal_distance' are non-date columns
    id_cols = ['zipcode', 'coastal_distance']
    date_cols = [col for col in df.columns if col not in id_cols]
    
    # Melt the dataframe to convert date columns to rows
    df_melted = df.melt(id_vars=id_cols,
                        value_vars=date_cols,
                        var_name='date',
                        value_name='price_index')

    # Convert date strings to datetime
    df_melted['date'] = pd.to_datetime(df_melted['date'])
    df_melted['months_since_start'] = (df_melted['date'] - df_melted['date'].min()).dt.total_seconds() / (30 * 24 * 60 * 60)

    # Check for missing values
    if df_melted.isnull().any().any():
        print("Warning: Dataset contains missing values")
        df_melted = df_melted.dropna()
        print("Missing values have been removed")

finally:
    # Ensure connection is closed even if an error occurs
    conn.close()

try:
    # Prepare features (X) and target (y)
    X = df_melted[['months_since_start', 'coastal_distance']]
    y = df_melted['price_index']

    # Check if we have enough data for splitting
    if len(df_melted) < 5:
        raise ValueError("Not enough data points for analysis")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print model performance and coefficients
    print(f"Model Score (R²): {model.score(X_test, y_test):.4f}")
    print(f"Time coefficient (price change per month): {model.coef_[0]:.4f}")
    print(f"Coastal distance coefficient: {model.coef_[1]:.4f}")

    # Visualize the results
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot with color indicating coastal distance
    scatter = plt.scatter(X_test['months_since_start'], 
                         y_test,
                         c=X_test['coastal_distance'],
                         cmap='viridis',
                         alpha=0.5,
                         label='Actual Data')
    
    # Add colorbar to show coastal distance scale
    plt.colorbar(scatter, label='Distance from Coast')
    
    # Plot predicted values
    plt.scatter(X_test['months_since_start'],
                y_pred,
                color='red',
                marker='x',
                label='Predicted Values')
    
    plt.xlabel('Months Since Start')
    plt.ylabel('Price Index')
    plt.title('Housing Price Index Prediction')
    plt.legend()
    plt.show()

    # Plot price trends for different coastal distances
    plt.figure(figsize=(12, 6))
    distances = [0, 10, 50, 100]  # example distances to plot
    months = np.linspace(df_melted['months_since_start'].min(), 
                        df_melted['months_since_start'].max(), 
                        100)
    
    for distance in distances:
        X_pred = pd.DataFrame({
            'months_since_start': months,
            'coastal_distance': [distance] * len(months)
        })
        y_pred = model.predict(X_pred)
        plt.plot(months, y_pred, label=f'Distance: {distance}km')
    
    plt.xlabel('Months Since Start')
    plt.ylabel('Price Index')
    plt.title('Predicted Price Trends by Coastal Distance')
    plt.legend()
    plt.show()

except NotFittedError:
    print("Error: Model failed to fit the data")
except ValueError as e:
    print(f"Error in data processing: {e}")
except Exception as e:
    print(f"Unexpected error occurred: {e}") 