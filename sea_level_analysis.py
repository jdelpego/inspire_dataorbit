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
    query = """
        SELECT Year, SmoothedGSML_GIA_sigremoved 
        FROM sealevel
    """
    df = pd.read_sql(query, conn)

    # Check if we got any data
    if df.empty:
        raise ValueError("No data retrieved from the database")

    # Check for missing values
    if df.isnull().any().any():
        print("Warning: Dataset contains missing values")
        df = df.dropna()
        print("Missing values have been removed")

finally:
    # Ensure connection is closed even if an error occurs
    conn.close()

try:
    # Prepare features (X) and target (y)
    X = df[['Year']]
    y = df['SmoothedGSML_GIA_sigremoved']

    # Check if we have enough data for splitting
    if len(df) < 5:  # arbitrary minimum size
        raise ValueError("Not enough data points for analysis")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print model performance
    print(f"Model Score (R²): {model.score(X_test, y_test):.4f}")
    print(f"Slope (mm/year): {model.coef_[0]:.4f}")  # Removed conversion since Year is already annual

    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Year'], df['SmoothedGSML_GIA_sigremoved'], 
                color='blue', alpha=0.5, label='Actual Data')
    
    # Sort X_test for proper line plotting
    X_test_sorted = X_test.sort_values('Year')
    y_pred_sorted = model.predict(X_test_sorted)
    
    plt.plot(X_test_sorted, y_pred_sorted, color='red', label='Predicted Line')
    plt.xlabel('Year')
    plt.ylabel('Global Mean Sea Level (mm)')
    plt.title('Sea Level Rise Linear Regression')
    plt.legend()
    plt.show()

except NotFittedError:
    print("Error: Model failed to fit the data")
except ValueError as e:
    print(f"Error in data processing: {e}")
except Exception as e:
    print(f"Unexpected error occurred: {e}") 