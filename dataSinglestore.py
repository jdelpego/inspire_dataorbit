import mysql.connector

# Attempt connection to the database
try:
    conn = mysql.connector.connect(
        host='svc-3482219c-a389-4079-b18b-d50662524e8a-shared-dml.aws-virginia-6.svc.singlestore.com',
        user='jdelpego',
        password='aqSUOyeg5GWQJ6NMWjVifT1WCoRmYE50',
        database='db_luke_503d4'
    )
    print("Connection successful!")
except mysql.connector.Error as err:
    print(f"Error: {err}")
    exit()  # Exit if unable to connect

cursor = conn.cursor()

# Run a query to fetch all data
cursor.execute("SELECT * FROM Neigborhood_Price_Time_Series")
print("Query executed!")

# Fetch all the rows from the query result
rows = cursor.fetchall()
print(f"Fetched {len(rows)} rows.")

# Print the rows to check the results
for row in rows:
    print(row)

# Close the connection
cursor.close()
conn.close()
