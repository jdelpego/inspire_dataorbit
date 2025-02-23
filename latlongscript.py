import pandas as pd
import requests
import time

# Your Google Maps API key (make sure to keep this secure)
api_key = 'AIzaSyDTpN7Kp2OkmQ27_DIAObaMVIm81tGXRzs'

# Function to get latitude and longitude for an address
def get_lat_long(address):
    # Google Maps Geocoding API endpoint
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
    
    # Make the API request
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        
        # Check if the API returned any results
        if data['status'] == 'OK':
            # Extract latitude and longitude from the response
            lat = data['results'][0]['geometry']['location']['lat']
            lng = data['results'][0]['geometry']['location']['lng']
            return lat, lng
        else:
            print(f"Geocoding failed for {address}: {data['status']}")
            return None, None
    else:
        print(f"API request failed for {address}. Status code: {response.status_code}")
        return None, None

# Read in your dataset (replace with your actual file path)
df = pd.read_csv('realtor-data.csv')
# Ensure all columns are strings, replacing NaN with empty strings
df['street'] = df['street'].fillna('').astype(str)
df['city'] = df['city'].fillna('').astype(str)
df['zip_code'] = df['zip_code'].fillna('').astype(str)

# Now concatenate the columns
df['full_address'] = df['street'] + ', ' + df['city'] + ', ' + df['zip_code']

# Empty lists to store latitudes and longitudes
latitudes = []
longitudes = []

# Loop through the dataset and get lat/lng for each address
for address in df['full_address']:
    lat, lng = get_lat_long(address)
    latitudes.append(lat)
    longitudes.append(lng)
    
    # Sleep for a moment to avoid hitting API rate limits
    time.sleep(0.1)  # Adjust sleep time based on your rate limits

# Add latitude and longitude columns to the dataframe
df['latitude'] = latitudes
df['longitude'] = longitudes

# Save the dataframe with latitudes and longitudes back to a CSV
df.to_csv('realtor-data_with_lat_lng.csv', index=False)

print("Script completed and output saved to 'realtor-data_lat_lng.csv'.")


import pandas as pd
import requests
import time

api_key = 'YOUR_GOOGLE_MAPS_API_KEY'

def get_lat_long(address):
    url = f'https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        
        if data['status'] == 'OK':
            lat = data['results'][0]['geometry']['location']['lat']
            lng = data['results'][0]['geometry']['location']['lng']
            return lat, lng
        else:
            print(f"Geocoding failed for {address}: {data['status']}")
            return None, None
    else:
        print(f"API request failed for {address}. Status code: {response.status_code}")
        return None, None

# Read in your dataset
df = pd.read_csv('path_to_your_file.csv')

# Ensure all columns are strings, replacing NaN with empty strings
df['street'] = df['street'].fillna('').astype(str)
df['city'] = df['city'].fillna('').astype(str)
df['zip_code'] = df['zip_code'].fillna('').astype(str)

df['full_address'] = df['street'] + ', ' + df['city'] + ', ' + df['zip_code']

latitudes = []
longitudes = []
failed_addresses = []  # To log failed addresses

for address in df['full_address']:
    lat, lng = get_lat_long(address)
    if lat is None or lng is None:
        failed_addresses.append(address)  # Log the failed address
    else:
        latitudes.append(lat)
        longitudes.append(lng)
    
    time.sleep(0.1)  # Avoid hitting API rate limits

df['latitude'] = latitudes
df['longitude'] = longitudes

# Save the dataframe with latitudes and longitudes
df.to_csv('output_with_lat_lng.csv', index=False)

# Save failed addresses to a file for inspection
with open('failed_addresses.txt', 'w') as f:
    for address in failed_addresses:
        f.write(address + '\n')

print("Script completed. Failed addresses logged in 'failed_addresses.txt'.")

