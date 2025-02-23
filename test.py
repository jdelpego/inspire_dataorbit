import requests
import streamlit as st
google_maps_api_key = str(st.secrets["api_key"]["google_maps_api_key"])  # Replace with your actual API key
lat, lon = 34.052235, -118.243683
print("API Key: ", google_maps_api_key)
url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={google_maps_api_key}"
response = requests.get(url)

if response.status_code == 200:
    result = response.json()
    if result["status"] == "OK":
        elevation = result["results"][0]["elevation"]
        print(f"The elevation at latitude {lat} and longitude {lon} is {elevation} meters.")
    else:
        print("Error: Status not OK", result)
else:
    print("Error: Unable to fetch data", response.status_code)
