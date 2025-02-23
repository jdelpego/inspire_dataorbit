🌊 Sea Level Rise Prediction App

📌 Overview

This web application predicts when a location will be affected by rising sea levels. Using historical sea level data, CO2 emissions, and elevation information, the model estimates when a specific point on the map will be submerged due to climate change.

🚀 Features

Interactive Map: Click on any land location to retrieve elevation data. Sea Level Prediction Model: Estimates the year when a location will be affected by rising sea levels. Google Maps Elevation API Integration: Fetches real-time elevation data. Data Visualization: Displays sea level rise. Educational Resources: Provides information on mitigating climate change.

📊 How It Works

User clicks on a location on the interactive map. Elevation is retrieved using the Google Maps Elevation API. Sea level rise prediction model estimates the flooding year based on CO2 emissions and historical sea level data. Results are displayed, including: Latitude & Longitude, Elevation (meters), Predicted year of flooding, Years until flooding

🛠️ Technologies Used

Python (Streamlit, Pandas, NumPy, Scikit-learn), Folium (for interactive maps), Google Maps Elevation API (to get real-world elevation data), SingleStoreDB (for data storage), Machine Learning (Polynomial Regression for predictions)

🔧 Setup & Installation

Prerequisites

Python 3.x

pip (Python package manager)

Installation Steps

Clone this repository:

git clone https://github.com/yourusername/sea-level-predictor.git
cd sea-level-predictor

Install dependencies:

pip install -r requirements.txt

Add your Google Maps API Key to secrets.toml:

[api_key]
google_maps_api_key = "YOUR_API_KEY"

Run the application:

streamlit run app.py

📖 Data Sources: NOAA Sea Level Data, Global CO2 Emissions Data, Google Maps Elevation API

📌 Future Improvements

🌍 Enhance accuracy by incorporating temperature trends.

📡 Add real-time climate change projections.

📊 Improve visualization of predicted sea level rise.

🤝 Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue.

📜 License

This project is licensed under the MIT License.

🌱 Join us in spreading awareness about climate change and rising sea levels! 🌎


