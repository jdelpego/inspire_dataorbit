Flood Prediction App
A Streamlit-based web application to predict the flooding year and time remaining until submersion for a given location. The app takes latitude, longitude, elevation, and relevant flooding data to provide predictions based on the provided parameters.

Features
Location Details: Displays latitude, longitude, and sea level information.
Flood Prediction: Shows the submersion year and the estimated time remaining until the flooding occurs.
Interactive Map: (Optional) View and explore the location's flooding prediction on a map.
Customizable Inputs: Enter location data via URL query parameters to dynamically update predictions.
Disclaimer: Notifications to inform users that predictions can vary based on external factors not measured in the model.
Installation
To run the app locally, clone the repository and install the necessary dependencies:
bash
Copy
Edit
git clone https://github.com/yourusername/flood-prediction-app.git
cd flood-prediction-app
pip install -r requirements.txt
Requirements
Python 3.x
Streamlit
Other Python dependencies (refer to requirements.txt)

2. Input Parameters
The app retrieves the following query parameters via the URL:

lat: Latitude of the location (e.g., lat=40.7128).
lon: Longitude of the location (e.g., lon=-74.0060).
elevation: Sea level in meters (e.g., elevation=10.5).
flooding_year: Predicted flooding year (e.g., flooding_year=2050).
years_until: Years remaining until flooding occurs (e.g., years_until=25).

3. Viewing Predictions
The app will display the coordinates, sea level, submersion year, and time remaining based on the provided parameters.
A visual disclaimer is included to remind users that predictions may vary.

Disclaimer
Predictions made by this model are based on limited data and assumptions. External factors not included in the model may affect predictions. Use the predictions as estimates and for informational purposes only.

## 👥 Team Members

- **Joaquin Del Pego**
- **Ivy Holiday**
- **Samprita Chakraborty**
- **Calvin Lu**
- **Luke Herbelin**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

_Acknowledgments will be added upon project completion_

---
<div align="center">
Made with ❤️ for the NFL Data Analysis Datathon 2024
</div>
