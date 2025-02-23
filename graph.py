import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.io import show

# Set the title for your app
st.title("Interactive Sea Level Rise Simulator")

# Add a description for the app
st.write(
    """
    This graph shows the rise in sea levels over time. 
    You can drag the point on the line to see the exact sea level at a given year.
    """
)

# Path to the CSV file (make sure to use the correct path for your dataset)
csv_file_path = 'seaData.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Check if the necessary columns exist in the CSV file
if 'Time' in df.columns and 'GMSL' in df.columns:
    # Convert 'Time' to datetime and extract the year
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Year'] = df['Time'].dt.year

    # Ensure 'GMSL' is numeric
    df['GMSL'] = pd.to_numeric(df['GMSL'], errors='coerce')

    # Drop rows with missing values
    df = df.dropna(subset=['Year', 'GMSL'])

    years = df['Year']
    sea_level = df['GMSL']

    # Create a ColumnDataSource for Bokeh
    source = ColumnDataSource(data=dict(years=years, sea_level=sea_level))

    # Create a Bokeh figure
    p = figure(title="Projected Sea Level Rise", x_axis_label='Year', y_axis_label='Sea Level (meters)', height=400, width=700)

    # Add the line to the figure
    p.line('years', 'sea_level', source=source, line_width=2, color="blue", legend_label="Sea Level Rise")

    # Add a circle (point) that we can drag on the line
    p.circle('years', 'sea_level', source=source, size=8, color="red", alpha=0.6)

    # Add a hover tool to show the year and sea level at that point
    hover_tool = HoverTool()
    hover_tool.tooltips = [("Year", "@years"), ("Sea Level", "@sea_level")]
    p.add_tools(hover_tool)

    # Display the plot in Streamlit
    st.bokeh_chart(p)
else:
    st.error("The CSV file must contain 'Time' and 'GMSL' columns.")