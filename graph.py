import streamlit as st
import numpy as np
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

# Create data for sea level rise (line)
years = np.arange(2020, 2051, 1)
sea_level = np.linspace(0, 3, len(years))  # Example sea level rise over time

# Create a ColumnDataSource for Bokeh
source = ColumnDataSource(data=dict(years=years, sea_level=sea_level))

# Create a Bokeh figure (correcting plot_height and plot_width to height and width)
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