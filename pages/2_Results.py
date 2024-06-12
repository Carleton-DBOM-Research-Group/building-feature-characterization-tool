# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:08:50 2023

@author: Shane
"""
import matplotlib.colors as mcolors
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import time
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
import plotly.graph_objects as go
from Methods import user_inputs as ui
from Methods import benchmarks as bm
from Methods import plots as pl
from Methods import ann as ann
from Methods import clustering as cl
from Methods import histograms as hs
from Methods import linear_regression as lr
from Methods import weather as w
from Methods import energy as e
from scipy import stats



st.set_page_config(
    page_title="Building Features Characterization",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={ })



######################################################################
# Retrieve user inputs from session state
######################################################################
st.set_option('deprecation.showPyplotGlobalUse', False)
building_name, floor_area, address_line1, address_line2, city, Postal_code, year_of_construction, number_of_floors, wwr, aspect_ratio = ui.get_user_inputs()

######################################################################
# read in weather data & User's uploaded energy use data
######################################################################
# Check if the weather data is already in the session state
if 'weather' in st.session_state:
    weather = st.session_state['weather']

if weather is not None:
    Tout = weather['Dry Bulb Temperature {C}']

if 'energy_use_df' in st.session_state:
    User_Energy = pd.read_json(st.session_state['energy_use_df'])
    User_Energy1 = User_Energy
else:
    st.error('No energy use data found. Please upload the energy use data.')
    st.stop()  # stop the script execution

# Replace negative values with NaN
User_Energy.loc[User_Energy['Heating (W/m²)'] < 0, 'Heating (W/m²)'] = np.nan
User_Energy.loc[User_Energy['Cooling (W/m²)'] < 0, 'Cooling (W/m²)'] = np.nan

# Linearly interpolate the NaN values
User_Energy['Heating (W/m²)'] = User_Energy['Heating (W/m²)'].interpolate(method='linear')
User_Energy['Cooling (W/m²)'] = User_Energy['Cooling (W/m²)'].interpolate(method='linear')

# Add weather data to Energy use dataframe
Tout.index = User_Energy.index
User_Energy['Dry Bulb Temperature (\u00b0C)'] = weather['Dry Bulb Temperature {C}']  

# Convert the 'Time' column to datetime:
User_Energy['Time'] = pd.to_datetime(User_Energy['Time'], format="%Y-%m-%d %H:%M")

# Set the 'Time' column as the index:
User_Energy.set_index('Time', inplace=True)

######################################################################
# Use rupture algorithm to find operating hours , and seperate into operating and afterhours
######################################################################

# weekday_hours = e.get_all_operating_hours(weekday_data)
# weekend_hours = e.get_all_operating_hours(weekend_data)

# weekday_hours = e.convert_operating_hours_to_hours(weekday_hours)
weekday_hours = st.session_state['weekday_hours']
# weekend_hours = e.convert_operating_hours_to_hours(weekend_hours)
weekend_hours = st.session_state['weekend_hours']
# weekday_hours_mode = e.get_mode_of_hours(weekday_hours)
weekday_hours_mode = st.session_state['weekday_hours_mode']
# weekend_hours_mode = e.get_mode_of_hours(weekend_hours)
weekend_hours_mode = st.session_state['weekend_hours_mode']

# start_modes_weekdays, end_modes_weekdays, start_modes_weekends, end_modes_weekends = e.convert_mode_to_datetime(weekday_hours_mode, weekend_hours_mode)
start_modes_weekdays = st.session_state['start_modes_weekdays']
end_modes_weekdays = st.session_state['end_modes_weekdays']
start_modes_weekends = st.session_state['start_modes_weekends']
end_modes_weekends = st.session_state['end_modes_weekends']

# start_modes_weekdays_str, end_modes_weekdays_str, start_modes_weekends_str, end_modes_weekends_str = e.convert_mode_to_string(start_modes_weekdays, end_modes_weekdays, start_modes_weekends, end_modes_weekends)
start_modes_weekdays_str = st.session_state['start_modes_weekdays_str']
end_modes_weekdays_str = st.session_state['end_modes_weekdays_str']
start_modes_weekends_str = st.session_state['start_modes_weekends_str']
end_modes_weekends_str = st.session_state['end_modes_weekends_str']

# energyOp_weekdays, energyOp_weekends, energyOp = e.generate_operational_energy(User_Energy, start_modes_weekdays_str, end_modes_weekdays_str, start_modes_weekends_str, end_modes_weekends_str)
energyOp_weekdays = st.session_state['energyOp_weekdays']
energyOp_weekends = st.session_state['energyOp_weekends']
energyOp = st.session_state['energyOp']

# energyAf_weekdays, energyAf_weekends, energyAf = e.generate_afterhours_energy(User_Energy, end_modes_weekdays_str, start_modes_weekdays_str)
energyAf_weekdays = st.session_state['energyAf_weekdays']
energyAf_weekends = st.session_state['energyAf_weekends']
energyAf = st.session_state['energyAf']

#################################################################
# Piecewise Linear Regression
#################################################################
# Generate variables
# tChp, tOutTest, energyOp_Tout, energyAf_Tout = lr.generate_variables(energyOp, energyAf)

tChp = st.session_state['tChp']
tOutTest = st.session_state['tOutTest']
energyOp_Tout = st.session_state['energyOp_Tout']
energyAf_Tout = st.session_state['energyAf_Tout']

#######################################################################################################################################
# For operational heating
#########################################################################################################################################
# Sweep through the change-point temperatures to find the best operational change point
# OpHtgTbal, OpHtgSlope, OpHtgYint, R2OpHtg, CVRMSEOpHtg, fitOpHtg = lr.operational_heating(energyOp_Tout, tChp, energyOp)

OpHtgTbal = st.session_state['OpHtgTbal']
OpHtgSlope = st.session_state['OpHtgSlope']
OpHtgYint = st.session_state['OpHtgYint']
R2OpHtg = st.session_state['R2OpHtg']
CVRMSEOpHtg = st.session_state['CVRMSEOpHtg']
fitOpHtg = st.session_state['fitOpHtg']

##########################################################################################################################################
# For operational cooling 
##########################################################################################################################################
# Sweep through the change-point temperatures to find the best operational change point
# OpClgTbal, OpClgSlope, OpClgYint, R2OpClg, CVRMSEOpClg, fitOpClg = lr.operational_cooling(energyOp_Tout, tChp, energyOp)

OpClgTbal = st.session_state['OpClgTbal']
OpClgSlope = st.session_state['OpClgSlope']
OpClgYint = st.session_state['OpClgYint']
R2OpClg = st.session_state['R2OpClg']
CVRMSEOpClg = st.session_state['CVRMSEOpClg']
fitOpClg = st.session_state['fitOpClg']

##########################################################################################################################################
# For afterhours heating 
##########################################################################################################################################
# Sweep through the change-point temperatures to find the best operational change point
# AfHtgTbal, AfHtgSlope, AfHtgYint, R2AfHtg, CVRMSEAfHtg, fitAfHtg = lr.afterhours_heating(energyAf_Tout, tChp, energyAf)

AfHtgTbal = st.session_state['AfHtgTbal']
AfHtgSlope = st.session_state['AfHtgSlope']
AfHtgYint = st.session_state['AfHtgYint']
R2AfHtg = st.session_state['R2AfHtg']
CVRMSEAfHtg = st.session_state['CVRMSEAfHtg']
fitAfHtg = st.session_state['fitAfHtg']

##########################################################################################################################################
# For afterhours cooling 
##########################################################################################################################################
# Sweep through the change-point temperatures to find the best operational change point
# AfClgTbal, AfClgSlope, AfClgYint, R2AfClg, CVRMSEAfClg, fitAfClg = lr.afterhours_cooling(energyAf_Tout, tChp, energyAf)

AfClgTbal = st.session_state['AfClgTbal']
AfClgSlope = st.session_state['AfClgSlope']
AfClgYint = st.session_state['AfClgYint']
R2AfClg = st.session_state['R2AfClg']
CVRMSEAfClg = st.session_state['CVRMSEAfClg']
fitAfClg = st.session_state['fitAfClg']

##########################################################################################################################################
# Store regressed parameters in a df
##########################################################################################################################################

# Store the rergressed parameters in a list CPM


# Prepare regressed CPM parameters to be scaled
# X_inputs, CPM_df = lr.create_x_inputs(CPM)

X_inputs = st.session_state['X_inputs']

CPM_df = st.session_state['CPM_df']

##############################################################################
# Load scaler and models and make predictions with CPM parameters
##############################################################################

ann_models = st.session_state['ann_models']
scaler_dir = st.session_state['scaler_dir']
scaler = st.session_state['scaler']
X_inputs_norm = st.session_state['X_inputs_norm']
predictions = st.session_state['predictions']

##############################################################################
# Stack predictions from ANN models, scale features, elbow method, k-means clustering, and take median of each cluster
##############################################################################
# building features
building_features = ['Uwindow', 'SHGC', 'Ropaque', 'Qinfil', 'Qventilation', 'Qcasual', 'Fafterhours',
                             'Mset up/down', 'Tsa,clg', 'Tsa,htg', 'Tsa,reset', 'Tafterhours,htg', 'Tafterhours,clg',
                             'Fvav,min-sp', 'Shtg,summer']

predictions_stack = st.session_state['predictions_stack']
df = st.session_state['df']
prediction_summary_stats = df.describe()


# Normalize your data
scaler = MinMaxScaler()
predictions_normalized = scaler.fit_transform(predictions_stack)

# Elbow Method
distortions = st.session_state['distortions']
K = st.session_state['K']


# Find the optimal number of clusters
optimal_clusters = st.session_state['optimal_clusters']

# Apply K-means
kmeans = st.session_state['kmeans']

# Get the cluster assignments for each prediction
cluster_assignments = kmeans.labels_

# medians = cl.create_medians(cluster_assignments, predictions_stack)
medians = st.session_state['medians']

# Round each feature to its respective decimal places
medians_df_original_scale = st.session_state['medians_df_original_scale']
raw_medians = st.session_state['raw_medians']

    
######################################################################
# Results Title + page config __Front end
######################################################################

# Title and Subtitle
st.title("Building details")

# Header for Building Details
st.markdown("---")

# Column Layout
col1, col2 = st.columns((2, 1))

# Building Details in Column 1
with col1:
    st.markdown(f"""
    - **Building Name:** {building_name}
    - **Address Line 1:** {address_line1}
    - **Address Line 2:** {address_line2}
    - **City:** {city}
    - **Postal Code:** {Postal_code}
   
    """)

# Building Configuration in Column 2
with col2:
    # Formatting the details
    building_details_formatted = f"""
    - **Aspect Ratio:** {aspect_ratio}
    - **Window-Wall Ratio (WWR):** {wwr}%
    - **Floor Area:** {floor_area} m²
    - **Number of Floors:** {number_of_floors}
    - **Year of Construction:** {year_of_construction}
    """
    st.markdown(building_details_formatted)

# Divider Line
st.markdown("---")


######################################################################
# Results section 1 : Energy dashboard 
######################################################################

st.header('1. Energy Dashboard & Benchmarking')
with st.expander('Learn more about the energy dashboard'):
    st.markdown("""
    Here, you can explore insights into your building's energy use behavior. 
    You'll find interactive plots and data tables that offer a deep understanding of how your building consumes energy. 
    Choose from various plot types to visualize your building's heating and cooling load data.
    """)


col1, col2, col3 = st.columns([1, 0.1, 1])

# Set the width of columns 1 and 3
col1.write("")
col3.write("")
col1._css = f"width: 20%;"
col3._css = f"width: 20%;"


# Display energy use data in the left column
with col1:
    st.subheader("Hourly Energy Use Data")
    st.markdown("""
This table provides a snapshot of your building's hourly energy consumption. 
Use the plot on the right to visualize and analyze this data in different ways.
""")
    st.dataframe(User_Energy)
    
    # Add CSS to center the DataFrame in the column
    st.markdown(
        f"""
        <style>
        .dataframe {{
            margin: 0 auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
)

    # Set the maximum number of rows to display
    pd.set_option('display.max_rows', 27)
    
with col3:

    # Display plots in the right column
    # Create a selectbox button to select the plot type
    plot_type = st.selectbox("Select plot type", ("Time series plots", "Scatter", "Box plots", "Heat maps"))

    if plot_type == "Time series plots":
        # Display the plot
        st.pyplot(pl.plot_time_series(User_Energy, User_Energy1))

    elif plot_type == "Scatter":
        # Display the plot
        st.pyplot(pl.plot_scatter(User_Energy1))
    
    elif plot_type == "Box plots":
        # Display the plot
        st.pyplot(pl.plot_box(User_Energy, User_Energy1))

    elif plot_type == "Heat maps":
        User_Energy1 = User_Energy1.reset_index()
        # Group the data by date and calculate the mean heating and cooling energy use for each date
        User_Energy1['Date'] = User_Energy1['Time'].dt.date
        df_daily = User_Energy1.groupby('Date')[['Heating (W/m²)', 'Cooling (W/m²)']].mean()

        # Create a selectbox button to select the date
        selected_date = st.selectbox("Select date", df_daily.index)
        # Display the plot
        st.pyplot(pl.plot_heat_map(User_Energy1, selected_date))

# Add a separator between plots
st.markdown("---")

######################################################################
# Results section 1 : Benchmarking
######################################################################
# Convert from Wh/m²-yr to kWh/m²-yr
user_heating_eui = User_Energy['Heating (W/m²)'].sum() / 1000
user_cooling_eui = User_Energy['Cooling (W/m²)'].sum() / 1000
user_combined_eui = user_heating_eui + user_cooling_eui

# Retrieve the benchmark data
benchmark_df = bm.get_benchmark_data()
benchmark_df['WWR'] = benchmark_df['WWR'] * 100  # Multiply the WWR values by 100

# Add user's building entry to the DataFrame
user_building_data = {
    "Building": building_name,
    "Heating energy use intensity (kWh/m2-yr)": user_heating_eui,
    "Cooling energy use intensity (kWh/m2-yr)": user_cooling_eui,
    "Vintage": year_of_construction,
    "Floor area (1000 m2)": floor_area/1000,
    "WWR": wwr
}
buildings_df = benchmark_df.append(user_building_data, ignore_index=True)

# Calculate the combined energy use (heating + cooling) and add it as a new column
buildings_df['Combined EUI'] = buildings_df['Heating energy use intensity (kWh/m2-yr)'] + buildings_df['Cooling energy use intensity (kWh/m2-yr)']

# Sort the DataFrame by combined energy use, in descending order
buildings_df_sorted = buildings_df.sort_values(by='Combined EUI', ascending=False)

# create benchmarking summary table
summary_table = bm.create_summary_table(building_name, floor_area, year_of_construction, wwr, user_heating_eui, user_cooling_eui, user_combined_eui, benchmark_df)

# Summary of the characteristics of the buildings
st.subheader("Benchmarking")
# Introduction to Benchmarking
with st.expander('Learn more about benchmarking and why is it Important?'):
    st.markdown("""
    Benchmarking is the process of comparing your building's performance metrics against a set of standards or best practices within your industry or region. 
    In the context of building energy performance, it involves analyzing how your building's energy consumption and efficiency measures up to other similar buildings.
    
    Here's why benchmarking is vital:
    - **Identifies Strengths and Weaknesses:** By comparing against a benchmark, you can easily pinpoint areas where your building excels or needs improvement.
    - **Informs Decision Making:** The insights gained from benchmarking can guide retrofit decisions, energy management strategies, and investment priorities.
    - **Encourages Continuous Improvement:** Regular benchmarking encourages ongoing efforts to maintain or improve performance, leading to long-term energy savings.
    - **Enhances Sustainability Efforts:** Understanding how your building compares to others can help in aligning with sustainability goals and regulations.
   
    """)
    st.write(buildings_df_sorted)
    
st.write('''Below, you will find a comparison of your building's energy consumption against a study of 35 local Ottawa office buildings. 
         This benchmarking will provide valuable insights into how your building performs relative to others.
''')

# Create two columns
col1, col2 = st.columns(2)

# Column 1: Summary Table
with col1:
    st.subheader("Summary Statistics")
    st.table(summary_table)  # Assuming 'summary_table' contains the data

# Column 2: Bar Chart
with col2:
    st.subheader("Benchmarking Chart")

    # Create a figure
    fig = go.Figure()
    
    # Add the heating EUI bar
    fig.add_trace(go.Bar(
        x=buildings_df_sorted['Building'],
        y=buildings_df_sorted['Heating energy use intensity (kWh/m2-yr)'],
        name='Heating EUI',
        marker_color='blue',
        hovertemplate="Building: %{x}<br>Vintage: %{customdata[0]}<br>WWR: %{customdata[1]}<br>Floor Area: %{customdata[2]} (1000 m²)<br>Heating EUI: %{y} kWh/m²-yr",
        customdata=np.stack([buildings_df_sorted['Vintage'], buildings_df_sorted['WWR'], buildings_df_sorted['Floor area (1000 m2)']], axis=1)
    ))
    
    # Add the cooling EUI bar
    fig.add_trace(go.Bar(
        x=buildings_df_sorted['Building'],
        y=buildings_df_sorted['Cooling energy use intensity (kWh/m2-yr)'],
        name='Cooling EUI',
        marker_color='orange',
        hovertemplate="Building: %{x}<br>Vintage: %{customdata[0]}<br>WWR: %{customdata[1]}<br>Floor Area: %{customdata[2]} (1000 m²)<br>Cooling EUI: %{y} kWh/m²-yr",
        customdata=np.stack([buildings_df_sorted['Vintage'], buildings_df_sorted['WWR'], buildings_df_sorted['Floor area (1000 m2)']], axis=1)
    ))
    
    # Update layout
    fig.update_layout(
        showlegend=True,
        legend=dict(x=0.8, y=0.9),
        xaxis_title='Building',
        yaxis_title='Energy Use Intensity (kWh/m²-yr)',
        barmode='stack',
        xaxis=dict(tickmode='array', tickvals=list(range(len(buildings_df_sorted['Building']))), ticktext=buildings_df_sorted['Building']),
        hoverlabel=dict(bgcolor='white'),
        autosize=True,
        width=None  # Set to None to allow Streamlit to control the width
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Add a separator between plots
st.markdown("---")


##############################################################################
# Results section 2 - operating hours table
##############################################################################
st.header("2. Operating hours")
st.markdown("""
These operating hours represent the most typical start and end times for each day of the week. 
It's a valuable reference for understanding your building's routine and making informed decisions 
on energy management.
""")

with st.expander('Learn more about about how your operating hours are determined'):
    st.markdown('''This app uses an advanced algorithm to analyze your building's energy use data and determine 
                its typical operating schedule. It first separates the data into heating periods and then splits it
                by weekdays and weekends. It identifies periods of significant energy use, which indicate when your 
                building is likely in operation. For each day, the app finds the longest period of significant energy 
                use to determine the operating hours. Then, it calculates the most common start and end times of these 
                operating hours across all weekdays and weekends. This gives an estimate of the most typical operating 
                hours for your building. With this information, you can gain insights into your building's energy use 
                patterns and potentially identify opportunities for energy-saving strategies''')

# Define a threshold for the minimum frequency 
frequency_threshold = 0.4  # for example, 10%

# Calculate the frequency of the mode operating hours for each day of the week
weekday_frequency = weekday_hours.count(weekday_hours_mode) / len(weekday_hours)
weekend_frequency = weekend_hours.count(weekend_hours_mode) / len(weekend_hours)

# Check if the frequency is below the threshold
if weekday_frequency < frequency_threshold:
    start_modes_weekdays_str = 'Building Not Operating'
    end_modes_weekdays_str = 'Building Not Operating'
if weekend_frequency < frequency_threshold:
    start_modes_weekends_str = 'Building Not Operating'
    end_modes_weekends_str = 'Building Not Operating'

# Add the frequencies to the operating_hours dictionary
operating_hours = {
    'Day of the Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'Start Time': [start_modes_weekdays_str]*5 + [start_modes_weekends_str]*2,
    'End Time': [end_modes_weekdays_str]*5 + [end_modes_weekends_str]*2,
    'Frequency': [weekday_frequency]*5 + [weekend_frequency]*2
}

# Convert the dictionary to a pandas DataFrame
df_operating_hours = pd.DataFrame(operating_hours)

# Display the DataFrame as a table in Streamlit
st.markdown("### Typical Operating Hours:")
st.table(df_operating_hours)
# Display a warning if the frequency is below the threshold
if weekday_frequency < frequency_threshold or weekend_frequency < frequency_threshold:
    st.warning("""
    Please note that some of the operating hours have been marked as 'Building Not Operating' 
    due to a frequency below the threshold. This may indicate inconsistencies or special 
    circumstances in the building's operating schedule.
    """)

# Add a separator between plots
st.markdown("---")

######################################################################
# Results section 3 - Calibrated Change Point Model Plots
######################################################################
st.header("3. Calibrated Change-Point Model Plots")
st.markdown("""
Understanding the transition points in a building's heating and cooling requirements is essential for optimizing 
energy consumption. This section provides calibrated change-point model plots to help with that understanding.
These plots showcase the relationship between outdoor temperature and load intensity for different operational states. 
They are valuable tools for understanding your building's energy behavior and developing strategies for efficiency.
""")

expander = st.expander('Learn more about change-point models')

with expander:
    st.write("Three-parameter change-point models describe key transitions in a building's heating and cooling requirements:")
    st.latex(r"\beta_1: \text{Base load energy consumption}")
    st.latex(r"\beta_2: \text{Rate of change in energy consumption}")
    st.latex(r"\beta_3: \text{Change point temperature}")
    st.write("These models are valuable tools for understanding your building's energy behavior and developing strategies for efficiency. Let's explore these models in more detail for both heating and cooling scenarios:")

    # Heating Change-Point Models
    st.subheader('Heating Change-Point Models')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**General Form**')
        st.latex(r'''Q_{htg} = \beta_1 + \beta_2(T_{out} - \beta_3)^{-}''')
    with col2:
        st.markdown('**Calibrated Form**')
        st.latex(f'''Q_{{op,htg}} = {CPM_df.loc['Op_htg_Intercept', 0]:.2f}  {CPM_df.loc['Op_htg_Slope', 0]:.2f}(T_{{out}} - {CPM_df.loc['Op_htg_balanceTemp', 0]:.2f})^-''')
        st.latex(f'''Q_{{af,htg}} = {CPM_df.loc['Af_htg_Intercept', 0]:.2f}  {CPM_df.loc['Af_htg_Slope', 0]:.2f}(T_{{out}} - {CPM_df.loc['Af_htg_balanceTemp', 0]:.2f})^-''')

    # Cooling Change-Point Models
    st.subheader('Cooling Change-Point Models')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**General Form**')
        st.latex(r'''Q_{clg} = \beta_1  + \beta_2(T_{out} - \beta_3)^{+}''')
    with col2:
        st.markdown('**Calibrated Form**')
        st.latex(f'''Q_{{op,clg}} = {CPM_df.loc['Op_Clg_Intercept', 0]:.2f} + {CPM_df.loc['Op_Clg_Slope', 0]:.2f}(T_{{out}} - {CPM_df.loc['Op_Clg_balanceTemp', 0]:.2f})^+''')
        st.latex(f'''Q_{{af,clg}} = {CPM_df.loc['Af_Clg_Intercept', 0]:.2f} + {CPM_df.loc['Af_Clg_Slope', 0]:.2f}(T_{{out}} - {CPM_df.loc['Af_Clg_balanceTemp', 0]:.2f})^+''')


# Plot the change point models for operating hours heating and cooling
# First subplot
col1, col2 = st.columns(2)
with col1:
    st.subheader("Operational heating")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tOutTest, fitOpHtg, color='black', linestyle='solid', linewidth=3, label='3P CPM')
    ax.scatter(energyOp['Dry Bulb Temperature (\u00b0C)'], energyOp['Heating (W/m²)'], color='r', s=5, label='Measured Data')
    ax.set_xlim(-30, 35)
    ax.set_xticks(range(-30, 35, 5))
    ax.set_title(f"Operational Heating\nCVRMSE: {CVRMSEOpHtg[0]:.1f}%  R2: {R2OpHtg:.3f}")
    ax.set_xlabel('Outdoor temperature ($^{o}$C)')
    ax.set_ylabel('Load Intensity (W/m$^{2}$)')
    ax.legend(loc='upper right')
    ax.grid(True)
    st.pyplot(fig=plt)


with col2:
    st.subheader("Operational cooling")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tOutTest, fitOpClg, color='black', linestyle='solid', linewidth=3)
    ax.scatter(energyOp['Dry Bulb Temperature (\u00b0C)'], energyOp['Cooling (W/m²)'], color='b', s=10)
    ax.set_xlim(-30, 35)
    ax.set_xticks(range(-30, 35, 5))
    ax.set_title(f"Operational Cooling\nCVRMSE: {CVRMSEOpClg[0]:.1f}%  R2: {R2OpClg:.3f}")
    ax.set_xlabel('Outdoor temperature ($^{o}$C)')
    ax.set_ylabel('Load Intensity (W/m$^{2}$)')
    ax.grid(True)
    st.pyplot(fig=plt)

# Add a separator between plots
st.markdown("---")

    # Second subplot
col3, col4 = st.columns(2)
with col3:
    st.subheader("Afterhours heating")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tOutTest, fitAfHtg, color='black', linestyle='solid', linewidth=3)
    ax.scatter(energyAf['Dry Bulb Temperature (\u00b0C)'], energyAf['Heating (W/m²)'], color='r', s=10)
    ax.set_xlim(-30, 35)
    ax.set_xticks(range(-30, 35, 5))
    ax.set_title(f"Afterhours Heating\nCVRMSE: {CVRMSEAfHtg[0]:.1f}%  R2: {R2AfHtg:.3f}")
    ax.set_xlabel('Outdoor temperature ($^{o}$C)')
    ax.set_ylabel('Load Intensity (W/m$^{2}$)')
    ax.grid(True)
    st.pyplot(fig=plt)
    
with col4:
    st.subheader("Afterhours cooling")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tOutTest, fitAfClg, color='black', linestyle='solid', linewidth=3)
    ax.scatter(energyAf['Dry Bulb Temperature (\u00b0C)'], energyAf['Cooling (W/m²)'], color='b', s=10)
    ax.set_xlim(-30, 35)
    ax.set_xticks(range(-30, 35, 5))
    ax.set_title(f"Afterhours Cooling\nCVRMSE: {CVRMSEAfClg[0]:.1f}%  R2: {R2AfClg:.3f}")
    ax.set_xlabel('Outdoor temperature ($^{o}$C)')
    ax.set_ylabel('Load Intensity (W/m$^{2}$)')
    ax.grid(True)
    st.pyplot(fig=plt)
    

# Model Summary
st.subheader('Change-Point Model Summary')
st.markdown("""
These change-point model (CPM) parameters are essential for predicting energy-related building features. 
Below is a summary of the parameters:
""")

# Format the data
data = {
    ('Heating', 'Change-Point Temp (\u00b0C)'): [f"{CPM_df.loc['Op_htg_balanceTemp', 0]:.0f}", f"{CPM_df.loc['Af_htg_balanceTemp', 0]:.0f}"],
    ('Heating', 'Slope (W/m²\u00b0C)'): [f"{CPM_df.loc['Op_htg_Slope', 0]:.2f}", f"{CPM_df.loc['Af_htg_Slope', 0]:.2f}"],
    ('Heating', 'Intercept (W/m²)'): [f"{CPM_df.loc['Op_htg_Intercept', 0]:.2f}", f"{CPM_df.loc['Af_htg_Intercept', 0]:.2f}"],
    ('Heating', 'CVRMSE (%)'): [f"{CVRMSEOpHtg[0]:.1f}", f"{CVRMSEAfHtg[0]:.1f}"],
    ('Heating', 'R2'): [f"{R2OpHtg:.3f}", f"{R2AfHtg:.3f}"],
    ('Cooling', 'Change-Point Temp (\u00b0C)'): [f"{CPM_df.loc['Op_Clg_balanceTemp', 0]:.0f}", f"{CPM_df.loc['Af_Clg_balanceTemp', 0]:.0f}"],
    ('Cooling', 'Slope (W/m²\u00b0C)'): [f"{CPM_df.loc['Op_Clg_Slope', 0]:.2f}", f"{CPM_df.loc['Af_Clg_Slope', 0]:.2f}"],
    ('Cooling', 'Intercept (W/m²)'): [f"{CPM_df.loc['Op_Clg_Intercept', 0]:.2f}", f"{CPM_df.loc['Af_Clg_Intercept', 0]:.2f}"],
    ('Cooling', 'CVRMSE (%)'): [f"{CVRMSEOpClg[0]:.1f}", f"{CVRMSEAfClg[0]:.1f}"],
    ('Cooling', 'R2'): [f"{R2OpClg:.3f}", f"{R2AfClg:.3f}"]
}


change_point_summary_df = pd.DataFrame(data, index=['Operating', 'Afterhours'])
change_point_summary_df.columns.name = 'Parameter'
change_point_summary_df.index.name = 'Time of Day'

col1, col2, col3 = st.columns([0.5, 6, 0.5])

# Define CSS style
style = """
    <style>
        table.dataframe {
            text-align: center;
            font-size: 14px;
            width: 100%; /* Use full width of the middle column */
            margin-left: auto; /* Center the table */
            margin-right: auto; /* Center the table */
        }
        th {
            text-align: center;
            font-weight: bold;
        }
        td {
            text-align: center;
        }
    </style>
"""

# Apply styling to the DataFrame and display in the middle column
col2.markdown(style, unsafe_allow_html=True)
col2.dataframe(change_point_summary_df)

# Extract the balance temperatures
heating_balance_temp_operating = float(CPM_df.loc['Op_htg_balanceTemp', 0])
cooling_balance_temp_operating = float(CPM_df.loc['Op_Clg_balanceTemp', 0])
heating_balance_temp_afterhours = float(CPM_df.loc['Af_htg_balanceTemp', 0])
cooling_balance_temp_afterhours = float(CPM_df.loc['Af_Clg_balanceTemp', 0])

# Define a threshold (e.g., 1 degree Celsius) to determine if the balance temperatures are close enough
threshold = 1.0

# Check if the balance temperatures for both heating and cooling are close to each other during operating hours
if abs(heating_balance_temp_operating - cooling_balance_temp_operating) <= threshold:
    st.markdown("**Note:** During operating hours, the building may be heating and cooling under identical conditions.")

# Check if the balance temperatures for both heating and cooling are close to each other during afterhours
if abs(heating_balance_temp_afterhours - cooling_balance_temp_afterhours) <= threshold:
    st.markdown("**Note:** During afterhours, the building may be heating and cooling under identical conditions.")

# Add a separator between plots
st.markdown("---")

######################################################################
# Results section 4 - Estimated building features
####################################################################
# Define the units for each feature
units = {
    'Uwindow': 'W/m²K', 
    'SHGC': '~', 
    'Ropaque': 'm²K/W', 
    'Qinfil': 'L/sm²', 
    'Qventilation': 'L/sm²', 
    'Qcasual': 'W/m²', 
    'Fafterhours': '%', 
    'Mset up/down': '~',  
    'Tsa,clg': '°C', 
    'Tsa,htg': '°C',
    'Tsa,reset': '~', 
    'Tafterhours,htg': '°C', 
    'Tafterhours,clg': '°C',
    'Fvav,min-sp': '%', 
    'Shtg,summer': '~'
}

# Define the columns for each table
table_7_columns = ['Uwindow (W/m²K)', 'SHGC (~)', 'Ropaque (m²K/W)', 'Qinfil (L/sm²)', 'Qventilation (L/sm²)',
                   'Qcasual (W/m²)', 'Fafterhours (%)']
table_8_columns = ['Mset up/down (~)', 'Tsa,clg (°C)', 'Tsa,htg (°C)', 'Tsa,reset (~)', 'Tafterhours,htg (°C)',
                   'Tafterhours,clg (°C)', 'Fvav,min-sp (%)', 'Shtg,summer (~)']

# Create Table 7
table_7 = medians_df_original_scale[table_7_columns]
table_7.insert(0, "Cluster", range(1, len(table_7)+1))  # Add a column for the cluster numbers

# Create Table 8
table_8 = medians_df_original_scale[table_8_columns]
table_8.insert(0, "Cluster", range(1, len(table_8)+1))  # Add a column for the cluster numbers


# Combine the tables
all_clusters = pd.concat([table_7.set_index('Cluster'), table_8.set_index('Cluster')], axis=1)



# Define the limits for each feature
limits = {
    'Uwindow (W/m²K)': (1.5, 3.6),
    'SHGC (~)': (0.3, 0.7),
    'Ropaque (m²K/W)': (1.5, 5),
    'Qinfil (L/sm²)': (0.1, 1),
    'Qventilation (L/sm²)': (0.2, 2),
    'Qcasual (W/m²)': (3, 20),
   'Fafterhours (%)': (0, 1),  # Assuming percentage scale
   'Mset up/down (~)': (0, 1),  # Assuming on/off scale
   'Tsa,clg (°C)': (12, 15),
   'Tsa,htg (°C)': (15.1, 24),
   'Tsa,reset (~)': (0, 1),  # Assuming constant/variable scale
#   'Tafterhours,htg (°C)': (15, 22),
#   'Tafterhours,clg (°C)': (25, 30),
   'Fvav,min-sp (%)': (0.1, 0.6),  # Assuming percentage scale
   'Shtg,summer (~)': (0, 1)  # Assuming on/off scale
   }

# Clip values for table_7
for feature, (lower, upper) in limits.items():
    if feature in table_7.columns:
        table_7[feature] = table_7[feature].clip(lower, upper)

# Clip values for table_8
for feature, (lower, upper) in limits.items():
    if feature in table_8.columns:
        table_8[feature] = table_8[feature].clip(lower, upper)
        
# Create a copy for display
table_7_display = table_7.copy()
table_8_display = table_8.copy()


# Format specific columns in the displayed table 7 and 8
table_7_display['Uwindow (W/m²K)'] = table_7['Uwindow (W/m²K)'].apply(lambda x: f"{x:.1f}")
table_7_display['SHGC (~)'] = table_7['SHGC (~)'].apply(lambda x: f"{x:.1f}")
table_7_display['Ropaque (m²K/W)'] = table_7['Ropaque (m²K/W)'].apply(lambda x: f"{x:.1f}")
table_7_display['Qinfil (L/sm²)'] = table_7['Qinfil (L/sm²)'].apply(lambda x: f"{x:.4f}")
table_7_display['Qventilation (L/sm²)'] = table_7['Qventilation (L/sm²)'].apply(lambda x: f"{x:.4f}")
table_7_display['Qcasual (W/m²)'] = table_7['Qcasual (W/m²)'].apply(lambda x: f"{x:.0f}")
table_7_display['Fafterhours (%)'] = table_7['Fafterhours (%)'].apply(lambda x: f"{x * 100:.0f}%")
table_8_display['Mset up/down (~)'] = table_8['Mset up/down (~)'].apply(round)
table_8_display['Tsa,clg (°C)'] = table_8['Tsa,clg (°C)'].apply(round)
table_8_display['Tsa,htg (°C)'] = table_8['Tsa,htg (°C)'].apply(round)
table_8_display['Tsa,reset (~)'] = table_8['Tsa,reset (~)'].apply(round)
table_8_display['Tafterhours,clg (°C)'] = table_8['Tafterhours,clg (°C)'].apply(lambda x: f"{x*23.5:.0f}")
table_8_display['Tafterhours,htg (°C)'] = table_8['Tafterhours,htg (°C)'].apply(lambda x: f"{x*22:.0f}")
table_8_display['Fvav,min-sp (%)'] = table_8['Fvav,min-sp (%)'].apply(lambda x: f"{x * 100:.0f}%")
table_8_display['Shtg,summer (~)'] = table_8['Shtg,summer (~)'].apply(round)




# Load the residuals CSV file
residuals_file_path = os.path.join(scaler_dir, 'residuals.csv')  # Adjust the path as needed
residuals_df = pd.read_csv(residuals_file_path)

# Get the list of clusters (assuming you have them, or you can create a dummy list)
clusters = range(1, len(K)-1)
print("Clusters")
print(clusters)

# Improved Headers and Organization
st.header('4.1 Predictions: Most Probable Building Features')
st.write('Explore the following sections to understand the insights gained from the surrogate model predictions and how they can guide retrofit actions.')

# Introduction to the Surrogate Model, Predictions, and Implications
with st.expander("Understanding the Predictions, Clustering, and User Actions"):
    st.markdown('''
    ### The Process
    The identification of the most probable building features follows these key steps:
    1. **Inputs to the Surrogate Model**: Scaled building parameters serve as inputs.
    2. **Predictions by ANNs**: An ensemble of 100 ANNs predicts plausible energy-related feature sets.
    3. **Clustering of Predictions**: To provide concise and actionable insights, predictions are clustered.
    4. **Displaying Clustered Predictions**: The median value of each cluster serves as the 'most likely' set of features.

    ### Why Clustering?
    Clustering serves multiple functions:
    - **Identify Realistic Feature Sets**: Groups similar predictions to pinpoint probable combinations of building features.
    - **Simplify Interpretation**: Reduces the complexity of results into a few representative groups.
    - **Highlight Non-Compliant Areas**: Helps identify groups or feature sets that may not meet building codes.
    
    ### What You Need to Know
    1. **Central Tendency**: The median value in each cluster acts as its central tendency.
    2. **Retrofit Decision Support**: Clusters can guide prioritizing retrofit actions.
    3. **User Action**: After reviewing, select a cluster that closely aligns with your building's known features for further analysis.
    ''')


# Thermal Transmittance, Air Exchange, and Casual Gains Table
st.header("Thermal Transmittance, Air Exchange, and Casual Gains")
st.markdown('''
This table relates to the building envelope and air exchange. Use it to understand thermal performance and identify retrofit opportunities.
''')
col1, col2, col3 = st.columns([0.5, 6, 0.5])
col2.dataframe(table_7_display)

with st.expander("Click here for thermal transmittance, air exchange, and casual gains feature descriptions"):
    st.markdown('''
    #### Feature Descriptions
    - **Window U-value (Uwindow)**: Measures the rate of heat loss through windows. Lower values are generally better. (Range: 1.5 - 3.6 W/m²·K)
    - **Solar Heat Gain Coefficient (SHGC)**: Indicates the fraction of solar radiation admitted through a window. (Range: 0.3 - 0.7)
    - **Opaque Assembly R-value (Ropaque)**: Insulation level of walls, roof, and floors. Higher values are generally better. (Range: 1.5 - 5 m²·K/W)
    - **Infiltration Rate (Qinfil)**: Rate of outdoor air leakage into the building. Lower values are generally better. (Range: 0.1 - 1 L/s·m²)
    - **Ventilation Rate (Qventilation)**: Rate of outdoor air brought into the building for ventilation. (Range: 0.2 - 2 L/s·m²)
    - **Casual Heat Gains (Qcasual)**: Internal heat gains from occupants, equipment, and lighting. (Range: 3 - 20 W/m²)
    - **Afterhours Casual Heat Gain Fraction (Fafterhours)**: Fraction of heat gains occurring outside regular hours. (Range: 0 - 1)
    ''')


# HVAC Operation and Control Table
st.header("HVAC Operation and Control Features")
st.markdown('''
This table focuses on the HVAC systems. Use it to evaluate your HVAC control strategies.
''')
col1, col2, col3 = st.columns([0.5, 6, 0.5])
col2.dataframe(table_8_display)

with st.expander("Click here for HVAC operation and control features feature descriptions"):
    st.markdown('''
    #### HVAC Operation and Control Feature Descriptions
    - **Night Cycle Availability (Mset up/down)**: Control strategy during unoccupied hours. A higher value indicates a more aggressive setback strategy. (Range: 0 - 1)
    - **Supply Air Temperature during Cooling (Tsa,clg)**: The setpoint temperature for cooling operations. (Range: 12 - 15℃)
    - **Supply Air Temperature during Heating (Tsa,htg)**: The setpoint temperature for heating operations. (Range: 15.1 - 24℃)
    - **Constant or Variable Supply Air Temperature (Tsa,reset)**: Indicates whether the supply air temperature is constant or variable. (Range: 0 - 1)
    - **Afterhours Thermostat Setback Heating (Tafterhours,htg)**: Thermostat settings for heating during unoccupied hours. (Range: 15 - 22℃)
    - **Afterhours Thermostat Setback Cooling (Tafterhours,clg)**: Thermostat settings for cooling during unoccupied hours. (Range: 25 - 30℃)
    - **Minimum VAV Terminal Airflow Fraction (Fvav,min-sp)**: Specifies the minimum airflow rate for VAV terminals. (Range: 0.1 - 0.6%)
    - **Summer Heating Availability (Shtg,summer)**: Indicates the availability of heating during the summer months. (Range: 0 - 1)
    ''')



# Main header for the section
st.header("Identify Your Building's Most Likely Feature Set")

# Sub-header and explanatory text
st.subheader("Step 1: Choose a Cluster")
st.markdown('''
Clusters represent groups of buildings with similar energy-related features.
By selecting a cluster, you align your building with a set of most probable features,
which can serve as a foundation for understanding its current performance and for planning retrofit actions.
''')

# Get the number of optimal clusters and list them
optimal_clusters = st.session_state['optimal_clusters']
clusters = list(range(1, optimal_clusters + 1))

# Generate cluster descriptions (this could be dynamically generated based on the data)
cluster_descriptions = [f"Cluster {i}: Description for cluster {i}" for i in clusters]

# Cluster selection with tooltip
selected_cluster = st.selectbox(
    'Choose a Cluster:',
    options=clusters,
    format_func=lambda x: cluster_descriptions[x-1],
    help="Select a cluster that you believe closely aligns with your building's known or suspected features."
)


# Extract the relevant predictions
predictions_critical_feat = table_7.iloc[0].drop("Cluster").values.astype(float)
predictions_HVAC_op_control = table_8.iloc[0].drop("Cluster").values.astype(float)

# Concatenate the predictions from both tables
predictions_combined = np.concatenate([predictions_critical_feat, predictions_HVAC_op_control])

# Repeat the combined predictions to match the shape of the residuals
predictions_for_cluster = np.tile(predictions_combined, (len(residuals_df), 1))

# Define a list of feature names
feature_names = ['Uwindow', 'Ropaque', 'Qinfil', 'Qventilation', 'Qcasual', 'Fafterhours']

# Define a dictionary that maps each feature to its unit
feature_units = {'Uwindow': 'W/m²K', 'Ropaque': 'm²K/W',
                 'Qinfil': 'L/sm²', 'Qventilation': 'L/sm²',
                 'Qcasual': 'W/m²', 'Fafterhours': '%'}

# Get the median values for the selected cluster from table_7 (first seven features)
selected_median_table_7 = table_7.iloc[selected_cluster - 1].drop("Cluster").values.astype(float)
selected_median_table_8 = table_8.iloc[selected_cluster - 1, 1:].values.astype(float)
combined_median = np.concatenate([selected_median_table_7, selected_median_table_8])

combined_median[11] *= 22  # Row 11 (0-based index is 10)
combined_median[12] *= 23.5  # Row 12 (0-based index is 11)


lower_bounds_df = st.session_state['lower_bounds_df']
upper_bounds_df = st.session_state['upper_bounds_df']
cluster_stds_df = st.session_state['cluster_stds_df']


# Extract the lower and upper bounds for the selected cluster (first seven features)
selected_lower = lower_bounds_df.iloc[selected_cluster - 1].values[:]
selected_upper = upper_bounds_df.iloc[selected_cluster - 1].values[:]
selected_std = cluster_stds_df.iloc[selected_cluster - 1].values[:]

alpha = 0.2
z_value = stats.norm.ppf(1 - alpha / 2)  # Z-value for 80% confidence

# Create and display the DataFrame using the first seven feature names
selected_cluster_df = pd.DataFrame({
    'Feature': building_features[:],
    'Median': combined_median,
    'Standard Deviation':selected_std,
    'Lower Bound': combined_median - z_value * selected_std,
    'Upper Bound': combined_median + z_value * selected_std
})


formatted_cluster_df = selected_cluster_df.copy()

# Define feature-specific formatting: Feature name mapped to number of decimal places
specific_formatting = {
    'Uwindow (W/m²K)': 1,  
    'SHGC (~)': 1,  
    'Ropaque (m²K/W)': 1,  
    'Qinfil (L/sm²)': 4,  
    'Qventilation (L/sm²)': 4,  
    'Qcasual (W/m²)': 1,  
    'Fafterhours (%)': 2,  
    'Mset up/down (~)': 1,  
    'Tsa,clg (°C)': 1,  
    'Tsa,htg (°C)': 1,  
    'Tsa,reset (~)': 0,  
    'Tafterhours,htg (°C)': 1,  
    'Tafterhours,clg (°C)': 1,  
    'Fvav,min-sp (%)': 2,  
    'Shtg,summer (~)': 0  
}

bounds_dict = {
    'Uwindow (W/m²K)': {'lower': 1.2, 'upper':4.0 }, 
    'SHGC (~)': {'lower': 0.2, 'upper': 0.8},  
    'Ropaque (m²K/W)': {'lower': 1.2, 'upper': 5.5},  
    'Qinfil (L/sm²)': {'lower': 0.1, 'upper': 1.7},  
    'Qventilation (L/sm²)': {'lower': 0.1, 'upper': 2.2},  
    'Qcasual (W/m²)': {'lower': 2, 'upper': 25},  
    'Fafterhours (%)': {'lower': 0, 'upper': 1},  
    'Mset up/down (~)': {'lower': 0, 'upper': 1},  
    'Tsa,clg (°C)': {'lower': 10, 'upper': 15},  
    'Tsa,htg (°C)': {'lower': 14, 'upper': 22},  
    'Tsa,reset (~)': {'lower': 0, 'upper': 1},  
    'Tafterhours,htg (°C)': {'lower': 5, 'upper': 22},  
    'Tafterhours,clg (°C)': {'lower': 5, 'upper': 32},  
    'Fvav,min-sp (%)': {'lower': 0.1, 'upper': 1},  
    'Shtg,summer (~)': {'lower': 0, 'upper': 10.0}  
}

# Format the 'Feature' column by appending the unit information
formatted_cluster_df['Feature'] = formatted_cluster_df['Feature'].apply(lambda feature: f"{feature} ({units.get(feature, 'Unknown')})")

# Specific formatting for certain features
for feature_with_unit, decimal_places in specific_formatting.items():
    row_to_update = formatted_cluster_df['Feature'] == feature_with_unit
    
    for col in ['Median', 'Lower Bound', 'Upper Bound']:
        formatted_cluster_df.loc[row_to_update, col] = formatted_cluster_df.loc[row_to_update, col].astype(float).apply(lambda x: f"{x:.{decimal_places}f}")

# Apply bounds
for feature, bounds in bounds_dict.items():
    lower_bound = bounds['lower']
    upper_bound = bounds['upper']
    
    row_to_update = formatted_cluster_df['Feature'] == feature
    
    if row_to_update.any():
        current_lower = formatted_cluster_df.loc[row_to_update, 'Lower Bound'].astype(float).iloc[0]
        current_upper = formatted_cluster_df.loc[row_to_update, 'Upper Bound'].astype(float).iloc[0]
        
        # Update and limit the lower and upper bounds
        formatted_cluster_df.loc[row_to_update, 'Lower Bound'] = max(current_lower, lower_bound)
        formatted_cluster_df.loc[row_to_update, 'Upper Bound'] = min(current_upper, upper_bound)

# Re-format 'Lower Bound' and 'Upper Bound' columns to remove trailing zeros
for feature_with_unit, decimal_places in specific_formatting.items():
    row_to_update = formatted_cluster_df['Feature'] == feature_with_unit
    for col in ['Lower Bound', 'Upper Bound']:
        formatted_cluster_df.loc[row_to_update, col] = formatted_cluster_df.loc[row_to_update, col].astype(float).apply(lambda x: f"{x:.{decimal_places}f}")

# Round the 'Standard Deviation' column to two decimal places
formatted_cluster_df['Standard Deviation'] = formatted_cluster_df['Standard Deviation'].astype(float).apply(lambda x: f"{x:.2f}")


###########################################################################################        
# Define retrofit criteria (example: features below these values need retrofit)
###########################################################################################        

retrofit_conditions = {
    'Uwindow': lambda x: x > 1.9,
    'Ropaque': lambda x: x < 4.05,
    'Qinfil': lambda x: x > 0.35,
    'Qventilation': lambda x: x > 0.5,
    'Qcasual': lambda x: x > (7.5 + 8.5),
    'Fafterhours': lambda x: x > 0.5,
    'Mset up/down': lambda x: x == 0,
    'Tsa,reset': lambda x: x == 0,
    'Tafterhours,htg': lambda x: x > 22,
    'Tafterhours,clg': lambda x: x < 23,
    'Fvav,min-sp': lambda x: x > 0.3,
    'Shtg,summer': lambda x: x == 1
}

# Identify features that need retrofit based on conditions
retrofit_features = [feature for feature, condition in retrofit_conditions.items() 
                     if condition(selected_cluster_df.loc[selected_cluster_df['Feature'] == feature, 'Median'].values[0])]

# Check if Uwindow needs retrofit, then mark SHGC as well
if 'Uwindow' in retrofit_features:
    retrofit_features.append('SHGC')

# Check if Ropaque needs retrofit, then mark Qinfil as well
if 'Ropaque' in retrofit_features:
    retrofit_features.append('Qinfil')

# Ensure unique features in the list
retrofit_features = list(set(retrofit_features))

# Update the 'Retrofit Potential' column
formatted_cluster_df['Retrofit Potential'] = selected_cluster_df['Feature'].apply(lambda x: 'Yes' if x in retrofit_features else 'No')

# Create two columns to display content
col1, col2 = st.columns(2)

# Heading and explanation for the selected cluster's Prediction Interval
col1.subheader("Step 2: Understanding Prediction Intervals")
col1.markdown('''
The Prediction Interval provides an 80% Confidence Interval for the features within the selected cluster. This gives you:
- **Median**: The most representative value of each feature within the cluster.
- **Standard Deviation**: A measure of the variability or dispersion for each predicted feature within the cluster. A smaller value suggests less variability in that particular feature , while a larger value indicates greater variability.
- **Lower_Bound**: The lower boundary of the 80% CI for each feature.
- **Upper_Bound**: The upper boundary of the 80% CI for each feature.
This interval is useful for assessing the precision of the cluster's feature estimations.
''')

# Display the formatted data table for Prediction Interval
col1.table(formatted_cluster_df)

# Heading and explanation for the selected cluster's Residuals
col2.subheader("Step 3: Dig Deeper - Feature-Specific Retrofit Insights")
selected_feature = col2.selectbox('Select a feature for more retrofit details:', retrofit_features)

retrofit_details = {
    'Qinfil': {'Cost': 'High', 'Code': 'NECB (2017) - Section 3.2.4', 'Code_Limit': '≤ 0.2 L/(s·m²) at 75 Pa', 'Description': 'Improve the buildings air barrier by sealing gaps, cracks, and openings in the building envelope, including windows, doors, and other junctions. This is typically done using a combination of caulks, sealants, weatherstripping, and sometimes more comprehensive measures like air barrier membranes. Infiltration can be measured before and after retrofit using techniques like blower door tests to quantify the improvement. This measure usually complements other retrofit activities like improving Ropaque (opaque wall insulation) and Uwindow (window U-values), as a tighter building envelope can enhance the effectiveness of these other measures.'},
    'Ropaque': {
        'Cost': 'High',
        'Code': 'NECB 2017 Table 3.2.2.2',
        'Code_Limit': 'Zone 6: > 4.05 (m²·K)/W',
        'Description': 'Enhance the thermal performance of the building\'s opaque envelope components—primarily walls—by adding or upgrading insulation materials. Options include mineral wool, fiberglass, and foam boards, among others. The choice of material and thickness should be guided by local climate conditions, building use, and existing wall assembly. A higher thermal resistance (measured in W/(m²·K)) reduces the building\'s heating and cooling loads, leading to energy savings. This measure often complements efforts to reduce air infiltration (Qinfil), as a well-insulated and airtight building envelope maximizes energy efficiency. Pre- and post-retrofit thermal imaging can be useful for assessing effectiveness.'
    },
    'Uwindow': {'Cost': 'High', 'Code': 'NECB 2017 Table 3.2.2.3', 'Code_Limit': 'Zone 6: ≤ 1.9 W/(m²·K)', 'Description': 'Upgrade the windows to improve their thermal transmittance, commonly denoted as U-value. Double-pane windows filled with an insulating gas like argon can significantly reduce heat transfer, thereby lowering energy consumption for both heating and cooling. The choice of window type should be based on the buildings specific needs, climate zone, and the intended balance between daylighting and thermal performance. This retrofit not only affects the windows U-value but also its Solar Heat Gain Coefficient (SHGC) and air infiltration rates (Qinfil). Therefore, it often makes sense to address these features concurrently. Special coatings can also be applied to optimize for low SHGC or high visible transmittance, depending on the requirement. This intervention usually requires a higher upfront investment but offers substantial long-term energy savings and occupant comfort.'},
    'SHGC': {'Cost': 'High', 'Code': 'NECB 2017 Table 3.2.2.3', 'Description': 'Optimize the Solar Heat Gain Coefficient (SHGC) by either applying specialized reflective coatings to the existing windows or replacing them with high-performance glazing units designed for lower solar heat gain. The choice between coating and replacement should be based on factors like the age and condition of the existing windows, as well as the buildings cooling load requirements. Lowering the SHGC can lead to substantial energy savings in cooling-dominated climates or buildings with significant solar exposure. This retrofit is often carried out in conjunction with upgrades to the windows thermal transmittance (U-value) to maximize overall performance. Pre- and post-retrofit measurements should be taken to assess the effectiveness of this intervention.'},
    'Qventilation': {'Cost': 'Medium', 'Code': 'ASHRAE 62.1', 'Description': 'Adopt a Demand-Controlled Ventilation (DCV) system to optimize indoor air quality while minimizing energy consumption. DCV systems utilize CO2 and occupancy sensors to dynamically adjust the ventilation rates based on real-time conditions. This allows for sufficient fresh air supply during high occupancy while scaling back ventilation during low-use periods. This approach is especially effective in spaces with variable and unpredictable occupancy patterns, such as conference rooms or auditoriums. Implementing DCV can result in energy savings ranging from 10-30% in ventilation-related energy consumption, depending on the existing system and building usage. Initial calibration and periodic sensor maintenance are essential for sustained performance.', 'Threshold': 'Implement when ventilation rate is over 0.5 L/s*m^2.', 'Sensors Required': 'CO2 sensors, occupancy sensors', 'Expected Savings': '10-30% in ventilation-related energy consumption', 'Maintenance': 'Periodic calibration and sensor replacement'},
    'Qcasual': {'Cost': 'Medium', 'Code': 'NECB 2017 Table 4.2.1.5', 'Description': 'Integrate a multi-faceted approach to manage casual loads. Utilize daylighting strategies to offset electric lighting demand, employ task lighting to focus illumination where needed, and manage plug loads through smart power strips or centralized controls. Consider upgrading to energy-efficient T8 bulbs. These measures can yield a 5-20% reduction in casual load-related energy consumption.', 'Threshold': 'Implement when lighting and receptacle load exceed code limits.', 'Expected Savings': '5-20%', 'Maintenance': 'Periodic bulb replacement and plug load audits'},
    
    'Fafterhours': {'Cost': 'Low', 'Description': 'Deploy energy management systems that include lighting schedules and motion sensor activation to control after-hours energy consumption. These systems can automatically shut off or reduce lighting and plug loads, contributing to 5-15% energy savings.', 'Threshold': 'Implement when after-hours energy use is significantly high.', 'Expected Savings': '5-15%', 'Maintenance': 'Periodic calibration and sensor replacement'},
    
    'Mset up/down': {'Cost': 'Low', 'Code': 'N/A', 'Description': 'Modify thermostat setpoints to better align with occupancy patterns and external weather conditions. A 1°F adjustment can result in up to 10% energy savings. However, occupant comfort must be closely monitored.', 'Expected Savings': 'Up to 10%', 'Challenges': 'Occupant comfort adjustments', 'Duration': 'Immediate', 'Co-benefits': 'Quick energy savings'},
    
    'Tsa,clg': {'Cost': 'Low', 'Code': 'N/A', 'Description': 'Tweak cooling setpoints to run your cooling systems more efficiently. While maintaining indoor comfort, even a slight increase in the cooling setpoint can yield up to 5% in energy savings.', 'Expected Savings': 'Up to 5%', 'Challenges': 'Occupant comfort', 'Duration': 'Immediate', 'Co-benefits': 'Quick energy savings'},
    
    'Tsa,htg': {'Cost': 'Low', 'Code': 'N/A', 'Description': 'Adjust heating setpoints to optimize energy use. A minor decrease in the heating setpoint, carefully balanced with occupant comfort, can achieve up to 5% energy savings.', 'Expected Savings': 'Up to 5%', 'Challenges': 'Occupant comfort', 'Duration': 'Immediate', 'Co-benefits': 'Quick energy savings'},
    
    'Tsa,reset': {'Cost': 'Medium', 'Code': 'N/A', 'Description': 'Implement reset control strategies to dynamically adjust setpoints based on external and internal factors like outdoor temperature and occupancy. This can offer up to 20% energy savings but may require control system upgrades.', 'Expected Savings': 'Up to 20%', 'Challenges': 'Control system compatibility', 'Duration': '1 week', 'Co-benefits': 'Enhanced control'},
    
    'Tafterhours,htg': {'Cost': 'Low', 'Code': 'N/A', 'Description': 'Utilize timer controls to modulate heating systems during unoccupied hours. This can contribute to a 15% reduction in heating-related energy consumption after-hours.', 'Expected Savings': 'Up to 15%', 'Challenges': 'Fine-tuning required', 'Duration': '1 day', 'Co-benefits': 'Reduced energy waste'},
    
    'Tafterhours,clg': {'Cost': 'Low', 'Code': 'N/A', 'Description': 'Implement timer controls to regulate cooling systems during unoccupied hours, which can reduce cooling-related after-hours energy consumption by up to 15%.', 'Expected Savings': 'Up to 15%', 'Challenges': 'Fine-tuning required', 'Duration': '1 day', 'Co-benefits': 'Reduced energy waste'},
    
    'Fvav,min-sp': {'Cost': 'Medium', 'Code': 'ASHRAE 90.1', 'Description': 'Upgrade to high-performance Variable Air Volume (VAV) boxes to improve zonal control and system responsiveness. These upgrades can enhance energy efficiency by up to 25%, but may require system downtime for installation.', 'Expected Savings': 'Up to 25%', 'Challenges': 'System downtime', 'Duration': '1 week', 'Co-benefits': 'Improved zone control'},
    
    'Shtg,summer': {'Cost': 'Low', 'Code': 'N/A', 'Description': 'Disable or significantly reduce heating during the summer months. While simple, this strategy can offer quick energy savings of up to 5%. Ensure proper communication with occupants before implementation.', 'Expected Savings': 'Up to 5%', 'Challenges': 'Occupant communication', 'Duration': 'Immediate', 'Co-benefits': 'Quick energy savings'}


}

# Streamlit code
col2.subheader(f"Details for {selected_feature}")

if selected_feature in retrofit_details:
    col2.write(f"**Cost**: {retrofit_details[selected_feature]['Cost']}")
    col2.write(f"**Code Clause**: {retrofit_details[selected_feature].get('Code', 'N/A')}")
    col2.write(f"**What It Entails**: {retrofit_details[selected_feature]['Description']}")
    col2.write(f"**Expected Savings**: {retrofit_details[selected_feature].get('Expected Savings', 'N/A')}")
    col2.write(f"**Challenges**: {retrofit_details[selected_feature].get('Challenges', 'N/A')}")
    col2.write(f"**Duration**: {retrofit_details[selected_feature].get('Duration', 'N/A')}")
    col2.write(f"**Co-Benefits**: {retrofit_details[selected_feature].get('Co-benefits', 'N/A')}")
else:
    col2.write("No details available for this feature.")







