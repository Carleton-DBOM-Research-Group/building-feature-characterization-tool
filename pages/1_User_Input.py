# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:02:49 2023

@author: Shane
"""
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import os
import base64

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from Methods import benchmarks as bm
from Methods import plots as pl
from Methods import ann as ann
from Methods import clustering as cl
from Methods import histograms as hs
from Methods import linear_regression as lr
from Methods import weather as w
from Methods import energy as e

st.set_page_config(
    page_title="Building Features Characterization",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={ })

########################################################
# User inputs functions & session state
########################################################
# Add this line to initialize session_state
if "selected_image_caption" not in st.session_state:
    st.session_state["selected_image_caption"] = ""
    
def handle_image_click(caption,message):
        # Set the selected image caption in session_state
        st.session_state["selected_image_caption"] = caption
        # With this line
        st.success(message)
        

########################################################
# Archetypes Info
########################################################
st.title('Getting Started with Your Building Analysis')

# Display a header for selecting the archetype
with st.expander("Learn more about user inputs"):
    st.markdown('## Selecting an Archetype That Most Resembles Your Building')
    
    # Information about building type
    st.markdown("""
    These archetypes are designed specifically for rectangular mid to high-rise office buildings. Here's how to know if this is right for your building:
    - **Mid to High-Rise Buildings:** Perfect fit! You're in the right place.
    - **Slightly Smaller Buildings:** You can still use these archetypes if your building has slightly fewer storeys and a floor area greater than 3000 m². Just be aware that the results might not be as precise.
    - **Low Rise Buildings (less than 3 storeys):** Unfortunately, this method won't work for your buildings.
    """)
    
    # Guide on selecting the Window-to-Wall Ratio and Aspect Ratio
    st.markdown("""
    ## Selecting the Right Window-to-Wall Ratio and Aspect Ratio
    - **Window-to-Wall Ratio (WWR):** Choose an overall WWR between 20% and 80% that best represents your building's windows.
    - **Aspect Ratio (Length-to-Width Ratio):** Select based on your building's shape.
        - **1:1:** If your building is a square.
        - **2.0:0.5:** If your building is twice as long as it is wide.
        - **1.75:0.57:** Somewhere between a square and twice as long.
    - **Note:** These are for rectangular buildings only. Other shapes or types of buildings, such as hospitals or schools, are not included.
    """)
    
    # Instructions for Uploading Heating and Cooling Load Data
    st.markdown("""
    ## Uploading Your Heating and Cooling Load Data
    To accurately analyze your building's energy use, follow these guidelines:
    - **Format:** Data should be in kilowatts (kW) and cover 8760 hours (a full year).
    - **Match the Example:** Use the same format as our example CSV file.
    - **Accuracy Matters:** Ensuring your data is correct helps us provide accurate and reliable analysis.
    
    """)
    

# Create two columns
col1, col2 = st.columns(2)

# Floor area input in Column 2
with col1:
    st.header("1. Please input your building's conditioned floor area (m²)")
    floor_area = st.number_input("Floor area (m²)", format="%f")
    if floor_area <= 3000:
        st.error('Floor area must be greater than 3000 m² ')
    else:
        st.session_state['floor_area'] = floor_area
        st.success("Floor area is valid")
        
# Download example energy use CSV and File Uploader in Column 1
with col2:
    st.header('2. Upload your buildings hourly heating and cooling energy use')
    uploaded_file = st.file_uploader("Upload energy use CSV here")
    # Providing an Example or Link to Download Template
    ExEnergy = pd.read_csv("Example Energy Data.csv")
    csv = ExEnergy.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="example_data.csv">Download CSV</a>'
    st.download_button(label='Download example energy use CSV', data=csv, file_name='example_data.csv', mime='text/csv')


    if uploaded_file is not None:   
        energy_use_df = pd.read_csv(uploaded_file)
    
        # Validate uploaded CSV against required conditions
        if 'Time' not in energy_use_df.columns:
            st.warning("The uploaded file does not contain a 'Time' column")
        elif 'Heating (kW)' not in energy_use_df.columns:
            st.warning("The uploaded file does not contain a 'Heating (kW)' column")
        elif 'Cooling (kW)' not in energy_use_df.columns:
            st.warning("The uploaded file does not contain a 'Cooling (kW)' column")
        elif len(energy_use_df) != 8760:
            st.warning("The uploaded file does not contain 8760 rows of data")
        else:

            energy_use_df.loc[:, ['Cooling (kW)', 'Heating (kW)']] = (energy_use_df.loc[:, ['Cooling (kW)', 'Heating (kW)']] ) *277.77778/ floor_area
            energy_use_df = energy_use_df.rename(columns={'Cooling (kW)': 'Cooling (W/m²)', 'Heating (kW)': 'Heating (W/m²)'}) 
    
            st.success("Energy use CSV file succesfully uploaded" )
            



##############################################################################
# Define the function to display images
##############################################################################

def display_images(base_path, captions):
    # this function displays images in columns for given captions and base path
    col1, col2, col3 = st.columns(3)

    for i, caption in enumerate(captions, start=1):
        image_path = os.path.join(base_path, f'AR{i}.png')
        image = Image.open(image_path)

        with locals()[f'col{i}']:
            if st.button(caption):
                handle_image_click(caption, f"You selected a building archetype with an aspect ratio of {caption.split('|')[-1].strip()} and a WWR of {caption.split('|')[0].split('-')[1].strip()}")
            st.image(image, use_column_width=True, caption=caption)

# Start of the section for image selection
st.header('3. Please select an archetype')

##############################################################################
# Display three images side by side - first row of images (20% WWR)
##############################################################################

# define the base directory for images
image_base_path = os.path.join('Assets', 'wwr20')

st.subheader('20% window to wall ratio:')
captions_20 = [
    "WWR-20% | Aspect ratio - 1:1",
    "WWR-20% | Aspect ratio - 1.75:0.57",
    "WWR-20% | Aspect ratio - 2.0:0.5"
]
display_images(image_base_path, captions_20)


##############################################################################
# Display three images side by side - second row of images (40% WWR)
##############################################################################

# define the base directory for images
image_base_path = os.path.join('Assets', 'wwr40')

st.subheader('40% window to wall ratio:')
captions_40 = [
    "WWR-40% | Aspect ratio - 1:1",
    "WWR-40% | Aspect ratio - 1.75:0.57",
    "WWR-40% | Aspect ratio - 2.0:0.5"
]
display_images(image_base_path, captions_40)


##############################################################################
# Display three images side by side - second row of images (60% WWR)
##############################################################################

# define the base directory for images
image_base_path = os.path.join('Assets', 'wwr60')

st.subheader('60% window to wall ratio:')
captions_60 = [
    "WWR-60% | Aspect ratio - 1:1",
    "WWR-60% | Aspect ratio - 1.75:0.57",
    "WWR-60% | Aspect ratio - 2.0:0.5"
]
display_images(image_base_path, captions_60)


##############################################################################
# Display three images side by side - second row of images (80% WWR)
##############################################################################

# define the base directory for images
image_base_path = os.path.join('Assets', 'wwr80')

st.subheader('80% window to wall ratio:')
captions_80 = [
    "WWR-80% | Aspect ratio - 1:1",
    "WWR-80% | Aspect ratio - 1.75:0.57",
    "WWR-80% | Aspect ratio - 2.0:0.5"
]
display_images(image_base_path, captions_80)

########################################################
# Additional Building Details'
########################################################

st.header('4. Additional Building Details (optional)')

details_keys = ["building_name", "address_line1", "address_line2", "city", "year_of_construction", "number_of_floors","Postal_code"]
for key in details_keys:
    if key not in st.session_state:
        st.session_state[key] = "" if key not in ["year_of_construction", "number_of_floors"] else 0

st.session_state.building_name = st.text_input("Building Name:", st.session_state.building_name)
st.session_state.address_line1 = st.text_input("Address Line 1:", st.session_state.address_line1)
st.session_state.address_line2 = st.text_input("Address Line 2:", st.session_state.address_line2)
st.session_state.city = st.text_input("City:", st.session_state.city)
st.session_state.Postal_code = st.text_input("Postal code:", st.session_state.Postal_code)
st.session_state.year_of_construction = st.number_input("Year of Construction:", value=st.session_state.year_of_construction)
st.session_state.number_of_floors = st.number_input("Number of Floors:", value=st.session_state.number_of_floors)
caption = st.session_state.get('selected_image_caption')

########################################################
# User Input Validation
########################################################

if st.button("Please save your selections here before moving on to the results"):

    if not floor_area:
        st.warning("Please input the floor area")
    elif not st.session_state.get("selected_image_caption"):
        st.warning("Please select an archetype")
    elif energy_use_df is None or energy_use_df.empty:
        st.warning("Please upload the energy use data")
    else:
        # Display a loading message
        st.markdown("### Analyzing Your Building's Data...")
        st.markdown("This may take a minute or two. Please be patient.")
        # Create a progress bar
        progress_bar = st.progress(0)
        # Simulate progress (replace this with the actual progress of your analysis)
        for percent_complete in range(100):
            # Update the progress bar
            progress_bar.progress(percent_complete + 1)
            # You can adjust the time delay to match the expected duration of the analysis
            time.sleep(0.1)
        st.session_state['floor_area'] = floor_area
        st.session_state['energy_use_df'] = energy_use_df.to_json()
        st.session_state['selected_image'] = st.session_state.get("selected_image_caption")
        if 'wwr' not in st.session_state:
            st.session_state['wwr'] = caption.split("WWR-")[1].split("%")[0]
        wwr = st.session_state['wwr']
        if 'aspect_ratio' not in st.session_state:
            st.session_state['aspect_ratio'] = caption.split("Aspect ratio - ")[1].split("|")[0]
        aspect_ratio = st.session_state['aspect_ratio']
        ###################################################
        ######################################################################
        # read in weather data & User's uploaded energy use data
        ######################################################################
        # Check if the weather data is already in the session state
        weather = w.load_weather()
        if 'weather' not in st.session_state:
            # Load the weather data
            # Cache the weather data in the session state
            st.session_state['weather'] = weather

        if weather is not None:
            Tout = weather['Dry Bulb Temperature {C}']

        # Save the DataFrame to session state
        st.session_state['energy_use_df'] = energy_use_df.to_json()

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
        # Separate the data into heating periods
        heating_months = [1, 2, 3, 4, 10, 11, 12]
        heating_data = User_Energy[User_Energy.index.month.isin(heating_months)]

        # Add 'day' column
        heating_data['day'] = heating_data.index.date

        # Add 'day_of_week' column
        heating_data['day_of_week'] = heating_data.index.dayofweek

        # Separate the data into weekdays and weekends
        weekday_data = heating_data[heating_data['day_of_week'].isin(range(5))]  # Monday=0, Tuesday=1, ..., Friday=4
        weekend_data = heating_data[heating_data['day_of_week'].isin([5, 6])]  # Saturday=5, Sunday=6

        # Get a list of all unique days
        unique_days = heating_data['day'].unique()

        weekday_hours = e.get_all_operating_hours(weekday_data)

        weekend_hours = e.get_all_operating_hours(weekend_data)

        weekday_hours = e.convert_operating_hours_to_hours(weekday_hours)
        st.session_state['weekday_hours'] = weekday_hours
        weekend_hours = e.convert_operating_hours_to_hours(weekend_hours)
        st.session_state['weekend_hours'] = weekend_hours

        weekday_hours_mode = e.get_mode_of_hours(weekday_hours)
        st.session_state['weekday_hours_mode'] = weekday_hours_mode
        weekend_hours_mode = e.get_mode_of_hours(weekend_hours)
        st.session_state['weekend_hours_mode'] = weekend_hours_mode

        start_modes_weekdays, end_modes_weekdays, start_modes_weekends, end_modes_weekends = e.convert_mode_to_datetime(
            weekday_hours_mode, weekend_hours_mode)

        start_modes_weekdays_str, end_modes_weekdays_str, start_modes_weekends_str, end_modes_weekends_str = e.convert_mode_to_string(
            start_modes_weekdays, end_modes_weekdays, start_modes_weekends, end_modes_weekends)

        energyOp_weekdays, energyOp_weekends, energyOp = e.generate_operational_energy(User_Energy,
                                                                                       start_modes_weekdays_str,
                                                                                       end_modes_weekdays_str,
                                                                                       start_modes_weekends_str,
                                                                                       end_modes_weekends_str)
        energyAf_weekdays, energyAf_weekends, energyAf = e.generate_afterhours_energy(User_Energy,
                                                                                      end_modes_weekdays_str,
                                                                                      start_modes_weekdays_str)

        #################################################################
        # Piecewise Linear Regression
        #################################################################
        # Generate variables
        tChp, tOutTest, energyOp_Tout, energyAf_Tout = lr.generate_variables(energyOp, energyAf)
        #######################################################################################################################################
        # For operational heating
        #########################################################################################################################################
        # Sweep through the change-point temperatures to find the best operational change point
        OpHtgTbal, OpHtgSlope, OpHtgYint, R2OpHtg, CVRMSEOpHtg, fitOpHtg = lr.operational_heating(energyOp_Tout, tChp,
                                                                                                  energyOp)

        ##########################################################################################################################################
        # For operational cooling
        ##########################################################################################################################################
        # Sweep through the change-point temperatures to find the best operational change point
        OpClgTbal, OpClgSlope, OpClgYint, R2OpClg, CVRMSEOpClg, fitOpClg = lr.operational_cooling(energyOp_Tout, tChp,
                                                                                                  energyOp)

        ##########################################################################################################################################
        # For afterhours heating
        ##########################################################################################################################################
        # Sweep through the change-point temperatures to find the best operational change point
        AfHtgTbal, AfHtgSlope, AfHtgYint, R2AfHtg, CVRMSEAfHtg, fitAfHtg = lr.afterhours_heating(energyAf_Tout, tChp,
                                                                                                 energyAf)

        ##########################################################################################################################################
        # For afterhours cooling
        ##########################################################################################################################################
        # Sweep through the change-point temperatures to find the best operational change point
        AfClgTbal, AfClgSlope, AfClgYint, R2AfClg, CVRMSEAfClg, fitAfClg = lr.afterhours_cooling(energyAf_Tout, tChp,
                                                                                                 energyAf)

        ##########################################################################################################################################
        # Store regressed parameters in a df
        ##########################################################################################################################################

        # Store the rergressed parameters in a list
        CPM = [OpHtgTbal + 1, -1 * OpHtgSlope, OpHtgYint, AfHtgTbal + 1, -1 * AfHtgSlope, AfHtgYint,
               OpClgTbal + 1, OpClgSlope, OpClgYint, AfClgTbal + 1, AfClgSlope, AfClgYint,
               R2OpHtg, R2AfHtg, R2OpClg, R2AfClg,
               CVRMSEOpHtg[0], CVRMSEAfHtg[0], CVRMSEOpClg[0], CVRMSEAfClg[0]]

        # Prepare regressed CPM parameters to be scaled
        X_inputs, CPM_df = lr.create_x_inputs(CPM)

        ##############################################################################
        # Load scaler and models and make predictions with CPM parameters
        ##############################################################################

        # Load the models
        ann_models, scaler_dir = ann.load_ann_models(wwr, aspect_ratio)

        # Load the scaler object from the file
        scaler = ann.load_scaler(scaler_dir)

        # Scale the input values using the loaded scaler
        X_inputs_norm = ann.scale_input_values(scaler, X_inputs)

        # Make predictions with the individual models
        predictions = ann.make_predictions(X_inputs_norm, ann_models)

        ##############################################################################
        # Stack predictions from ANN models, scale features, elbow method, k-means clustering, and take median of each cluster
        ##############################################################################
        building_features = ['Uwindow', 'SHGC', 'Ropaque', 'Qinfil', 'Qventilation', 'Qcasual', 'Fafterhours',
                             'Mset up/down', 'Tsa,clg', 'Tsa,htg', 'Tsa,reset', 'Tafterhours,htg', 'Tafterhours,clg',
                             'Fvav,min-sp', 'Shtg,summer']
        # Stack predictions
        predictions_stack, df = ann.stack_predictions(predictions, building_features)
        prediction_summary_stats = df.describe()

        # print the summary statistics of the stacked predictions
        print(prediction_summary_stats)

        # Normalize your data
        scaler = MinMaxScaler()
        predictions_normalized = scaler.fit_transform(predictions_stack)

        # Elbow Method
        distortions, K = cl.create_distortions(KMeans, predictions_normalized)

        # Find the optimal number of clusters
        optimal_clusters = cl.find_elbow(distortions)

        # Apply K-means
        kmeans = cl.apply_kmeans(KMeans, optimal_clusters, predictions_normalized)

        # Get the cluster assignments for each prediction
        cluster_assignments = kmeans.labels_

        medians = cl.create_medians(cluster_assignments, predictions_stack)

        medians_df_original_scale = cl.create_median_dataframe(medians, building_features)

        # Define the lower and upper limits for each feature
        limits = {
            'Uwindow': (1.5, 3.6),
            'SHGC': (0.3, 0.7),
            'Ropaque': (1.5, 5),
            'Qinfil': (0.1, 1),
            'Qventilation': (0.2, 2),
            'Qcasual': (3, 20),
            'Fafterhours': (0, 1),
            'Mset up/down': (0, 1),  # Assuming 'on' is 1 and 'off' is 0
            'Tsa,clg': (12, 15),
            'Tsa,htg': (15.1, 24),
            'Tsa,reset': (0, 1),  # Assuming 'constant' is 1 and 'variable' is 0
            'Tafterhours,htg': (15, 22),
            'Tafterhours,clg': (25, 30),
            'Fvav,min-sp': (0.1, 0.6),
            'Shtg,summer': (0, 1)  # Assuming 'on' is 1 and 'off' is 0
        }

        # Clip each feature to its respective range
#        medians_df_original_scale = cl.clip_feature_to_range(limits, medians_df_original_scale)

        # Define the number of decimal places for each feature
        decimal_places = {
            'Uwindow': 1,
            'SHGC': 1,
            'Ropaque': 1,
            'Qinfil': 4,
            'Qventilation': 4,
            'Qcasual': 1,
            'Fafterhours': 1,
            'Mset up/down': 0,
            'Tsa,clg': 0,
            'Tsa,htg': 0,
            'Tsa,reset': 0,
            'Tafterhours,htg': 0,
            'Tafterhours,clg': 0,
            'Fvav,min-sp': 1,
            'Shtg,summer': 0
        }

        # Round each feature to its respective decimal places
#        medians_df_original_scale = cl.round_decimal_places_for_feature(decimal_places, medians_df_original_scale)
        st.session_state['raw_medians'] = medians_df_original_scale

        ##############################################################################
        # create prediction intervals for each cluster
        ##############################################################################
        cl.create_prediction_intervals(cluster_assignments, predictions_stack, building_features, limits,
                                       decimal_places)
        ###################################################

        cl.define_cluster_columns(medians_df_original_scale)

        st.success("Please proceed to the Results page")













    
