# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:25:15 2023

@author: Shane
"""

#########################################################
# Import Libraries
#########################################################

import pandas as pd
import streamlit as st
from PIL import Image

import datetime
from datetime import datetime
from sklearn.linear_model import LinearRegression
import sys
import base64
from base64 import b64encode
import os

###########################################################
# Page Layout
###########################################################

st.set_page_config(
    page_title="Building Features Characterization",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={})

# os.chdir(r'C:\Users\Shane\Desktop\App')

########################################################
# Page Title
########################################################
# Set the image path
# image_path = r"C:\\Users\Shane\Desktop\MasterThesis\Database\App\Archetypes\dream_00e716mnkdh.jpg"
# image = Image.open('dream_00e716mnkdh.jpg')
# Display the image as a banner with full width
# st.image(image, use_column_width=True)


st.title('Building Feature Characterization Tool')
st.subheader('A Novel Approach to Building Energy Characterization')


st.markdown("""
The Building Feature Characterization Tool is a state-of-the-art application designed to provide insights 
into a building's energy use behavior using heating and cooling load signatures. 
This tool leverages the power of Artificial Neural Networks (ANNs) to predict energy-related building features, 
offering a scalable and efficient approach for identifying retrofit opportunities.
 """)

# Methodology Section
st.header('Methodology:')


st.markdown("""
1. **Base Model Generation:**
   - Different mid-rise office archetypes are created to represent various mid-rise office building types, considering 
   window-to-wall ratios (WWRs) and aspect ratios.
   
2. **Base Model Variants:**
   - The energy-related features (envelope, internal gains, and HVAC operation) are randomly sampled into feature sets, 
   and simulations are performed using EnergyPlus to extract heating and cooling load signatures.
2. **Three-Parameter Change Point Models (3P CPMs):**
   - Heating and cooling load signatures are characterized by 3P CPMs, serving as indicators of energy use behavior.
3. **ANN-Based Surrogate Modeling:**
   - A specialized ANN model is trained to map 3P CPMs to plausible energy-related feature sets.
   - The ANN model consists of multiple layers, including input, hidden, and output layers, fine-tuned to achieve 
   optimal performance.
4. **Validation and Testing:**
   - The model is validated against real-world case studies, ensuring its reliability and accuracy.
""")

st.image('MethodWorkflow2.png', use_column_width=True)


# Inputs Section
st.header('Inputs:')
st.markdown("""
- **Archetype Configuration:** Information related to the building's aspect ratio and WWR configuration can further 
refine the predictions.
- **Heating and Cooling Load Signatures:** These are the primary inputs required to characterize the building's energy 
performance.
""")

# Outputs Section
st.header('Outputs:')
st.markdown("""
- **Probable Energy-Related Building Features:** The tool predicts a range of plausible feature sets: 
    - Thermal transittance: window U-values, SHGC, wall R-values 
    - Air exchange: infiltration rate and ventilation rate
    - Casual gains: lighting and plug load density (LPD) and the fraction of afterhour LPD use
    - A range of HVAC operation and control and features
- **Recommendations for Retrofit Opportunities:** Based on the predicted features, the tool can guide building energy 
managers and owners 
toward potential retrofit interventions.
""")

# Application Section
st.header('Application:')
st.markdown("""
This tool offers a user-friendly interface for building professionals seeking to understand and optimize 
energy performance. 
By integrating advanced machine learning techniques with building energy simulation, it provides a novel 
pathway for remote 
building feature characterization, informing energy efficiency strategies, and supporting sustainable 
building practices.
""")

# References Section
st.header('References:')
st.markdown("""
1. Ferreira, Shane; Gunay, H. Burak; Ashouri, Araz; and Shillinglaw, Scott. (2023). "Unsupervised learning of 
load signatures to estimate energy-related building features using surrogate modelling techniques." 
Building Simulation, 16:1273 – 1286.
2. Ferreira, Shane; Gunay, H. Burak; Wills, Adam; and Shillinglaw, Scott. "Neural network-based surrogate model to 
predict building features from heating and cooling load signatures." [Under Review].
""")

# Optional: Adding a Note
st.markdown("""
*Note:* The above references provide foundational theories, methodologies, or empirical studies that have 
contributed to the development of this tool. For a detailed understanding and further reading, 
please refer to these publications.
""")
