import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st


def create_x_inputs(CPM):
    # You can create a Series or DataFrame from CPM
    CPM_df = pd.DataFrame(CPM)

    change_point_data = ['Op_htg_balanceTemp', 'Op_htg_Slope', 'Op_htg_Intercept',
                         'Af_htg_balanceTemp', 'Af_htg_Slope', 'Af_htg_Intercept',
                         'Op_Clg_balanceTemp', 'Op_Clg_Slope', 'Op_Clg_Intercept',
                         'Af_Clg_balanceTemp', 'Af_Clg_Slope', 'Af_Clg_Intercept',
                         'R2_Op_Htg', 'R2_Af_Htg', 'R2_Op_Clg', 'R2_Af_Clg',
                         'CVRMSE_Op_Htg', 'CVRMSE_Af_Htg', 'CVRMSE_Op_Clg', 'CVRMSE_Af_Clg']

    # Convert CPM to a DataFrame with a single row
    CPM_df = pd.DataFrame([CPM])
    # Assign column names to the dataframe
    CPM_df.columns = change_point_data

    columns_to_drop = ['R2_Af_Htg', 'R2_Af_Clg', 'CVRMSE_Af_Htg',
                       'CVRMSE_Af_Clg']  # Replace with your desired column names
    CPM_df1 = CPM_df.drop(columns=columns_to_drop)

    # Transpose the DataFrame
    CPM_df = CPM_df.T

    # Prepare regressed CPM parameters to be scaled
    # Convert the list to a Numpy array
    X_inputs = np.array(CPM_df1, dtype=np.float64)
    # replace NaN values with 0
    X_inputs[np.isnan(X_inputs)] = 0

    st.session_state['X_inputs'] = X_inputs
    st.session_state['CPM_df'] = CPM_df

    return X_inputs, CPM_df

def afterhours_cooling(energyAf_Tout, tChp, energyAf):
    # Sweep through the change-point temperatures to find the best operational change point
    RMSEAfClg = np.zeros((25, 1))
    R2AfClg = np.zeros((25, 1))
    CVRMSE = np.zeros((25, 1))

    for j in range(25):
        regressorsAfClg = np.multiply(energyAf_Tout > tChp[j], energyAf_Tout - tChp[j])
        mdlAfClg = LinearRegression().fit(regressorsAfClg.reshape(-1, 1),
                                          energyAf['Cooling (W/m²)'].values.reshape(-1, 1))  # convert to numpy array
        R2AfClg[j, 0] = mdlAfClg.score(regressorsAfClg.reshape(-1, 1),
                                       energyAf['Cooling (W/m²)'].values.reshape(-1, 1))  # convert to numpy array
        predAfClg = mdlAfClg.predict(regressorsAfClg.reshape(-1, 1))
        RMSEAfClg[j, 0] = np.sqrt(np.mean((energyAf['Cooling (W/m²)'].values.reshape(-1, 1) - predAfClg) ** 2))
        CVRMSE[j] = RMSEAfClg[j, 0] / np.mean(energyAf['Cooling (W/m²)'])

    valAfClg = np.min(CVRMSE)
    AfClgTbal = np.argmin(CVRMSE)
    AfClgTbal = np.argmin(CVRMSE)
    R2AfClg = np.max(R2AfClg)

    # Construct the model only for the selected change-point temperature
    regressorsAfClg = np.multiply(energyAf['Dry Bulb Temperature (\u00b0C)'] > tChp[AfClgTbal],
                                  energyAf['Dry Bulb Temperature (\u00b0C)'] - tChp[AfClgTbal])
    mdlAfClg = LinearRegression().fit(regressorsAfClg.to_numpy().reshape(-1, 1),
                                      energyAf['Cooling (W/m²)'].to_numpy().reshape(-1, 1))  # convert to numpy array

    # Generate a wider range of temperatures for better plotting
    tOutTest = np.arange(-20, 30)
    regressorsAfClg = np.multiply(tOutTest > tChp[AfClgTbal], tOutTest - tChp[AfClgTbal])
    fitAfClg = mdlAfClg.predict(regressorsAfClg.reshape(-1, 1))
    AfClgYint = np.min(fitAfClg[fitAfClg > 0])
    AfClgYint_float = float(AfClgYint)  # Convert to float
    AfClgSlope = mdlAfClg.coef_[0, 0]
    AfClgSlope = (np.max(fitAfClg) - AfClgYint_float) / (30 - AfClgTbal)

    # Use the RMSE value corresponding to the selected change-point temperature
    RMSEAfClg = RMSEAfClg[AfClgTbal]
    # Calculate CVRMSE (Coefficient of Variation of Root Mean Square Error)
    CVRMSEAfClg = RMSEAfClg / np.mean(energyAf['Cooling (W/m²)']) * 100

    st.session_state['AfClgTbal'] = AfClgTbal
    st.session_state['AfClgSlope'] = AfClgSlope
    st.session_state['AfClgYint'] = AfClgYint
    st.session_state['R2AfClg'] = R2AfClg
    st.session_state['CVRMSEAfClg'] = CVRMSEAfClg
    st.session_state['fitAfClg'] = fitAfClg

    return AfClgTbal, AfClgSlope, AfClgYint, R2AfClg, CVRMSEAfClg, fitAfClg


def afterhours_heating(energyAf_Tout, tChp, energyAf):
    # Sweep through the change-point temperatures to find the best operational change point
    RMSEAfHtg = np.zeros((25, 1))
    R2AfHtg = np.zeros((25, 1))
    CVRMSE = np.zeros((25, 1))

    for j in range(25):
        regressorsAfHtg = np.multiply(energyAf_Tout < tChp[j], tChp[j] - energyAf_Tout)  # Corrected sign
        mdlAfHtg = LinearRegression().fit(regressorsAfHtg.reshape(-1, 1),
                                          energyAf['Heating (W/m²)'].values.reshape(-1, 1))
        R2AfHtg[j, 0] = mdlAfHtg.score(regressorsAfHtg.reshape(-1, 1), energyAf['Heating (W/m²)'].values.reshape(-1, 1))
        predAfHtg = mdlAfHtg.predict(regressorsAfHtg.reshape(-1, 1))
        RMSEAfHtg[j, 0] = np.sqrt(np.mean((energyAf['Heating (W/m²)'].values.reshape(-1, 1) - predAfHtg) ** 2))
        CVRMSE[j] = RMSEAfHtg[j, 0] / np.mean(energyAf['Heating (W/m²)'])

    valAfHtg = np.min(CVRMSE)
    AfHtgTbal = np.argmin(CVRMSE)
    R2AfHtg = np.max(R2AfHtg)

    # Construct the model only for the selected change-point temperature
    regressorsAfHtg = np.multiply(energyAf['Dry Bulb Temperature (°C)'] < tChp[AfHtgTbal],
                                  tChp[AfHtgTbal] - energyAf['Dry Bulb Temperature (°C)'])  # Corrected sign
    mdlAfHtg = LinearRegression().fit(regressorsAfHtg.to_numpy().reshape(-1, 1),
                                      energyAf['Heating (W/m²)'].to_numpy().reshape(-1, 1))

    # Generate a wider range of temperatures for better plotting
    tOutTest = np.arange(-20, 30)
    regressorsAfHtg = np.multiply(tOutTest < tChp[AfHtgTbal], tChp[AfHtgTbal] - tOutTest)  # Corrected sign
    fitAfHtg = mdlAfHtg.predict(regressorsAfHtg.reshape(-1, 1))
    AfHtgYint = np.min(fitAfHtg[fitAfHtg > 0])
    AfHtgYint_float = float(AfHtgYint)
    AfHtgSlope = mdlAfHtg.coef_[0, 0]

    # Use the RMSE value corresponding to the selected change-point temperature
    RMSEAfHtg = RMSEAfHtg[AfHtgTbal]
    # Calculate CVRMSE (Coefficient of Variation of Root Mean Square Error)
    CVRMSEAfHtg = RMSEAfHtg / np.mean(energyAf['Heating (W/m²)']) * 100

    st.session_state['AfHtgTbal'] = AfHtgTbal
    st.session_state['AfHtgSlope'] = AfHtgSlope
    st.session_state['AfHtgYint'] = AfHtgYint
    st.session_state['R2AfHtg'] = R2AfHtg
    st.session_state['CVRMSEAfHtg'] = CVRMSEAfHtg
    st.session_state['fitAfHtg'] = fitAfHtg

    return AfHtgTbal, AfHtgSlope, AfHtgYint, R2AfHtg, CVRMSEAfHtg, fitAfHtg


def operational_cooling(energyOp_Tout, tChp, energyOp):
    # Sweep through the change-point temperatures to find the best operational change point
    CVRMSE = np.zeros((25, 1))
    RMSEOpClg = np.zeros((25, 1))
    R2OpClg = np.zeros((25, 1))

    for j in range(25):
        regressorsOpClg = np.multiply(energyOp_Tout > tChp[j], energyOp_Tout - tChp[j])
        mdlOpClg = LinearRegression().fit(regressorsOpClg.reshape(-1, 1),
                                          energyOp['Cooling (W/m²)'].values.reshape(-1, 1))  # convert to numpy array
        R2OpClg[j, 0] = mdlOpClg.score(regressorsOpClg.reshape(-1, 1),
                                       energyOp['Cooling (W/m²)'].values.reshape(-1, 1))  # convert to numpy array
        predOpClg = mdlOpClg.predict(regressorsOpClg.reshape(-1, 1))
        RMSEOpClg[j, 0] = np.sqrt(np.mean((energyOp['Cooling (W/m²)'].values.reshape(-1, 1) - predOpClg) ** 2))
        CVRMSE[j] = RMSEOpClg[j, 0] / np.mean(energyOp['Cooling (W/m²)'])

    valOpClg = np.min(CVRMSE)
    OpClgTbal = np.argmin(CVRMSE)
    R2OpClg = np.max(R2OpClg)

    # Construct the model only for the selected change-point temperature
    regressorsOpClg = np.multiply(energyOp['Dry Bulb Temperature (\u00b0C)'] > tChp[OpClgTbal],
                                  energyOp['Dry Bulb Temperature (\u00b0C)'] - tChp[OpClgTbal])
    mdlOpClg = LinearRegression().fit(regressorsOpClg.to_numpy().reshape(-1, 1),
                                      energyOp['Cooling (W/m²)'].to_numpy().reshape(-1, 1))  # convert to numpy array

    # Generate a wider range of temperatures for better plotting
    tOutTest = np.arange(-20, 30)
    regressorsOcClg = np.multiply(tOutTest > tChp[OpClgTbal], tOutTest - tChp[OpClgTbal])
    fitOpClg = mdlOpClg.predict(regressorsOcClg.reshape(-1, 1))
    OpClgYint = np.min(fitOpClg[fitOpClg > 0])
    OpClgSlope = mdlOpClg.coef_[0, 0]

    # Use the RMSE value corresponding to the selected change-point temperature
    RMSEOpClg = RMSEOpClg[OpClgTbal]
    # Calculate CVRMSE (Coefficient of Variation of Root Mean Square Error)
    CVRMSEOpClg = RMSEOpClg / np.mean(energyOp['Cooling (W/m²)']) * 100

    st.session_state['OpClgTbal'] = OpClgTbal
    st.session_state['OpClgSlope'] = OpClgSlope
    st.session_state['OpClgYint'] = OpClgYint
    st.session_state['R2OpClg'] = R2OpClg
    st.session_state['CVRMSEOpClg'] = CVRMSEOpClg
    st.session_state['fitOpClg'] = fitOpClg

    return OpClgTbal, OpClgSlope, OpClgYint, R2OpClg, CVRMSEOpClg, fitOpClg


def operational_heating(energyOp_Tout, tChp, energyOp):
    # Sweep through the change-point temperatures to find the best operational change point
    RMSEOpHtg = np.zeros((25, 1))
    R2OpHtg = np.zeros((25, 1))
    CVRMSE = np.zeros((25, 1))

    for j in range(25):
        regressorsOpHtg = np.multiply(energyOp_Tout < tChp[j], tChp[j] - energyOp_Tout)
        mdlOpHtg = LinearRegression().fit(regressorsOpHtg.reshape(-1, 1),
                                          energyOp['Heating (W/m²)'].values.reshape(-1, 1))  # convert to numpy array
        R2OpHtg[j, 0] = mdlOpHtg.score(regressorsOpHtg.reshape(-1, 1),
                                       energyOp['Heating (W/m²)'].values.reshape(-1, 1))  # convert to numpy array
        predOpHtg = mdlOpHtg.predict(regressorsOpHtg.reshape(-1, 1))
        RMSEOpHtg[j, 0] = np.sqrt(np.mean((energyOp['Heating (W/m²)'].values.reshape(-1, 1) - predOpHtg) ** 2))
        CVRMSE[j] = RMSEOpHtg[j, 0] / np.mean(energyOp['Heating (W/m²)'])

    valOpHtg = np.min(CVRMSE)
    OpHtgTbal = np.argmin(CVRMSE)
    R2OpHtg = np.max(R2OpHtg)

    # Construct the model only for the selected change-point temperature
    regressorsOpHtg = np.multiply(energyOp['Dry Bulb Temperature (°C)'] < tChp[OpHtgTbal],
                                  tChp[OpHtgTbal] - energyOp['Dry Bulb Temperature (°C)'])
    mdlOpHtg = LinearRegression().fit(regressorsOpHtg.to_numpy().reshape(-1, 1),
                                      energyOp['Heating (W/m²)'].to_numpy().reshape(-1, 1))

    # Generate a wider range of temperatures for better plotting
    tOutTest = np.arange(-20, 30)
    regressorsOpHtg = np.multiply(tOutTest < tChp[OpHtgTbal], tChp[OpHtgTbal] - tOutTest)
    fitOpHtg = mdlOpHtg.predict(regressorsOpHtg.reshape(-1, 1))
    OpHtgYint = np.min(fitOpHtg[fitOpHtg > 0])
    OpHtgSlope = mdlOpHtg.coef_[0, 0]

    # Use the RMSE value corresponding to the selected change-point temperature
    RMSEOpHtg = RMSEOpHtg[OpHtgTbal]
    # Calculate CVRMSE (Coefficient of Variation of Root Mean Square Error)
    CVRMSEOpHtg = RMSEOpHtg / np.mean(energyOp['Heating (W/m²)']) * 100

    st.session_state['OpHtgTbal'] = OpHtgTbal
    st.session_state['OpHtgSlope'] = OpHtgSlope
    st.session_state['OpHtgYint'] = OpHtgYint
    st.session_state['R2OpHtg'] = R2OpHtg
    st.session_state['CVRMSEOpHtg'] = CVRMSEOpHtg
    st.session_state['fitOpHtg'] = fitOpHtg

    return OpHtgTbal, OpHtgSlope, OpHtgYint, R2OpHtg, CVRMSEOpHtg, fitOpHtg


def generate_variables(energyOp, energyAf):
    tChp = np.arange(1, 27)
    tOutTest = np.arange(-20, 30)
    energyOp_Tout = energyOp['Dry Bulb Temperature (\u00b0C)'].to_numpy()  # convert to numpy array
    energyAf_Tout = energyAf['Dry Bulb Temperature (\u00b0C)'].to_numpy()  # convert to numpy array

    st.session_state['tChp'] = tChp
    st.session_state['tOutTest'] = tOutTest
    st.session_state['energyOp_Tout'] = energyOp_Tout
    st.session_state['energyAf_Tout'] = energyAf_Tout

    return tChp, tOutTest, energyOp_Tout, energyAf_Tout
