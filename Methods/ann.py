import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model


def load_ann_models(wwr, aspect_ratio):
    # Map aspect ratios to their respective folder names
    ar_folder_map = {
        '1:1': 'AR1',
        '1.75:0.57': 'AR2',
        '2.0:0.5': 'AR3'
    }

    # Get the correct subfolder for the given aspect ratio
    ar_folder = ar_folder_map.get(aspect_ratio)

    if ar_folder is None:
        st.error(f'Invalid aspect ratio: {aspect_ratio}')
        return None

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Construct the path to the models directory
    models_dir = os.path.join(script_dir, '..', 'Models', f'wwr{wwr}', ar_folder, 'models')
    scaler_dir = os.path.join(script_dir, '..', 'Models', f'wwr{wwr}', ar_folder)

    def custom_mse(feature_index, feature_name):
        def mse(y_true, y_pred):
            return tf.keras.metrics.mean_squared_error(y_true[:, feature_index], y_pred[:, feature_index])

        mse.__name__ = f"mse_{feature_name}"
        return mse

    # Define the features of interest
    features_of_interest = ['Uwindow', 'SHGC', 'Ropaque', 'Qinfil', 'Qventilation', 'Qcasual']

    # Define a dictionary to hold the custom metrics
    custom_metrics = {f'mse_{feature}': custom_mse(i, feature) for i, feature in enumerate(features_of_interest)}

    ann_models = []
    for i in range(1, 101):
        model_path = os.path.join(models_dir, f"model{i}.h5")  # use models_dir instead of SAVE_DIR
        model = load_model(model_path, custom_objects=custom_metrics)
        ann_models.append(model)

    # Make sure the models are loaded correctly
    if ann_models is None:
        st.error('Failed to load ANN models.')
        st.stop()

    st.session_state['ann_models'] = ann_models
    st.session_state['scaler_dir'] = scaler_dir

    return ann_models, scaler_dir  # Return the directory of the scaler instead


def load_scaler(scaler_dir):
    # Load the scaler object from the file
    scaler_path = os.path.join(scaler_dir, 'data_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    st.session_state['scaler'] = scaler

    return scaler


def scale_input_values(scaler, X_inputs):
    # Scale the input values using the loaded scaler
    X_inputs_norm = scaler.transform(X_inputs.reshape(1, -1))

    st.session_state['X_inputs_norm'] = X_inputs_norm

    return X_inputs_norm


def make_predictions(X_inputs_norm, ann_models):
    # Make predictions with the individual models
    predictions = [model.predict(X_inputs_norm) for model in ann_models]

    st.session_state['predictions'] = predictions

    return predictions


def stack_predictions(predictions, building_features):
    # Stack predictions
    predictions_stack = np.vstack(predictions)
    df = pd.DataFrame(predictions_stack, columns=building_features)

    st.session_state['predictions_stack'] = predictions_stack
    st.session_state['df'] = df

    return predictions_stack, df
