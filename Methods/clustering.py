import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import streamlit as st
from scipy.spatial.distance import cdist


def create_prediction_intervals(cluster_assignments, predictions_stack, building_features, limits, decimal_places):
    ##############################################################################
    # create prediction intervals for each cluster
    ##############################################################################

    # Initialize lists to hold the lower and upper bounds of the prediction intervals
    lower_bounds = []
    upper_bounds = []
    cluster_stds = []

    # Confidence level for prediction interval (e.g., 95%)
    alpha = 0.2
    z_value = stats.norm.ppf(1 - alpha / 2)  # Z-value for 80% confidence

    # Loop over each unique cluster
    for cluster in np.unique(cluster_assignments):
        # Get the predictions belonging to the current cluster
        cluster_predictions = predictions_stack[cluster_assignments == cluster]

        # Compute the mean and standard deviation of the predictions in the current cluster
        cluster_median = np.median(cluster_predictions, axis=0)
        cluster_std = np.std(cluster_predictions, axis=0)

        # Calculate the lower and upper bounds of the prediction interval
        cluster_lower_bound = cluster_median - z_value * cluster_std
        cluster_upper_bound = cluster_median + z_value * cluster_std
        
        # Debugging: Print raw values
        print(f"Cluster {cluster}")
        print(f"Raw Median: {cluster_median}")
        print(f"Raw Lower Bound: {cluster_lower_bound}")
        print(f"Raw Upper Bound: {cluster_upper_bound}")
        print(f"Standard Deviation: {cluster_std}")
        
        # Debugging: Print values after calculation but before clipping
        print(f"Calculated Lower Bound: {cluster_lower_bound}")
        print(f"Calculated Upper Bound: {cluster_upper_bound}")


        # Append the clipped and rounded lower and upper bounds to the respective lists
        lower_bounds.append(cluster_lower_bound)
        upper_bounds.append(cluster_upper_bound)
        # Append the standard deviations to the list
        cluster_stds.append(cluster_std)

    # Convert to DataFrames if desired
    lower_bounds_df = pd.DataFrame(lower_bounds, columns=building_features)
    if 'lower_bounds_df' not in st.session_state:
        st.session_state['lower_bounds_df'] = lower_bounds_df

    upper_bounds_df = pd.DataFrame(upper_bounds, columns=building_features)
    if 'upper_bounds_df' not in st.session_state:
        st.session_state['upper_bounds_df'] = upper_bounds_df
        
    # Convert to DataFrame if desired
    cluster_stds_df = pd.DataFrame(cluster_stds, columns=building_features)
    if 'cluster_stds_df' not in st.session_state:
        st.session_state['cluster_stds_df'] = cluster_stds_df
        
    


def create_median_dataframe(medians, building_features):
    # Create DataFrame from the medians
    medians_df_original_scale = pd.DataFrame(medians, columns=building_features)
    return medians_df_original_scale


def create_medians(cluster_assignments, predictions_stack):
    # Initialize an empty list to hold the median of each cluster
    medians = []

    # Loop over each unique cluster
    for cluster in np.unique(cluster_assignments):
        # Get the predictions belonging to the current cluster
        cluster_predictions = predictions_stack[cluster_assignments == cluster]

        # Compute the median of the predictions in the current cluster
        cluster_median = np.median(cluster_predictions, axis=0)

        # Add the median to the list of medians
        medians.append(cluster_median)

    st.session_state['medians'] = medians
    return medians


def apply_kmeans(KMeans, optimal_clusters, predictions_normalized):
    # Apply K-means
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(predictions_normalized)
    st.session_state['kmeans'] = kmeans
    return kmeans


def find_elbow(distortions):
    # Get the absolute values of the slopes
    slopes = [abs(distortions[i + 1] - distortions[i]) for i in range(len(distortions) - 1)]

    # Get the differences between consecutive slopes
    diffs = [slopes[i + 1] - slopes[i] for i in range(len(slopes) - 1)]

    # The optimal value is the index where the difference between slopes is the highest
    optimal_clusters = np.argmax(diffs) + 2  # +2 because the first index is for 2 clusters

    st.session_state['optimal_clusters'] = optimal_clusters
    clusters = list(range(1, optimal_clusters + 1))

    return optimal_clusters


def create_distortions(KMeans, predictions_normalized):
    # Elbow Method
    distortions = []
    K = range(1, 10)  # change according to your data
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(predictions_normalized)
        distortions.append(
            sum(np.min(cdist(predictions_normalized, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) /
            predictions_normalized.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    st.session_state['distortions'] = distortions
    st.session_state['K'] = K

    return distortions, K


def define_cluster_columns(medians_df_original_scale):
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
    # Append the unit to the feature name
    medians_df_original_scale.columns = [f'{col} ({units[col]})' for col in medians_df_original_scale.columns]

    st.session_state['medians_df_original_scale'] = medians_df_original_scale

