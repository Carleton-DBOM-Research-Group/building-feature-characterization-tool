from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def generate_histograms(processed_data, feature_names, feature_units):
    fig, axs = plt.subplots(2, 3, figsize=(10, 8), sharex=False, sharey=True)

    for i, (values_with_residuals, mean_error, std_error, lower_bound, upper_bound) in enumerate(processed_data):
        ax = plt.subplot(2, 3, i+1)
        sns.histplot(values_with_residuals, kde=True, alpha=0.5, ax=ax)

        # Set the range of the x-axis to be within three standard deviations from the mean
        xlim_lower = mean_error - 3 * std_error
        xlim_upper = mean_error + 3 * std_error
        ax.set_xlim(xlim_lower, xlim_upper)

        # Add confidence interval lines to the plot
        ax.axvline(x=lower_bound, color='red', linestyle='--')
        ax.axvline(x=upper_bound, color='blue', linestyle='--')
        ax.axvline(x=mean_error, color='green', linestyle='--')

        # Set xticks to correspond to the 80% CI lines
        ax.set_xticks([lower_bound, upper_bound])

        ax.set_xlabel('{}'.format(feature_units[feature_names[i]]))
        ax.set_ylabel('')
        ax.set_title('{}\n$\mu$: {:.2f}, $\sigma$: {:.2f}'.format(feature_names[i], mean_error, std_error))

    # Create custom legend
    red_patch = mpatches.Patch(color='red', linestyle='--', label='80% CI lower bound')
    blue_patch = mpatches.Patch(color='blue', linestyle='--', label='80% CI upper bound')
    green_patch = mpatches.Patch(color='green', linestyle='--', label='Cluster prediction')
    plt.legend(handles=[red_patch, green_patch, blue_patch], loc='upper center', bbox_to_anchor=(-0.75, 3.3), ncol=3)

    # Add shared x-axis and y-axis labels
    fig.text(0.5, 0.01, 'Prediction error', ha='center')
    fig.text(0.04, 0.65, 'Density of errors from testing data', va='center', rotation='vertical')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.7)

    return fig


def process_histogram_data(selected_cluster, residuals_df, table_7, feature_names):
    # Extract the relevant predictions for the selected cluster
    predictions = table_7.iloc[selected_cluster - 1].drop("Cluster").values.astype(float)[[0, 2, 3, 4, 5, 6]]

    processed_data = []
    for i, col in enumerate(feature_names):
        values_with_residuals = predictions[i] + residuals_df[col]
        mean_error = values_with_residuals.mean()
        std_error = values_with_residuals.std()
        lower_bound = mean_error - 1.28 * std_error
        upper_bound = mean_error + 1.28 * std_error
        processed_data.append((values_with_residuals, mean_error, std_error, lower_bound, upper_bound))

    return processed_data

