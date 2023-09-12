from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def plot_feature_intervals_subplots(selected_cluster_df, feature_units):
    # Exclude 'SHGC' from the features
    selected_features = selected_cluster_df[selected_cluster_df['Feature'] != 'SHGC']
    selected_features.reset_index(drop=True, inplace=True)

    # Determine the number of subplots
    num_features = len(selected_features)
    num_rows = (num_features + 2) // 3

    # Create the subplots
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))

    # Flatten the axes for easy iteration
    axs = axs.flatten()

    # Plot each feature in a separate subplot
    for index, row in selected_features.iterrows():
        ax = axs[index]

        # Plot the median value as a point
        ax.scatter(0, row['Median'], marker='o', color='blue')

        # Add error bars for the bounds
        lower_error = row['Median'] - row['Lower_Bound']
        upper_error = row['Upper_Bound'] - row['Median']
        ax.errorbar(0, row['Median'], yerr=[[lower_error], [upper_error]], fmt='o', capsize=5, color='blue')

        # Add labels and units
        feature_unit = feature_units[row['Feature']]
        ax.set_title('{} ({})'.format(row['Feature'], feature_unit))
        ax.set_ylabel('Value')
        ax.set_xticks([])  # Hide x-axis ticks

    # Hide any unused subplots
    for i in range(index + 1, len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    return fig


def plot_time_series(User_Energy, User_Energy1):
    # Resample to get mean heating and cooling energy at daily frequency
    df_daily = User_Energy.resample('D').mean()
    df_dailyHtg = User_Energy1['Heating (W/m²)'].resample('D').mean()
    df_dailyClg = User_Energy1['Cooling (W/m²)'].resample('D').mean()
    df_dailyTout = User_Energy1['Dry Bulb Temperature (\u00b0C)'].resample('D').mean()

    # Set the style and context for the plot
    sns.set_style('whitegrid')
    sns.set_context('talk', rc={'font.size': 16})

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # Plot mean heating and cooling energy over time on the first subplot
    sns.lineplot(data=df_dailyHtg, ax=ax1, color='red', label='Heating (W/m²)')
    sns.lineplot(data=df_dailyClg, ax=ax1, color='blue', label='Cooling (W/m²)')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.20), ncol=2)
    ax1.set_ylabel('Load Intensity (W/m²)')

    # Plot outdoor air temperature on the second subplot
    sns.lineplot(data=df_dailyTout, ax=ax2, color='green', label='Outdoor Air Temperature (\u00b0C)')
    ax2.set_ylabel('Outdoor Air Temperature (\u00b0C)')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 1.20))

    return fig


def plot_scatter(User_Energy1):
    # Extract the data you want to plot
    x = User_Energy1['Dry Bulb Temperature (\u00b0C)']
    y1 = User_Energy1['Heating (W/m²)']
    y2 = User_Energy1['Cooling (W/m²)']

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x, y1, color='red', alpha=0.1, s=20, label='Heating (W/m²)')
    ax.scatter(x, y2, color='blue', alpha=0.1, s=20, label='Cooling (W/m²)')

    # Add labels and title
    ax.set_xlabel('Outdoor Air Temperature (\u00b0C)')
    ax.set_ylabel('Load Intensity  (W/m²)')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol=2)

    return fig


def plot_box(User_Energy, User_Energy1):
    User_Energy1 = User_Energy1.reset_index()

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # Extract the month from the Time column and create a dictionary of month names
    User_Energy = User_Energy.reset_index()
    User_Energy1['Month'] = User_Energy1['Time'].dt.month.map(
        {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov',
         12: 'Dec'})

    # Group the data by month and calculate the mean heating and cooling energy use for each month
    df_monthly = User_Energy1.groupby('Month')[['Heating (W/m²)', 'Cooling (W/m²)']].mean()

    # Rename the index to show the month name instead of the month number
    df_monthly.index.name = 'Month'

    # Plot heating and cooling energy use by month
    heating_box = sns.boxplot(x="Month", y="Heating (W/m²)", data=User_Energy1, ax=ax1, color='red')
    cooling_box = sns.boxplot(x="Month", y="Cooling (W/m²)", data=User_Energy1, ax=ax2, color='blue')

    # Add labels and title
    ax1.set_ylabel('Heating Energy Use (W/m²)')
    ax2.set_ylabel('Cooling Energy Use (W/m²)')

    # Combine the handles and labels of the heating and cooling legends
    handles = [mpatches.Patch(color='red', label='Heating'),
               mpatches.Patch(color='blue', label='Cooling')]
    labels = [handle.get_label() for handle in handles]

    # Create a new legend that combines the heating and cooling legends
    fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2,
               bbox_transform=ax1.transAxes)

    # Hide the x-axis label on the first subplot
    ax1.set_xlabel('')

    return fig


def plot_heat_map(User_Energy1, selected_date):
    # Select the data for the selected date
    selected_data = User_Energy1[User_Energy1['Date'] == selected_date]

    # Calculate the total energy use
    selected_data['Total (W/m²)'] = selected_data['Heating (W/m²)'] + selected_data['Cooling (W/m²)']

    # Group the data by hour and outdoor temperature and calculate the mean total energy use for each group
    pivot = selected_data.groupby([selected_data.Time.dt.strftime('%I %p'), 'Dry Bulb Temperature (\u00b0C)'])[
        'Total (W/m²)'].mean().unstack()

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap='coolwarm', cbar_kws={'label': 'Mean Total Energy (W/m²)'}, ax=ax)

    # Add labels and title
    ax.set_xlabel('Outdoor Air Temperature (\u00b0C)')
    ax.set_ylabel('Time of Day')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
