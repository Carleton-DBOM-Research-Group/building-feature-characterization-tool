import pandas as pd
import streamlit as st


def get_benchmark_data():
    # Data for the 35 office buildings
    data = {
        "Building": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10",
                     "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19", "B20",
                     "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28", "B29", "B30",
                     "B31", "B32", "B33", "B34", "B35"],
        "Floor area (1000 m2)": [11, 11, 6, 8, 6, 149, 8, 14, 75, 12, 26, 7, 60, 72, 32, 62, 13, 31, 12, 63, 8,
                                 21, 17, 8, 34, 7, 17, 12, 33, 61, 12, 19, 41, 4, 39],
        "Heating energy use intensity (kWh/m2-yr)": [244, 200, 154, 156, 101, 146, 212, 220, 227, 220, 193, 242,
                                                     55, 18, 281, 159, 327, 89, 86, 83, 334, 160, 142, 721, 458,
                                                     133, 73, 170, 79, 33, 815, 54, 70, 271, 121],
        "Cooling energy use intensity (kWh/m2-yr)": [172, 73, 91, 49, 90, 72, 141, 91, 114, 183, 125, 101, 71,
                                                     81, 165, 190, 253, 56, 80, 55, 149, 127, 50, 369, 101, 71,
                                                     17, 34, 82, 56, 181, 24, 70, 55, 67],
        "Vintage": [1847, 1911, 1924, 1930, 1919, 1977, 1911, 1913, 1974, 1866, 1949, 1889, 2015, 1990, 1967,
                    1973, 1940, 1960, 1961, 1970, 1990, 1990, 1962, 1965, 1978, 1952, 1957, 1960, 1970, 1979,
                    1954, 1965, 1974, 1955, 1952],
        "WWR": [0.2, 0.6, 0.2, 0.2, 0.4, 1.0, 0.6, 0.2, 0.6, 0.2, 0.4, 0.4, 0.8, 0.8, 0.2, 0.4, 0.4, 0.4, 0.6,
                0.4, 0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4, 0.4, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4]
    }

    # Create a DataFrame
    benchmark_df = pd.DataFrame(data)

    return benchmark_df


def create_summary_table(building_name, floor_area, year_of_construction, wwr, user_heating_eui,
                         user_cooling_eui, user_combined_eui, benchmark_df):
    summary_data = {
        "Metric": ["Floor Area (1000 m²)", "Vintage", "WWR (%)", "Heating EUI (kWh/m²-yr)",
                   "Cooling EUI (kWh/m²-yr)", "Combined EUI (kWh/m²-yr)"],
        building_name: [
            int(floor_area / 1000),  # Changed this line
            int(year_of_construction),  # Removed extra , 0
            int(wwr),
            int(user_heating_eui),
            int(user_cooling_eui),
            int(user_combined_eui)
        ],
        "Benchmark Average": [
            round(benchmark_df['Floor area (1000 m2)'].mean(), 0),
            int(benchmark_df['Vintage'].mean()),
            int(benchmark_df['WWR'].mean()),
            round(benchmark_df['Heating energy use intensity (kWh/m2-yr)'].mean(), 0),
            round(benchmark_df['Cooling energy use intensity (kWh/m2-yr)'].mean(), 0),
            round((benchmark_df['Heating energy use intensity (kWh/m2-yr)'] + benchmark_df[
                'Cooling energy use intensity (kWh/m2-yr)']).mean(), 0)
        ],
        "Benchmark Minimum": [
            round(benchmark_df['Floor area (1000 m2)'].min(), 0),
            int(benchmark_df['Vintage'].min()),
            int(benchmark_df['WWR'].min()),
            round(benchmark_df['Heating energy use intensity (kWh/m2-yr)'].min(), 0),
            round(benchmark_df['Cooling energy use intensity (kWh/m2-yr)'].min(), 0),
            round((benchmark_df['Heating energy use intensity (kWh/m2-yr)'] + benchmark_df[
                'Cooling energy use intensity (kWh/m2-yr)']).min(), 0)
        ],
        "Benchmark Maximum": [
            round(benchmark_df['Floor area (1000 m2)'].max(), 0),
            int(benchmark_df['Vintage'].max()),
            int(benchmark_df['WWR'].max()),
            round(benchmark_df['Heating energy use intensity (kWh/m2-yr)'].max(), 0),
            round(benchmark_df['Cooling energy use intensity (kWh/m2-yr)'].max(), 0),
            round((benchmark_df['Heating energy use intensity (kWh/m2-yr)'] + benchmark_df[
                'Cooling energy use intensity (kWh/m2-yr)']).max(), 0)
        ],
        "Benchmark Std Dev": [
            round(benchmark_df['Floor area (1000 m2)'].std(), 0),
            round(benchmark_df['Vintage'].std(), 0),  # Vintage standard deviation as rounded integer
            (benchmark_df['WWR'].std()),
            round(benchmark_df['Heating energy use intensity (kWh/m2-yr)'].std(), 0),
            round(benchmark_df['Cooling energy use intensity (kWh/m2-yr)'].std(), 0),
            round((benchmark_df['Heating energy use intensity (kWh/m2-yr)'] + benchmark_df[
                'Cooling energy use intensity (kWh/m2-yr)']).std(), 0)
        ]
    }

    # Convert the summary_data dictionary to a DataFrame
    summary_table = pd.DataFrame.from_dict(summary_data)

    # Format the table
    summary_table['Benchmark Average'] = summary_table['Benchmark Average'].round(0).astype(int)
    summary_table['Benchmark Minimum'] = summary_table['Benchmark Minimum'].round(0).astype(int)
    summary_table['Benchmark Maximum'] = summary_table['Benchmark Maximum'].round(0).astype(int)
    summary_table['Benchmark Std Dev'] = summary_table['Benchmark Std Dev'].round(0).astype(int)

    # Convert Vintage to integer (remove commas)
    summary_table.loc[1, building_name] = int(summary_table.loc[1, building_name])
    summary_table.loc[1, 'Benchmark Average':'Benchmark Std Dev'] = summary_table.loc[1,
                                                                    'Benchmark Average':'Benchmark Std Dev'].astype(
        int)

    # Correct WWR (multiply by 100 for benchmark)
    summary_table.loc[2, building_name] = round(float(summary_table.loc[2, building_name]), 1)
    summary_table.loc[2, 'Benchmark Average':'Benchmark Std Dev'] = summary_table.loc[2,
                                                                    'Benchmark Average':'Benchmark Std Dev'].astype(
        float).round(1)

    return summary_table