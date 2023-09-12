import statistics
from collections import defaultdict
import ruptures as rpt
import pandas as pd
import streamlit as st


def generate_afterhours_energy(User_Energy, end_modes_weekdays_str, start_modes_weekdays_str):
    energyAf_weekdays = User_Energy[(User_Energy.index.dayofweek < 5)]
    energyAf_weekdays = energyAf_weekdays.between_time(end_modes_weekdays_str, start_modes_weekdays_str, include_start=True, include_end=False)
    energyAf_weekends = User_Energy[(User_Energy.index.dayofweek >= 5)]
    energyAf_weekends = energyAf_weekends.between_time('00:00', '23:59')
    energyAf = pd.concat([energyAf_weekdays, energyAf_weekends])

    st.session_state['energyAf_weekdays'] = energyAf_weekdays
    st.session_state['energyAf_weekends'] = energyAf_weekends
    st.session_state['energyAf'] = energyAf

    return energyAf_weekdays, energyAf_weekends, energyAf


def generate_operational_energy(User_Energy, start_modes_weekdays_str, end_modes_weekdays_str, start_modes_weekends_str, end_modes_weekends_str):
    energyOp_weekdays = User_Energy[(User_Energy.index.dayofweek < 5)]
    energyOp_weekdays = energyOp_weekdays.between_time(start_modes_weekdays_str, end_modes_weekdays_str, include_end=False)
    energyOp_weekends = User_Energy[(User_Energy.index.dayofweek >= 5)]
    energyOp_weekends = energyOp_weekends.between_time(start_modes_weekends_str, end_modes_weekends_str, include_end=False)
    energyOp = pd.concat([energyOp_weekdays])

    st.session_state['energyOp_weekdays'] = energyOp_weekdays
    st.session_state['energyOp_weekends'] = energyOp_weekends
    st.session_state['energyOp'] = energyOp

    return energyOp_weekdays, energyOp_weekends, energyOp


def convert_mode_to_string(start_modes_weekdays, end_modes_weekdays, start_modes_weekends, end_modes_weekends):
    # Convert the mode values to strings in the format 'HH:MM:SS'
    start_modes_weekdays_str = start_modes_weekdays.strftime('%H:%M:%S')
    end_modes_weekdays_str = end_modes_weekdays.strftime('%H:%M:%S')
    start_modes_weekends_str = start_modes_weekends.strftime('%H:%M:%S')
    end_modes_weekends_str = end_modes_weekends.strftime('%H:%M:%S')

    st.session_state['start_modes_weekdays_str'] = start_modes_weekdays_str
    st.session_state['end_modes_weekdays_str'] = end_modes_weekdays_str
    st.session_state['start_modes_weekends_str'] = start_modes_weekends_str
    st.session_state['end_modes_weekends_str'] = end_modes_weekends_str

    return start_modes_weekdays_str, end_modes_weekdays_str, start_modes_weekends_str, end_modes_weekends_str


def convert_mode_to_datetime(weekday_hours_mode, weekend_hours_mode):
    start_modes_weekdays = pd.to_datetime(weekday_hours_mode[0], format='%H')
    end_modes_weekdays = pd.to_datetime(weekday_hours_mode[1], format='%H')
    start_modes_weekends = pd.to_datetime(weekend_hours_mode[0], format='%H')
    end_modes_weekends = pd.to_datetime(weekend_hours_mode[1], format='%H')

    st.session_state['start_modes_weekdays'] = start_modes_weekdays
    st.session_state['end_modes_weekdays'] = end_modes_weekdays
    st.session_state['start_modes_weekends'] = start_modes_weekends
    st.session_state['end_modes_weekends'] = end_modes_weekends

    return start_modes_weekdays, end_modes_weekdays, start_modes_weekends, end_modes_weekends


def get_mode_of_hours(operating_hours):
    if not operating_hours:  # Check if the list is empty
        return None, None
    start_times, end_times = zip(*operating_hours)
    try:
        mode_start = statistics.mode(start_times)
        mode_end = statistics.mode(end_times)
    except statistics.StatisticsError:
        return None, None

    st.session_state['mode_start'] = mode_start
    st.session_state['mode_end'] = mode_end

    return (mode_start, mode_end)


def convert_operating_hours_to_hours(operating_hours):
    hours_operating_hours = []
    for day, periods in operating_hours.items():
        for period in periods:
            start_hours = convert_to_hours(period[0].time())
            end_hours = convert_to_hours(period[1].time())
            hours_operating_hours.append((start_hours, end_hours))

    st.session_state['start_hours'] = start_hours
    st.session_state['end_hours'] = end_hours
    st.session_state['hours_operating_hours'] = hours_operating_hours

    return hours_operating_hours


def convert_to_hours(time_obj):
    return time_obj.hour


# Function to get operating hours for all days
def get_all_operating_hours(data):
    operating_hours_dict = defaultdict(list)
    unique_days = data['day'].unique()
    for day in unique_days:
        daily_data = data[data['day'] == day]['Heating (W/m²)']

        # Check if the daily data is not constant
        if not daily_data.std() == 0:
            operating_hours = get_operating_hours(daily_data)

            # Take only the longest period of the day as the operating hours
            if operating_hours:  # This condition will skip over the empty lists
                longest_period = max(operating_hours, key=lambda x: x[1] - x[0])
                operating_hours_dict[day.isoformat()].append(
                    longest_period)  # Convert the date to a string using isoformat()

    st.session_state['operating_hours_dict'] = operating_hours_dict

    return operating_hours_dict


def get_operating_hours(daily_data):
    algo = rpt.Pelt(model="l1",jump = 1, min_size = 1).fit(daily_data.values.reshape(-1, 1))
    try:
        result = algo.predict(pen=50)
    except Exception:
        return []
    else:
        # The -1 is to exclude the last break point which is always the end of the data
        bkps = result[:-1]
        # Check if there are at least two breakpoints
        if len(bkps) < 2:
            return []
        # Convert the breakpoints (indices) into time
        bkps_times = [daily_data.index[i] if i < len(daily_data) else daily_data.index[-1] for i in bkps]
        # Create a list of tuples with the start and end times of each detected period
        operating_periods = [(bkps_times[i], bkps_times[i + 1]) for i in range(len(bkps_times) - 1)]

        st.session_state['operating_periods'] = operating_periods

        return operating_periods
