import streamlit as st


def get_user_inputs():
    floor_area = int(st.session_state.get('floor_area'))
    caption = st.session_state.get('selected_image_caption')
    address_line1 = st.session_state.get("address_line1", "N/A") or "N/A"
    address_line2 = st.session_state.get("address_line2", "N/A") or "N/A"
    city = st.session_state.get("city", "N/A") or "N/A"
    Postal_code = st.session_state.get("Postal_code", "N/A") or "N/A"
    year_of_construction = int(st.session_state.get("year_of_construction", 1900) or 1900)
    number_of_floors = int(st.session_state.get("number_of_floors", 1) or 1)
    building_name = st.session_state.get("building_name", "N/A") or "N/A"
    aspect_ratio = None
    wwr = None

    # error_handling
    if not (floor_area and caption):
        st.error('Please provide all user inputs in the previous page.')
        st.stop()
    else:
        try:
            aspect_ratio = caption.split("Aspect ratio - ")[1].split("|")[0]
            wwr = caption.split("WWR-")[1].split("%")[0]
        except IndexError:
            st.error('Invalid caption string: could not extract aspect ratio and/or WWR.')

    return building_name, floor_area, address_line1, address_line2, city, Postal_code, year_of_construction, number_of_floors, wwr, aspect_ratio