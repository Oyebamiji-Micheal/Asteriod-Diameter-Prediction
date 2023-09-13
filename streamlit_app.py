import streamlit as st
import pandas as pd
import joblib


def write_project_info():
    st.write("## Asteroid Diameter Prediction")

    st.write("""
        Predicting the diameters of asteroids using machine learning
        and astronomical data
    """
    )

    st.image("images/web_cover.jpeg")

    st.write("""
        ## About

        <p align="justify">
            One of the intriguing challenges in the field of astronomy is the prediction of asteroid diameters
            <a href="https://www.kaggle.com/datasets/basu369victor/prediction-of-asteroid-diameter" target="_blank" style="text-decoration: None"> [Source]</a>. Asteroids, celestial bodies orbiting the Sun, come in various shapes and sizes, making the estimation of their diameters a complex task. Over the years, numerous methods and approaches have been developed to tackle this challenge, each striving to outperform its predecessors. I have decided to step outside of my domain into astronomy, aiming to leverage the power of data-driven algorithms to predict the diameters of asteroid accurately to a reasonable extent. In doing so, it not only contributes to the field of space science (lol 🤤) but also demonstrates the potential of machine learning in solving complex problems in alien domains.
        </p>
    """, unsafe_allow_html=True)

    st.write("""
        <p align="justify">
            I have benchmarked my result against a <a href="http://www.iraj.in/journal/journal_file/journal_pdf/12-555-156136953136-40.pdf" target="_blank" style="text-decoration: None">research paper</a> by the 'owner' of this dataset on <a href="https://www.kaggle.com/datasets/basu369victor/prediction-of-asteroid-diameter" target="_blank" style="text-decoration: None">kaggle</a>. This web app isn't a detailed policy blueprint for comparison but testing only. For the more technical aspect including the documentation, notebook, dataset, evaluation metric, models etc, kindly refer to the repository on <a href="https://github.com/Oyebamiji-Micheal/Asteriod-Diameter-Prediction/tree/main" target="_blank" style="text-decoration: None">Github</a>.
        </p>
    """, unsafe_allow_html=True)

    st.write("""**Made by Oyebamiji Micheal**""")


def take_user_inputs():
    st.sidebar.header("User Input Features")

    semi_major_axis = st.sidebar.number_input(
        "**Semi Major Axis**: This represents the size of the object's orbit around the Sun in astronomical units", step=0.001, min_value=0.0, max_value=100.0
    )

    eccentricity = st.sidebar.number_input(
        "**Asteroid Eccentricity**: Indicates how elliptical or circular the object's orbit is, with values close to 1 indicating high eccentricity", step=0.001, min_value=0.0, max_value=1.0
    )

    inclination_deg = st.sidebar.number_input(
        "**Inclination Degree**: Angle describing the tilt of the object's orbit relative to the solar system's plane", step=0.001, min_value=0.0, max_value=360.0
    )

    longitude_of_the_ascending_node = st.sidebar.number_input(
        "**Longitude of the Ascending Node**: Specifies the position of the object's orbital ascending node", step=0.001, min_value=0.0, max_value=1000.0
    )

    argument_of_perihelion = st.sidebar.number_input(
        "**Argument of Perihelion**: Defines the angle between perihelion and the ascending node", step=0.001, min_value=0.0, max_value=1000.0
    )

    perihelion_distance = st.sidebar.number_input(
        "**Perihelion Distance**: Closest distance between the object and the Sun during its orbit, measured in astronomical units", step=0.001, min_value=0.0, max_value=1000.0
    )

    aphelion_distance = st.sidebar.number_input(
        "**Aphelion Distance**: Farthest distance between the object and the Sun during its orbit, measured in astronomical units", step=0.001, min_value=0.0, max_value=1000.0
    )

    orbital_period = st.sidebar.number_input(
        "**Orbital Period**: Time taken for the object to complete one orbit around the Sun, measured in years", step=0.001, min_value=0.0, max_value=10000.0
    )

    data_arc_span = st.sidebar.number_input(
        "**Data arc-span (d)**: Duration over which observational data has been collected for the object, measured in days", step=0.001, min_value=0.0, max_value=1000000000.0
    )

    number_of_observations_used = st.sidebar.number_input(
        "**Number of observations used**: Number of observational data points used to calculate the object's orbital parameters", step=0.001, min_value=0.0, max_value=1000000.0
    )

    absolute_magnitude_parameter = st.sidebar.number_input(
        "**Absolute Magnitude Parameter**: Measure of the object's intrinsic brightness or reflectivity, indicating its size and composition", step=0.001, min_value=0.0, max_value=10000.0
    )

    geometric_albedo = st.sidebar.number_input(
        "**Geometric Albedo**: Reflectivity of the object's surface, indicating how much sunlight it reflects", step=0.001, min_value=0.0, max_value=100.0
    )

    rotation_period = st.sidebar.number_input(
        "**Rotation Period**: Time taken for the object to complete one full rotation around its axis, measured in hours", step=0.001, min_value=0.0, max_value=100.0
    )

    earth_minimum_orbit_intersection_distance = st.sidebar.number_input(
        "**Earth Minimum orbit Intersection Distance**: Quantifies the closest approach of the object's orbit to Earth's orbit, providing information about potential close encounters with our planet", step=0.001, min_value=0.0, max_value=1000.0
    )

    near_earth_object = st.sidebar.selectbox(
        "**Near Earth Object**: Indicates whether the object is classified as a Near Earth Object (NEO), with orbits in close proximity to Earth", ("Yes", "No")
    )

    physically_hazardous_asteroid = st.sidebar.selectbox(
        "**Physically Hazardous Asteroid**: Identifies whether the object is classified as a Physically Hazardous Asteroid (PHA) with the potential to pose a physical threat to Earth", ("Yes", "No")
    )

    orbit_condition_code = st.sidebar.selectbox(
        "**Orbit Condition Code**: The orbit condition code, also known as the U uncertainty parameter, is an integer between 0 and 9 indicating  the quality and reliability of the object's orbital data on a logarithmic scale, where 0 indicates a well-determined orbit.", ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    )

    # Format inputs to training data representation
    mapping = {"Yes": "Y", "No": "N" }

    single_input = {
        "semi_major_axis": semi_major_axis,
        "eccentricity": eccentricity,
        "inclination_deg": inclination_deg,
        "longitude_of_the_ascending_node": longitude_of_the_ascending_node, 
        "argument_of_perihelion": argument_of_perihelion,
        "perihelion_distance": perihelion_distance,
        "aphelion_distance": aphelion_distance,
        "orbital_period": orbital_period,
        "data_arc_span": data_arc_span,
        "number_of_observations_used": number_of_observations_used,
        "absolute_magnitude_parameter": absolute_magnitude_parameter,
        "geometric_albedo": geometric_albedo,
        "rotation_period": rotation_period,
        "earth_minimum_orbit_intersection_distance": earth_minimum_orbit_intersection_distance,
        "orbit_condition_code": orbit_condition_code,
        "near_earth_object": mapping[near_earth_object],
        "physically_hazardous_asteroid": mapping[physically_hazardous_asteroid]
    }

    return single_input


def predict_input(single_input):
    # Convert input into a pandas dataframe
    input_df = pd.DataFrame([single_input])

    model = joblib.load("asteroid_model.joblib")

    numeric_cols = model['numeric_cols']
    categorical_cols = model['categorical_cols']
    encoded_cols = model['encoded_cols']

    # Load fitted scaler and encoder
    scaler = model['scaler']
    encoder = model['encoder']
    
    # Load trained random forest model
    lgbm_model = model['lgbm_model']
    
    # Transform numeric columns
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Encoded categorical columns
    encoded_data = encoder.transform(input_df[categorical_cols])
    
    input_df[encoded_cols] = encoded_data.toarray()
    
    input_cols = numeric_cols.tolist() + encoded_cols.tolist()
    
    X_input = input_df[input_cols]
    
    prediction = lgbm_model.predict(X_input)
    
    return prediction[0]


if __name__ == "__main__":
    write_project_info()

    user_input = take_user_inputs()

    predict_asteroid_diameter = st.button("Predict Asteroid Diameter")

    if predict_asteroid_diameter:
        prediction = predict_input(user_input)
        
        st.write("Model = LightGBM")

        st.write(f"Predicted asteroid diameter = {prediction}")
