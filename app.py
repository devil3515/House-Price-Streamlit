import streamlit as st
import pickle
import pandas as pd

# Load your model using pickle
with open('model_rf.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app header
st.title("House Price Prediction")

# Create a form for user input
with st.form(key='prediction_form'):
    square_footage = st.number_input('Square Footage', min_value=0.0, format="%.2f")
    bedrooms = st.number_input('Bedrooms', min_value=1, step=1)
    bathrooms = st.number_input('Bathrooms', min_value=1, step=1)
    floors = st.number_input('Floors', min_value=1, step=1)
    garage = st.number_input('Garage', min_value=0, step=1)
    pool = st.selectbox('Pool', ['No', 'Yes'])
    central_air = st.selectbox('Central Air', ['No', 'Yes'])
    heating_type = st.selectbox('Heating Type', ['Gas', 'Electric', 'Oil'])
    distance_to_city_center = st.number_input('Distance to City Center', min_value=0.0, format="%.2f")
    crime_rate = st.selectbox('Crime Rate', ['Low', 'Medium', 'High'])
    property_tax = st.number_input('Property Tax', min_value=0.0, format="%.2f")
    previous_sale_price = st.number_input('Previous Sale Price', min_value=0.0, format="%.2f")
    # Categorical Inputs in the specified order
#     pool = st.selectbox("Pool", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1], index=0)
#     central_air = st.selectbox("Central Air", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1], index=0)
#     heating_type = st.selectbox("Heating Type", options=[(0, "Gas"), (1, "Electric"), (2, "Oil")], format_func=lambda x: x[1], index=0)
#     crime_rate = st.selectbox("Crime Rate", options=[(0, "Low"), (1, "Medium"), (2, "High")], format_func=lambda x: x[1], index=0)
    
    submit_button = st.form_submit_button(label='Predict Price')

# Check if form was submitted
if submit_button:
    # Convert form data to a dictionary
    heating_type_mapping = {"Gas": 0, "Electric": 1, "Oil": 2}
    crime_rate_mapping = {"Low": 0, "Medium": 1, "High": 2}

    # Update form data with numerical values
    form_data = {
        'Square Footage': float(square_footage),
        'Bedrooms': float(bedrooms),
        'Bathrooms': float(bathrooms),
        'Floors': int(floors),
        'Garage': int(garage),
        'Pool': 1 if pool == 'Yes' else 0,
        'Central Air': 1 if central_air == 'Yes' else 0,
        'Heating Type': heating_type_mapping[heating_type],  # Map to number
        'Distance to City Center': float(distance_to_city_center),
        'Crime Rate': crime_rate_mapping[crime_rate],  # Map to number
        'Property Tax': float(property_tax),
        'Previous Sale Price': float(previous_sale_price)
    }


    df = pd.DataFrame([form_data])

    # Make prediction
    prediction = model.predict(df)

    # Show the prediction result
    st.subheader("Predicted House Price")
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")

