import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="House Price Prediction")
# Load your model using pickle
with open('model_rf.pkl', 'rb') as file:
    model = pickle.load(file)



def app1():
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
        st.success(f"Predicted Price: ₹{prediction[0]:,.0f}")


def app2():


    # Load dataset
    st.title("House Pricing Dataset")

    # Upload file
    dataset_path = "dataset/extended_dataset.csv"
    
    data = pd.read_csv(dataset_path)
    data = data.drop(columns=[col for col in data.columns if 'Unnamed' in col or col == 'ID'])
    data = data.drop(columns=['Age of Property','Nearby Schools Rating'],axis=1)
    st.write("### Dataset Preview:")
    st.write(data.head())
    
    # st.write(data.quality.value_counts())

    # Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis")
    if st.checkbox("Show Dataset Summary"):
        st.write(data.describe())

    if st.checkbox("Show Data type Distribution"):
        data_types = data.dtypes.value_counts()
        data_types.plot(kind='bar', figsize=(8, 4))
        plt.title("Data Type Distribution")
        plt.xlabel("Data Type")
        plt.ylabel("Count")
        st.pyplot(plt)

    if st.checkbox("Show Missing Values"):
        st.write(data.isnull().sum())
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Data Heatmap")
        st.pyplot(plt)

    

    ##BoxPlot for the outliers
    if st.checkbox("BoxPlot for the Outliers"):
        for column in data.select_dtypes(include='number').columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=data[column])
            plt.title(f"Outliers in {column}")
            plt.xlabel(column)
            st.pyplot(plt)

    ##Encoding the Categorical Features
    labelencoder = LabelEncoder()
    categorical_columns = ['Pool', 'Central Air', 'Heating Type', 'Crime Rate']
    for col in categorical_columns:
        data[col] = labelencoder.fit_transform(data[col])
    scale_columns = [ 'Square Footage','Distance to City Center', 'Property Tax', 'Previous Sale Price']

    ##Normalizing the Numerical Values
    scaler = StandardScaler()
    data[scale_columns] = scaler.fit_transform(data[scale_columns])
        
    #Target Selection
    st.subheader("Feature Selection")
    target_column = st.selectbox("Select Target Column", data.columns)
    # feature_columns = st.multiselect("Select Feature Columns", [col for col in data.columns if col != target_column])
    feature_columns = data.drop(columns=[target_column]).columns.tolist()

    if st.checkbox("Target Variable analysis"):
        sns.histplot(data[target_column], kde=True)
        plt.title("Distribution of Target")
        st.pyplot(plt)
    
    # if st.checkbox("Show Target Distribution"):
    # # Create the plot
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     sns.countplot(x=target_column, data=data, ax=ax)
    #     ax.set_title('Distribution of Wine Quality')
    #     ax.set_xlabel('Quality')
    #     ax.set_ylabel('Count')
    #     st.pyplot(fig)

    #For Corelation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


    #splitting  
    if target_column and feature_columns:
        X = data[feature_columns]
        y = data[target_column]
        model_name = st.selectbox("Select Model", ["Random Forest","Linear Regression"])
        
        

        if st.button("Train Model"):
            # Model selection
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_name == "Linear Regression":
                model = LinearRegression()

            # Model training
            model.fit(X_train, y_train)
            
            

            # Model evaluation
            y_pred = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            st.subheader("Accuraccy:")
            st.success(accuracy)

            # st.subheader("Classification Report")
            # report = classification_report(y_test, y_pred, output_dict=True)
            # report_df = pd.DataFrame(report).transpose()
            # st.write(report_df.style.format(precision=2).set_table_styles(
            #     [{'selector': 'th', 'props': [('background-color', 'black'), ('color', 'white')]}]
            # ))

            rmse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.subheader("Mean Squared Error")
            st.success(f'{rmse:.2f}')
            st.subheader("r2_score")
            st.success(f'{r2:.2f}')
            

            

            # Feature importance (only for Random Forest and Decision Tree)
            if model_name in ["Random Forest"]:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    "Feature": feature_columns,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                st.bar_chart(importance_df.set_index("Feature"))

            # if model_name in ["Linear Regression"]:
            st.subheader("Scatter plot for Price prediction vs Actual Price")
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel("Actual Prices")
            plt.ylabel("Predicted Prices")
            plt.title("Actual vs Predicted Prices")
            st.pyplot(plt)

            st.subheader("""Conclusion Random forest was performing good on the given dataset\n""")
            st.error("Linear regression was performing bad on the used dataset")             
                    


selected_app = st.sidebar.selectbox("Select an App", ["House Price Prediction", "Model Visualizations"])

if selected_app == "House Price Prediction":
    app1()
elif selected_app == "Model Visualizations":
    app2()


