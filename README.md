# House Price Prediction with Streamlit

## Overview
This project is a web application designed to predict house prices based on various features. The application is developed using Streamlit and includes two main functionalities:

1. **House Price Prediction:** An interactive tool where users input house features to get a predicted price.
2. **Model Visualizations:** A comprehensive interface to explore the dataset, visualize key insights, and train machine learning models.

## Features
- **House Price Prediction:**
  - Input features such as square footage, number of bedrooms, bathrooms, and more.
  - Real-time prediction using a pre-trained Random Forest model loaded with `pickle`.
  - User-friendly interface with forms for data input.

- **Dataset Exploration:**
  - Preview the dataset and summary statistics.
  - Check for missing values and visualize their distribution.
  - Visualize correlations and outliers using heatmaps and boxplots.

- **Model Training and Evaluation:**
  - Train models such as Random Forest and Linear Regression directly in the app.
  - Evaluate models with metrics like RMSE, R2 score, and accuracy.
  - Visualize feature importance and actual vs predicted prices.

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - Streamlit (for building the web app)
  - pandas, numpy (for data manipulation)
  - matplotlib, seaborn (for data visualization)
  - scikit-learn (for machine learning and evaluation)
  - pickle (for saving and loading the pre-trained model)

## How to Run the Application
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd house-price-prediction
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```
5. Open the provided local URL in your browser to interact with the application.

## Dataset
The dataset includes various features such as square footage, number of bedrooms and bathrooms, property tax, crime rate, and more. It has been preprocessed to remove unnecessary columns and normalize numerical features.

## Application Interface
- **House Price Prediction:**
  - Form-based interface for users to input house features.
  - Outputs a predicted house price in INR.

- **Model Visualizations:**
  - Data summary and type distribution.
  - Missing value heatmaps and boxplots for outliers.
  - Correlation heatmaps to analyze feature relationships.
  - Feature importance visualization for Random Forest.

## Model Details
- Pre-trained **Random Forest Regressor** model loaded using `pickle`.
- Option to train new models (Random Forest or Linear Regression) within the app.
- Evaluation metrics:
  - **Accuracy:** Percentage of correctly predicted house prices.
  - **RMSE:** Root Mean Squared Error of the predictions.
  - **R2 Score:** Coefficient of determination.

## Future Improvements
- Add more advanced models such as Gradient Boosting or XGBoost.
- Include a feature to upload custom datasets for predictions.
- Enhance visualizations with more detailed insights and user controls.

## Author
**Abhishek Kumar**  
MCA Graduate passionate about machine learning and data analysis. Let’s connect to discuss AI, data, and tech trends.

---
Feel free to contribute by submitting issues or pull requests!

