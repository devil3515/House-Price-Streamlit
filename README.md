# 🏠 House Price Prediction with Streamlit

## 🌟 Overview
Welcome to the **House Price Prediction** app—a sleek and interactive tool to predict house prices based on key property features. This project combines data analysis, machine learning, and visualization to deliver insights and predictions directly to your fingertips.

Whether you’re a data enthusiast, homeowner, or just curious about how machine learning works in real estate, this app has something for you! 🚀

**👉 Try it now:** [Live Demo]([https://example.com](https://house-price-streamlit.onrender.com))
Note:- Live Link will take time to load

---

## ✨ Features at a Glance
### 1️⃣ **House Price Prediction**
- 🏡 Input property details like square footage, bedrooms, bathrooms, and more.
- 📊 Get instant predictions powered by a pre-trained **Random Forest Regressor** model.
- 🖥️ Clean and intuitive interface for seamless user experience.

### 2️⃣ **Model Visualizations**
- 🔍 **Explore the Dataset**  
  - View the dataset and its summary statistics.
  - Detect missing values and visualize them with heatmaps.  
  - Spot outliers using boxplots.

- 🤖 **Train and Evaluate Models**  
  - Compare the performance of **Random Forest** and **Linear Regression** models.  
  - Dive into metrics like **R2 Score**, **RMSE**, and accuracy.  
  - Visualize feature importance and actual vs. predicted prices.

- 📈 **Interactive Plots**  
  - Heatmaps to reveal feature correlations.  
  - Histograms for target variable analysis.

---

## 🛠️ Tech Stack
### Languages & Frameworks
- **Python**: The backbone of the app.
- **Streamlit**: For building an interactive web interface.

### Libraries
- **pandas**, **numpy**: Data manipulation and analysis.  
- **matplotlib**, **seaborn**: Visualizations and plotting.  
- **scikit-learn**: Machine learning and evaluation.  
- **pickle**: To save and load pre-trained models.

---

## 🚀 Quick Start Guide
Follow these steps to get the app running locally:

1. Clone this repository:
   ```bash
   git clone devil3515/House-Price-Streamlit
   ```
2. Navigate to the project directory:
   ```bash
   cd House-Price-Streamlit
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the app:
   ```bash
   streamlit run app.py
   ```
5. Open the URL provided in your terminal to interact with the app!

---

## 📊 Dataset Overview
The dataset includes features like:
- Square footage, number of bedrooms/bathrooms, floors, and garage space.
- Property-specific attributes like central air, heating type, and pool.
- Neighborhood insights like distance to the city center, crime rate, and property tax.

### Preprocessing Highlights
- Removed redundant columns (e.g., `ID`, `Unnamed` columns).  
- Normalized numerical features like square footage and property tax.  
- Encoded categorical variables such as pool and heating type.

---

## 🔍 Insights & Performance
- **Random Forest Regressor** consistently outperforms **Linear Regression** on this dataset.
- Key evaluation metrics:
  - **Accuracy**: Measures how well the model predicts prices.  
  - **R2 Score**: Indicates how well the model fits the data.  
  - **RMSE**: Tracks the error margin in predictions.

---

## 📈 Room for Improvement
Here’s how we plan to take this app to the next level:
1. Integrate advanced models like **Gradient Boosting** or **XGBoost**.
2. Enable users to upload custom datasets for personalized predictions.
3. Add more interactive visualizations and dynamic dashboards.

---

## 👩‍💻 About the Author
Hi there! I’m **Abhishek Kumar**, an MCA graduate with a knack for crafting machine learning solutions. From data exploration to model deployment, I love simplifying the complex world of AI. Connect with me to discuss **machine learning**, **data science**, and all things **tech**!

---

