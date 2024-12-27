import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pickle


file_path = 'dataset/extended_dataset.csv'
dataset = pd.read_csv(file_path)

# Drop unnecessary columns
dataset = dataset.drop(columns=[col for col in dataset.columns if 'Unnamed' in col or col == 'ID'])


