# Import dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset in a dataframe object and include only four features as mentioned
dataset = pd.read_csv("football dataset.csv")

dataset.drop(['GameId'], axis = 1, inplace = True)

X1_var = dataset[['SuccessfulDribbling','SuccessfulPass','PlayerAttackingScore','PlayerDefendingScore','PlayerTeamPlayScore']]
y_var = dataset['PlayerTotalScore'] # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X1_var, y_var, test_size = 1, random_state = 0)

lr = LinearRegression()
lr.fit(X_train, y_train)

# Save your model
import joblib
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X1_var.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")