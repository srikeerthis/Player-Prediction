# Import dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset in a dataframe object and include only four features as mentioned
dataset = pd.read_csv("football dataset.csv")
#finding missing data percentage in each column
for col in dataset.columns:
    pct_missing = np.mean(dataset[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

# impute the missing values and create the missing value indicator variables for each numeric column.
df_numeric = dataset.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values

for col in numeric_cols:
    missing = dataset[col].isnull()
    num_missing = np.sum(missing)
    
    if num_missing > 0:  # only do the imputation for the columns that have missing values.
        print('imputing missing values for: {}'.format(col))
        dataset['{}_ismissing'.format(col)] = missing
        med = dataset[col].median()
        dataset[col] = dataset[col].fillna(med)

# drop duplicates
dataset = dataset.drop_duplicates(subset=dataset.columns)
# there were duplicate rows
print(dataset.shape)
dataset.to_csv("football dataset.csv",index=False)

dataset.drop(['GameId'], axis = 1, inplace = True)

player_id = dataset['PlayerId']

unique_player_data = []
# create list of player id
value = dataset['PlayerId']
x = np.array(value)
# compute average score of each attribute of individual player
unique_player_id,counts = np.unique(x,return_counts=True)
j=0
for i in counts:
  k=j
  j= j+i
  val = dataset.loc[k:j].mean()
  unique_player_data.append(np.array(val))
  
new_dataframe = pd.DataFrame(data=unique_player_data,columns=dataset.columns)
new_dataframe['PlayerId'] = new_dataframe['PlayerId'].astype(int) 
print(new_dataframe)
X1_var = new_dataframe[['OnTargetShot','SuccessfulDribbling','SuccessfulPass','PlayerAttackingScore','PlayerDefendingScore','PlayerTeamPlayScore']]
y_var = new_dataframe['PlayerTotalScore'] # dependent variable

X_train, X_test, y_train, y_test = train_test_split(X1_var, y_var, test_size = 1, random_state = 0)

lr = LinearRegression()
lr.fit(X_train, y_train)

# Save your model
import joblib
joblib.dump(lr, 'model1.pkl')
print("Model dumped!")

# Load the model that you just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(X1_var.columns)
joblib.dump(model_columns, 'model_columns1.pkl')
print("Models columns dumped!")