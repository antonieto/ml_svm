import pandas as pd
from sklearn.model_selection import KFold

# Load the dataset
train = pd.read_csv("train.csv").dropna()

# Drop columns
train = train.drop(columns='Name')
train = train.drop(columns='PassengerId')
train = train.drop(columns='Ticket')
train = train.drop(columns='Cabin')

# Preprocess the dataframe (categorizing)
train['Sex'] = train['Sex'].replace({'female': 0, 'male': 1})
train['Embarked'] = train['Embarked'].replace({'C': 0, 'S': 1, 'Q': 2})

# Split data frame in data and target
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']

# Set up a 10-Fold cross-validation
kf = KFold(n_splits=10, random_state=42, shuffle=True)
