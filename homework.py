# -*- coding: utf-8 -*-
# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains data of Titanic with various features and survivals.

Try to use what you have learnt today to predict whether the passenger shall survive or not.

Evaluate your model.
"""
# %%
# load data
import pandas as pd

data = pd.read_csv('/Users/sherry/Desktop/aiSummerCamp2025-master/day1/assignment/data/train.csv')
df = data.copy()
df.sample(10)
# %%
# delete some features that are not useful for prediction
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
df.info()
# %%
# check if there is any NaN in the dataset
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
df.dropna(inplace=True)
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# convert categorical data into numerical data using one-hot encoding
# For example, a feature like sex with categories ['male', 'female'] would be transformed into two new binary features, sex_male and sex_female, represented by 0 and 1.
df = pd.get_dummies(df)
df.sample(10)
# %% 
# separate the features and labels
X = df.drop('Survived', axis=1)
y = df['Survived']

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(random_state=42)
}

# 训练模型
for name, model in models.items():
    model.fit(X_train, y_train)

# predict and evaluate
from sklearn.metrics import accuracy_score, classification_report

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f'--- {name} ---')
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()

