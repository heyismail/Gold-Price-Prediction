## Importing all the libraries required ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

## Data Collecting and Processing ##

# Converting the data from 'csv' format to 'Dataframe' using pandas.
gold_data = pd.read_csv("gld_price_data.csv")

# Insights of the dataset.
print(gold_data.head())  # prints the first five rows of the dataset.
print(gold_data.tail())  # prints the last five rows of the dataset.
print(gold_data.columns)  # prints the headings or column names of the dataset.
print(gold_data.shape)  # specifies the number of columns and rows in the dataset.
print(gold_data.info())  # short information of the dataset regarding their structure and datatypes.

# Check for the missing values in dataset.
print(gold_data.isnull().sum())

# Statistical description of the data.
print(gold_data.describe())

# Finding the correlation between the variables.
gold_data['Date'] = pd.to_datetime(gold_data['Date'], format='%m/%d/%Y')  # converting the 'Date' column to '%m/%d/%Y'.
gold_correlation = gold_data.corr()

# Plotting a heatmap of the dataset.
plt.figure(figsize=(8,8))
sns.heatmap(gold_correlation,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')
plt.show()  # uncomment to see the heatmap.

# Finding the 'GLD' correlation with other variables in the frame.
print(gold_correlation['GLD'])

# Plotting the distribution of 'GLD'
sns.distplot(gold_data['GLD'],kde=True,bins=20,color="green")
plt.show()


# Splitting Features and Target.
X = gold_data.drop(['Date','GLD'],axis=1)  # remove the 'Date' and 'GLD' column from the Dataframe.
Y = gold_data['GLD']

# Splitting into training and test data.
X_test, X_train, Y_test, Y_train = train_test_split(X,Y,test_size=0.2,random_state=2)

# Model Training: Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)  # generates the random forest regressor of 100 decision trees.
regressor.fit(X_train,Y_train)  # generates a model for following values of features and target.

## Model Evaluation ##

# Prediction on test data.
test_data_prediction = regressor.predict(X_test)  # predicts the target variables for values of features.

# Estimating the sum of squares or R square error.
error_score = metrics.r2_score(Y_test,test_data_prediction)
print(error_score)

# Conversion of Y-test values to list.
Y_test = list(Y_test)

# Final Result:
plt.plot(Y_test,color='black',label="Actual Value")
plt.plot(test_data_prediction,color='red',label="Predicted Value")
plt.title("Actual Price vs Predicted Price")
plt.xlabel("Number Of Values")
plt.ylabel("Gold Price")
plt.legend()
plt.show()
