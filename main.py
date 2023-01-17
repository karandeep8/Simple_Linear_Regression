# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the Dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the Dataset into Testing and Training Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Simple Regression Model on the Training Dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# The method that we are going to use to train our Model is fit()
# fit method belongs to the LinearRegression Class
# In the fit method we have to pass the training set as a parameter
# But the training data has to be passed in a particular format
# And that format is x_train, y_train
regressor.fit(x_train, y_train)

# Predicting the Test set Results
y_pred = regressor.predict(x_test)

# Visualising the Training Set Results
"""X-Axis --> Number of Years of Experience
Y-Axis --> Salaries
Red Point --> Real Salary
Blue Straight Line --> Predicted Salary"""
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Years of Experience vs Salary (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test Set Results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_pred, color='blue')
plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

