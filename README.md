Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: lokesh khanna R
RegisterNumber:212222040088

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()
df.tail()

#segregating data to variables
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data

plt.scatter(x_train,y_train,color="black") 
plt.plot(x_train,regressor.predict(x_train),color="red") 
plt.title("Hours VS scores (learning set)") 
plt.xlabel("Hours") 
plt.ylabel("Scores") 
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color="cyan")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("Hours VS scores (learning set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

import numpy as np
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
*/
Output:
df.head():
Screenshot 2023-08-26 120009

df.tail():
Screenshot 2023-08-26 120021

Array value of X:
Screenshot 2023-08-26 120040

Array value of Y:
Screenshot 2023-08-26 120053

Values of Y prediction:
Screenshot 2023-08-26 120104

Values of Y test:
Screenshot 2023-08-26 120116

Training Set Graph:
Screenshot 2023-08-26 120137

Test Set Graph:
Screenshot 2023-08-26 120210

Values of MSE, MAE and RMSE:
Screenshot 2023-08-26 120222

Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
