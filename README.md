# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4.Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5.Determine training and test data set.
6.Apply decision tree Classifier on to the dataframe
7.Get the values of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: AMURTHA VAAHINI.KN
RegisterNumber:  212222240008
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
## Initial data set:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679102/ceda7e48-5367-4d0b-af53-01d7c3e677c4)
## Data info:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679102/64f8f988-bb66-439f-a476-c8c2c430f955)
## Optimization of null values:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679102/71273186-38b6-416e-8617-03f93037e8ae)
## Assignment of x and y values:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679102/58d2ce46-5dcd-44b2-b32d-c52944618948)
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679102/08702070-a680-49ea-a025-9d6032db4f2e)
## Converting string literals to numerical values using label encoder:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679102/25aaef04-a446-4687-b2bc-ce9abc3be426)
## Accuracy:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679102/7232b75a-10b5-45a9-9166-c2d6ed6087b1)
## Prediction:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118679102/81eae1b0-660e-43f8-a43e-52447c0987d5)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
