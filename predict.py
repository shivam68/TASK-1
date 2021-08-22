# Importing the required Functions
import pandas
from sklearn.linear_model import LinearRegression

# Storing my Data in db varaible
db=pandas.read_csv("Salary_Data.csv")
x=db["YearsExperience"]
y=db["Salary"]

# Converting Series Datatype into ndarry
x=x.values.reshape(-1,1)

# Creatig a Model
model=LinearRegression()
model.fit(x,y)

#Printing the Calculated Weight and Biased
print("Weight is : ",model.coef_)
print("Biased is : ",model.intercept_)

#Predicting the Values:
value=float(input("Enter the value to be Predicted: "))
print(model.predict([[value]])
