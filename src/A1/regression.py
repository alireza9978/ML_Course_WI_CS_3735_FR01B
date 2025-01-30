import kagglehub
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from src.A1.MyRefressions import LinearRegression as MyLinearRegression
# Download latest version
path = kagglehub.dataset_download("hanaksoy/customer-purchasing-behaviors")

print("Path to dataset files:", path)

df = pd.read_csv(path + "/Customer Purchasing Behaviors.csv")

X = df[['age', 'annual_income', 'purchase_amount', 'loyalty_score', 'region']]
y = df['purchase_frequency']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
le = LabelEncoder()
x_train['region'] = le.fit_transform(x_train['region'])
x_test['region'] = le.transform(x_test['region'])

### Part 1
reg = LinearRegression()
reg.fit(x_train, y_train)
print("Train R^2 Score: ", reg.score(x_train, y_train))
print("Test R^2 Score: ", reg.score(x_test, y_test))
# print(reg.coef_)
# print(reg.intercept_)
y_pred = reg.predict(x_test)
print("Test MAPE: ", mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred))

### Part 2
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

my_reg = MyLinearRegression()
total_rounds = my_reg.fit(x_train, y_train)
print("Train R^2 Score: ", my_reg.score(x_train, y_train))
print("Test R^2 Score: ", my_reg.score(x_test, y_test))
y_pred = my_reg.predict(x_test)
print("MyRegression Test MAPE: ", mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred))



