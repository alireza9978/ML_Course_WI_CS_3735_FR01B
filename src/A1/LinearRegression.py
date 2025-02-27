import kagglehub
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.A1.MyRegressions import LinearRegression as MyLinearRegression
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


# Download latest version
path = kagglehub.dataset_download("hanaksoy/customer-purchasing-behaviors")

print("Path to dataset files:", path)

df = pd.read_csv(path + "/Customer Purchasing Behaviors.csv")

X = df[['age', 'annual_income', 'purchase_amount', 'purchase_frequency', 'region']]
y = df['loyalty_score']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
le = LabelEncoder()
x_train['region'] = le.fit_transform(x_train['region'])
x_test['region'] = le.transform(x_test['region'])

plt.figure(figsize=(11,8))
sns.heatmap(pd.concat([x_train, y_train], axis=1).corr(), cmap="Greens",annot=True)
plt.savefig("result/A1/corr.jpeg")

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
print("PART 2")
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

my_reg = MyLinearRegression()
total_rounds = my_reg.fit(x_train, y_train)
print("Train R^2 Score: ", my_reg.score(x_train, y_train))
print("Test R^2 Score: ", my_reg.score(x_test, y_test))
y_pred = my_reg.predict(x_test)
print("MyRegression Test MAPE: ", mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred))



