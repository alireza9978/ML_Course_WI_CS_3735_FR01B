import kagglehub
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_percentage_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.A1.MyRegressions import LogisticRegression as MyLogisticRegression

# Download latest version
path = kagglehub.dataset_download("uciml/iris")

print("Path to dataset files:", path)

df = pd.read_csv(path + "/Iris.csv")
print(df)

x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df[['Species']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
le = LabelEncoder()
y_train['Species'] = le.fit_transform(y_train['Species'])
y_test['Species'] = le.transform(y_test['Species'])

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

clf = LogisticRegression(random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_train)
print("Train Accuracy: ", accuracy_score(y_train, y_pred))
y_pred = clf.predict(x_test)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


### Part 2
print("PART 2")
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

my_clf = MyLogisticRegression(num_classes=3)
my_clf.fit(x_train, y_train)
print(my_clf.cost_history)
y_pred = my_clf.predict(x_train)
print("Train Accuracy: ", accuracy_score(y_train, y_pred))
y_pred = my_clf.predict(x_test)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


