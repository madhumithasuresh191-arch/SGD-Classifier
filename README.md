# SGD-Classifier
## Name: S Madhumitha
## Reg No:212225040217
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Python libraries and load the Placement_Data (1).csv dataset into Google Colab.
2.Remove unnecessary columns, convert categorical data into numerical values, and separate input features and target variables.
3.Split the dataset into training and testing data, apply feature scaling, and train the SGD Classifier model.
4.Predict the student placement status and evaluate the model using accuracy score and confusion matrix
## Program:
~~~
Program to implement the prediction of iris species using SGD Classifier.
Developed by: S Madhumitha
RegisterNumber:212225040217
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('/content/Placement_Data (1).csv')

data = data.drop(['sl_no', 'salary'], axis=1)

data = pd.get_dummies(data, drop_first=True, dtype=int)

X = data.drop('status_Placed', axis=1)
y = data['status_Placed']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SGDClassifier(
    loss='log_loss',
    learning_rate='constant',
    eta0=0.01,
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Values:", y_test.values)
print("Predicted Values:", y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

if y_pred[0] == 1:
    print("Student is Placed")
else:
    print("Student is Not Placed")  
~~~

## Output:
<img width="1920" height="1020" alt="Screenshot 2026-11-02 081130" src="https://github.com/user-attachments/assets/08cec523-436f-42e6-a8fd-37a1934dc80a" />
<img width="1920" height="1020" alt="Screenshot 2026-11-02 081139" src="https://github.com/user-attachments/assets/723465bc-e50f-4101-be28-7fb59b3754e1" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
