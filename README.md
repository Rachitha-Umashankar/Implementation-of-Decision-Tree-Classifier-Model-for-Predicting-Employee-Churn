# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load employee data and split it into training and testing sets.

2.Train a Decision Tree classifier using entropy as the split criterion.

3.Evaluate the model using accuracy, confusion matrix, and classification report.

4.Use the trained model to predict whether a new employee will stay or leave.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rachitha U
RegisterNumber: 212225220078
*/

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
data = {
    'Satisfaction_Level': [0.8, 0.4, 0.9, 0.3, 0.7, 0.2, 0.85, 0.5, 0.95, 0.45],
    'Last_Evaluation': [0.9, 0.6, 0.8, 0.5, 0.75, 0.4, 0.88, 0.7, 0.95, 0.65],
    'Number_of_Projects': [5, 3, 6, 2, 4, 2, 6, 3, 7, 3],
    'Average_Monthly_Hours': [220, 150, 250, 120, 200, 100, 240, 160, 260, 140],
    'Years_at_Company': [3, 2, 4, 1, 3, 1, 4, 2, 5, 2],
    'Churn': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
print("Employee Data:\n", df, "\n")
X = df[['Satisfaction_Level', 'Last_Evaluation', 'Number_of_Projects',
        'Average_Monthly_Hours', 'Years_at_Company']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(12, 6))
plot_tree(model,
          feature_names=X.columns,
          class_names=['Stayed', 'Left'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree for Employee Churn Prediction")
plt.show()
new_emp = [[0.4, 0.6, 3, 150, 2]]  
prediction = model.predict(new_emp)
print("\nNew Employee Prediction:")
if prediction[0] == 1:
    print(" Employee is likely to LEAVE (Churn).")
else:
    print(" Employee is likely to STAY.")
```

## Output:
<img width="747" height="518" alt="Screenshot 2026-02-10 093755" src="https://github.com/user-attachments/assets/937dfd54-83e5-47d3-b5a6-71a175e53e10" />
<img width="674" height="332" alt="Screenshot 2026-02-10 093821" src="https://github.com/user-attachments/assets/368b4499-490b-405b-8e72-709b5bd83fa9" />
<img width="1204" height="707" alt="Screenshot 2026-02-10 093853" src="https://github.com/user-attachments/assets/2eab656e-8191-450e-84c8-e9432b1675e8" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
