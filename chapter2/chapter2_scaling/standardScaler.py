import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = np.array([
    [25, 10000000], [30, 20000000], [45, 50000000], 
    [22, 12000000], [35, 30000000], [50, 80000000],
    [20, 11000000], [40, 45000000]
])
y = np.array([0, 0, 1, 0, 1, 1, 0, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("Dữ liệu Train sau khi scale:\n", X_train_scaled)
print(f"Độ chính xác (Accuracy): {accuracy_score(y_test, y_pred)}")