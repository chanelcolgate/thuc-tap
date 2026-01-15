import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.DataFrame({
    'Tuổi': [25, np.nan, 30, 22, 40, 50, 35, 28],
    'Lương': [1000, 2000, np.nan, 1200, 5000, 6000, 3200, 1500],
    'MuaXe': [0, 1, 0, 0, 1, 1, 0, 1]
})

X = df[['Tuổi', 'Lương']]
y = df['MuaXe']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

simple_pipeline = Pipeline(steps=[
    ('dien_khuyet', SimpleImputer(strategy='mean')),
    ('co_gian', StandardScaler()),
    ('mo_hinh', LogisticRegression())
])

print("Đang huấn luyện Pipeline cơ bản...")
simple_pipeline.fit(X_train, y_train)

y_pred = simple_pipeline.predict(X_test)
print(f"Độ chính xác: {accuracy_score(y_test, y_pred)*100:.1f}%")