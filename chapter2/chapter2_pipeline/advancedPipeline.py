import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    'Tuổi': [25, np.nan, 30, 22, 40, np.nan, 35, 28], 
    'ThànhPhố': ['HCM', 'HN', 'HCM', 'ĐN', 'HN', 'ĐN', 'HCM', 'HN'],
    'MuaXe': [0, 1, 0, 0, 1, 1, 0, 1]
})

X = df.drop('MuaXe', axis=1)
y = df['MuaXe']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

num_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('nhanh_so', num_pipe, ['Tuổi']),
        ('nhanh_chu', cat_pipe, ['ThànhPhố'])
    ]
)

full_pipeline = Pipeline(steps=[
    ('tien_xu_ly', preprocessor),
    ('mo_hinh', LogisticRegression())
])

print("Đang chạy Pipeline nâng cao...")
full_pipeline.fit(X_train, y_train)
score = full_pipeline.score(X_test, y_test)
print(f"Độ chính xác: {score*100:.1f}%")