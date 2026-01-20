import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

print("--- 1. CHUẨN BỊ DỮ LIỆU ---")
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = (X ** 2).flatten() + np.random.randn(100) * 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")

print("\n--- 2. CHẠY THỬ NGHIỆM SO SÁNH (BENCHMARK) ---")

model_org = LinearRegression()
model_org.fit(X_train, y_train)
score_org = r2_score(y_test, model_org.predict(X_test))

print(f" Model A (Gốc - Không làm gì): Score = {score_org:.4f}")

model_fe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)), 
    ('model', LinearRegression())
])
model_fe.fit(X_train, y_train)
score_fe = r2_score(y_test, model_fe.predict(X_test))

print(f" Model B (Có Feature Engineering): Score = {score_fe:.4f}")

print("\n--- 3. KẾT LUẬN ---")
if score_fe > score_org:
    print(" Feature Engineering đã cải thiện Model đáng kể!")
    print(f"   (Tăng từ {score_org:.2f} lên {score_fe:.2f})")
else:
    print(" Không hiệu quả.")