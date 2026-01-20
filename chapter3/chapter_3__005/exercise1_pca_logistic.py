import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target
print(f"Dữ liệu Ung thư vú: {X.shape} ")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

start_a = time.time()
pipe_full = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
pipe_full.fit(X_train, y_train)
acc_full = pipe_full.score(X_test, y_test)
time_a = time.time() - start_a

start_b = time.time()
pipe_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)), 
    ('model', LogisticRegression())
])
pipe_pca.fit(X_train, y_train)
acc_pca = pipe_pca.score(X_test, y_test)
time_b = time.time() - start_b

print("\n--- KẾT QUẢ SO SÁNH ---")
print(f"1. Full (30 features): Acc = {acc_full:.4f} | Time = {time_a:.6f}s")
print(f"2. PCA  (2 features):  Acc = {acc_pca:.4f}  | Time = {time_b:.6f}s")

ratio_acc = acc_pca / acc_full
print(f"\n=> PCA giữ được {ratio_acc:.2%} độ chính xác so với bản gốc.")


# Dữ liệu Ung thư vú: (569, 30)   (30 đặc trưng)

# --- KẾT QUẢ SO SÁNH ---
# 1. Full (30 features): Acc = 0.9825 | Time = 0.007590s
# 2. PCA  (2 features):  Acc = 0.9766  | Time = 0.006656s

# => PCA giữ được 99.40% độ chính xác so với bản gốc.
# => PCA chạy nhanh hơn và quan trọng là giảm độ phức tạp tính toán đi 15 lần (30 -> 2).