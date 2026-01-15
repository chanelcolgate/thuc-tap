import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

X_raw = np.array([
    ["Đỏ"], 
    ["Xanh"], 
    ["Vàng"], 
    ["Đỏ"], 
    ["Vàng"]
])

print("--- DỮ LIỆU GỐC ---")
print(X_raw)

encoder = OneHotEncoder(sparse_output=False)

X_onehot = encoder.fit_transform(X_raw)

print("\n--- KẾT QUẢ ONE-HOT (Ma trận số 0 và 1) ---")
print(X_onehot)

print("\n--- TÊN CÁC CỘT MỚI ---")
print(encoder.get_feature_names_out(['Màu']))

df_onehot = pd.DataFrame(X_onehot, columns=encoder.get_feature_names_out(['Màu']))
print("\n--- BẢNG DỮ LIỆU CUỐI CÙNG ---")
print(df_onehot)