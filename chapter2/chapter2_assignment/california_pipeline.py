import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

raw_data = fetch_california_housing()

df_features = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
target_price = raw_data.target 

print(f"   Dữ liệu có {df_features.shape[0]} căn nhà và {df_features.shape[1]} đặc trưng.")

X_train, X_test, y_train, y_test = train_test_split(
    df_features, target_price, test_size=0.2, random_state=42
)

my_model_pipeline = Pipeline(steps=[
    ('lam_sach_du_lieu', StandardScaler()),
    ('tao_dac_trung_moi', PolynomialFeatures(degree=2, include_bias=False)),
    ('bo_nao_du_doan', LinearRegression())
])

my_model_pipeline.fit(X_train, y_train)

y_pred = my_model_pipeline.predict(X_test)

rmse_score = np.sqrt(mean_squared_error(y_test, y_pred)) 
r2_acc = r2_score(y_test, y_pred) 

print("KẾT QUẢ ĐÁNH GIÁ")
print(f"Sai số trung bình (RMSE): {rmse_score:.4f}")
print(f"Độ chính xác (R2 Score):  {r2_acc:.2%}")

if r2_acc > 0.6:
    print(" Đánh giá: Mô hình hoạt động TỐT.")
else:
    print(" Đánh giá: Mô hình cần cải thiện thêm.")