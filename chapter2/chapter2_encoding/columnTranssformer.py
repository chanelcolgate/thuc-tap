import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = {
    'ThànhPhố': ['Hà Nội', 'HCM', 'Đà Nẵng', 'Hà Nội', 'HCM'],
    'Tuổi': [25, 30, 35, 22, 40],
    'Lương': [1000, 2000, 3000, 1100, 5000]
}
df = pd.DataFrame(data)

print("--- DỮ LIỆU GỐC  ---")
print(df)

cat_features = ['ThànhPhố']
cat_transformer = OneHotEncoder(sparse_output=False)

num_features = ['Tuổi', 'Lương']
num_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('xu_ly_chu', cat_transformer, cat_features),
        ('xu_ly_so', num_transformer, num_features)
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(df)

print("\n--- KẾT QUẢ SAU KHI QUA COLUMN TRANSFORMER ---")
print(np.round(X_processed, 2))

print("\n--- GIẢI MÃ KẾT QUẢ ---")
print("3 cột đầu là OneHot (Đà Nẵng, Hà Nội, HCM). 2 cột sau là Số đã Scale (Tuổi, Lương)")