import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

X = np.array([
    [10, 20],
    [5, 10],
    [2, 5],
    [8, 15]
])
df_org = pd.DataFrame(X, columns=['Dài', 'Rộng'])

print("--- 1. DỮ LIỆU GỐC ---")
print(df_org)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

feature_names = poly.get_feature_names_out(['Dài', 'Rộng'])
df_poly = pd.DataFrame(X_poly, columns=feature_names)

print("\n--- 2. SAU KHI BIẾN ĐỔI (POLYNOMIAL) ---")
print(df_poly)