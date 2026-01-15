from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

data = [
    [8.0, 7.5, 9.0],
    [7.0, np.nan, 8.0],
    [np.nan, 6.5, 7.5],
    [9.0, 8.0, np.nan],
    [8.5, 7.0, 8.0],
]
df = pd.DataFrame(data, columns=["Toan", "Van", "Anh"])
print("Dữ liệu ban đầu:")
print(df)
imputer = SimpleImputer(strategy="mean")
imputed_data = imputer.fit_transform(df)
imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
imputed_df
print("\nDữ liệu sau khi xử lý thiếu:")
print(imputed_df)

# PR #2 - handle missing data using SimpleImputer


