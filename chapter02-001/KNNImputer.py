import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

data = {
    'Tuoi': [20, 22, 40, 42, 21],
    'Thu_Nhap': [200, 220, 800, 820, np.nan]
}
df = pd.DataFrame(data)

print("Dữ liệu ban đầu:")
print(df)

imputer = KNNImputer(n_neighbors=2)
df_filled = imputer.fit_transform(df)
df_result = pd.DataFrame(df_filled, columns=['Tuoi', 'Thu_Nhap'])

print("\nDữ liệu sau khi xử lý thiếu bằng KNNImputer:")
print(df_result)