import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

data = {
    'Tuoi': [25, 30, np.nan, 35],
    'Thu_Nhap': [50, 80, 70, np.nan],
    'Thanh_Pho': ['Hà Nội', 'Sài Gòn', np.nan, 'Đà Nẵng']
}
df = pd.DataFrame(data)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Tuoi', 'Thu_Nhap']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Thanh_Pho'])
    ],
    remainder='passthrough' 
)

data_final = preprocessor.fit_transform(df)

print("Dữ liệu sau khi qua trạm điều phối ColumnTransformer:")
print(data_final)