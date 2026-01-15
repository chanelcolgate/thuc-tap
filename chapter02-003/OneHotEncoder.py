from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

data = [['Hà Nội'], ['Sài Gòn'], ['Đà Nẵng'], ['Sài Gòn']]
df = pd.DataFrame(data, columns=['Thanh_Pho'])

encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df[['Thanh_Pho']]) 

encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Thanh_Pho']))

print("Dữ liệu sau khi One-Hot Encoding:")
print(encoded_df)