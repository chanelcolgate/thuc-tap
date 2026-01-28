from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np

data = {
    'Cam': [1.0, 10.0],
    'Tao': [1.0, 10.0]
}
df = pd.DataFrame(data)

normalizer = Normalizer()
data_normalized = normalizer.fit_transform(df)
df_final = pd.DataFrame(data_normalized, columns=['Cam_Normalized', 'Tao_Normalized'])

print("Dữ liệu gốc:")
print(df)
print("\nDữ liệu sau khi Normalizer (Nhìn theo dòng):")
print(df_final)