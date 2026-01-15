from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data_imputed = {
    'Dien_Tich': [50, 60, 70, 80, 90],
    'Gia_Nha': [110, 130, 150, 170, 190]
}
df_ready = pd.DataFrame(data_imputed)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_ready)
df_final = pd.DataFrame(data_scaled, columns=['Dien_Tich_Scaled', 'Gia_Nha_Scaled'])

print("Dữ liệu sau khi Điền thiếu VÀ Chuẩn hóa:")
print(df_final)