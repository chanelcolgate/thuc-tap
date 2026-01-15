from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

data_ready = {
    'Dien_Tich': [50, 60, 70, 80, 90],
    'Gia_Nha': [110, 130, 150, 170, 190]
}
df = pd.DataFrame(data_ready)

scaler = MinMaxScaler()
df_minmax = scaler.fit_transform(df)
df_final = pd.DataFrame(df_minmax, columns=['Dien_Tich_01', 'Gia_Nha_01'])

print("Dữ liệu sau khi nén vào khoảng 0-1:")
print(df_final)

print(f"\nGiá trị nhỏ nhất mới: {df_final.min().values}")
print(f"Giá trị lớn nhất mới: {df_final.max().values}")