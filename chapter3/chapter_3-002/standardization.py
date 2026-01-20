import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

data = load_wine()
X = data.data
print(f"Dữ liệu gốc 2 dòng đầu:\n{X[:2, :3]} ...") 

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

print(f"Dữ liệu sau chuẩn hóa 2 dòng đầu:\n{X_std[:2, :3]} ...")
print(f"\nTrung bình (Mean) sau khi scale: {np.mean(X_std):.2f} (Xấp xỉ 0)")
print(f"Độ lệch (Std) sau khi scale:   {np.std(X_std):.2f} (Bằng 1)")




# kết quả
# Dữ liệu gốc 2 dòng đầu:
# [[14.23  1.71  2.43]
#  [13.2   1.78  2.14]] ...
# Dữ liệu sau chuẩn hóa 2 dòng đầu:
# [[ 1.51861254 -0.5622498   0.23205254]
#  [ 0.24628963 -0.49941338 -0.82799632]] ...

# Trung bình (Mean) sau khi scale: 0.00 (Xấp xỉ 0)
# Độ lệch (Std) sau khi scale:   1.00 (Bằng 1)