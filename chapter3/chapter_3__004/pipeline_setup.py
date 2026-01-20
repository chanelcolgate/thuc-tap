import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data
y = digits.target

print(f"Kích thước dữ liệu gốc: {X.shape}")
print("Mỗi dòng là một tấm ảnh 8x8 pixel được duỗi phẳng thành 64 cột.")

plt.imshow(digits.images[0], cmap='gray')
plt.title(f"Đây là số: {y[0]}")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# kết quả 
# Kích thước dữ liệu gốc: (1797, 64)
# Mỗi dòng là một tấm ảnh 8x8 pixel được duỗi phẳng thành 64 cột.
