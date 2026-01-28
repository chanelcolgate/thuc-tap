import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

X = np.array([[2], [4], [5], [6], [8], [9], [10]])

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')

X_binned = est.fit_transform(X)

print("Các cột mốc chia thùng:", est.bin_edges_)
print("\nDữ liệu sau khi chia vào 3 nhóm (0, 1, 2):")
print(X_binned)