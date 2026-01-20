import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

X = load_wine().data
X_std = StandardScaler().fit_transform(X)

cov_mat = np.cov(X_std, rowvar=False)

print(f"Kích thước ma trận: {cov_mat.shape} (Vì có 13 đặc trưng nên ra ma trận 13x13)")
print("\nGiá trị 5x5 :")
print(cov_mat[:5, :5])
print("\nSố càng lớn (dương) nghĩa là 2 biến cùng tăng. Số âm là ngược chiều.")


# kết quả Kích thước ma trận: (13, 13) (Vì có 13 đặc trưng nên ra ma trận 13x13)

# Giá trị 5x5 :
# [[ 1.00564972  0.09493026  0.21273976 -0.31198788  0.27232816]
#  [ 0.09493026  1.00564972  0.16497228  0.29013035 -0.05488343]
#  [ 0.21273976  0.16497228  1.00564972  0.44587209  0.28820583]
#  [-0.31198788  0.29013035  0.44587209  1.00564972 -0.0838039 ]
#  [ 0.27232816 -0.05488343  0.28820583 -0.0838039   1.00564972]]

#  Số càng lớn (dương) nghĩa là 2 biến cùng tăng. Số âm là ngược chiều.