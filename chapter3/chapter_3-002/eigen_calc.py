import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

X = load_wine().data
X_std = StandardScaler().fit_transform(X)
cov_mat = np.cov(X_std, rowvar=False)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print("\n1. Eigenvalues (Độ quan trọng của từng chiều):")
print(eigen_vals)
print("\n2. Eigenvectors (Hướng của từng chiều - In cột đầu tiên):")
print(eigen_vecs[:, 0])

print("\nNhững số Eigenvalue rất to (chiều quan trọng) và số rất bé (rác).")

# kết quả 
# 1. Eigenvalues (Độ quan trọng của từng chiều):
# [4.73243698 2.51108093 1.45424187 0.92416587 0.85804868 0.64528221
#  0.55414147 0.10396199 0.35046627 0.16972374 0.29051203 0.22706428
#  0.25232001]

# 2. Eigenvectors (Hướng của từng chiều - In cột đầu tiên):
# [-0.1443294   0.24518758  0.00205106  0.23932041 -0.14199204 -0.39466085
#  -0.4229343   0.2985331  -0.31342949  0.0886167  -0.29671456 -0.37616741
#  -0.28675223]

# Những số Eigenvalue rất to (chiều quan trọng) và số rất bé (rác).