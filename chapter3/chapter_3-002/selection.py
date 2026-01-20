import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

X = load_wine().data
X_std = StandardScaler().fit_transform(X)
cov_mat = np.cov(X_std, rowvar=False)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

eigen_pairs.sort(key=lambda k: k[0], reverse=True)

print("Top 2 Eigenvalues cao nhất:")
print(f"1. {eigen_pairs[0][0]:.4f}")
print(f"2. {eigen_pairs[1][0]:.4f}")

tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
print(f"\nPhần trăm thông tin giữ lại của Top 2: {sum(var_exp[:2]):.2%}")
print("=> sẽ giữ lại 2 chiều đầu tiên này!")


# kết quả
# Top 2 Eigenvalues cao nhất:
# 1. 4.7324
# 2. 2.5111

# Phần trăm thông tin giữ lại của Top 2: 55.41%
# => sẽ giữ lại 2 chiều đầu tiên này!