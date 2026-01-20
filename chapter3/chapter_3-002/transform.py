import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

X = load_wine().data
y = load_wine().target
X_std = StandardScaler().fit_transform(X)
cov_mat = np.cov(X_std, rowvar=False)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))

X_pca = X_std.dot(w)

print(f"Kích thước ban đầu: {X_std.shape}")
print(f"Kích thước sau PCA: {X_pca.shape}")

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y), colors, markers):
    plt.scatter(X_pca[y==l, 0], X_pca[y==l, 1], c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Loại rượu')
plt.title('Kết quả cuối cùng của PCA Thủ công')
plt.show()


# kết quả 
# Kích thước ban đầu: (178, 13)
# Kích thước sau PCA: (178, 2)
