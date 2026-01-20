import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_std = StandardScaler().fit_transform(X)

pca = PCA(n_components=2) 

X_pca = pca.fit_transform(X_std)

print(f"Kích thước gốc: {X.shape}")
print(f"Kích thước sau PCA: {X_pca.shape}")
print(f"Tỉ lệ thông tin giữ lại được: {sum(pca.explained_variance_ratio_) * 100:.2f}%")

plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=2,
                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Ket qua PCA tren bo du lieu Iris')
plt.xlabel('Thanh phan chinh 1 (PC1)')
plt.ylabel('Thanh phan chinh 2 (PC2)')
plt.show()

# Kích thước gốc: (150, 4)
# Kích thước sau PCA: (150, 2)        
# Tỉ lệ thông tin giữ lại được: 95.81%