import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data 
y = iris.target 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
final_df = pd.concat([pca_df, pd.Series(y, name='target')], axis=1)

plt.figure(figsize=(10, 7))
colors = ['r', 'g', 'b']
labels = iris.target_names

for target, color, label in zip([0, 1, 2], colors, labels):
    indicesToKeep = final_df['target'] == target
    plt.scatter(final_df.loc[indicesToKeep, 'PC1'],
                final_df.loc[indicesToKeep, 'PC2'],
                c=color, s=50, label=label)

plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.title('Biểu đồ PCA - Giảm chiều dữ liệu Iris (4D về 2D)')
plt.legend()
plt.grid(True)
plt.show()