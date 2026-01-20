import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(12, 10))

scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='jet', alpha=0.7)

plt.title("t-SNE Visualization: Bộ số viết tay (Digits)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.colorbar(scatter, ticks=range(10), label='Digit Label (0-9)')
plt.show()