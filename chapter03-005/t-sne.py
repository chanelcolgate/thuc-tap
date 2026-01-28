import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
X_tsne = tsne.fit_transform(X_std)

plt.figure(figsize=(10, 7))

scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)

legend1 = plt.legend(*scatter.legend_elements(), title="Chữ số", loc="best")
plt.gca().add_artist(legend1)

plt.title('Trực quan hóa t-SNE (Sử dụng Matplotlib thuần)')
plt.xlabel('Thành phần t-SNE 1')
plt.ylabel('Thành phần t-SNE 2')
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()