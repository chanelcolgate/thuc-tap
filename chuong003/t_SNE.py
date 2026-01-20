from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')

X_tsne = tsne.fit_transform(X_std)

print(f"Kich thuoc ban dau: {X_std.shape}")
print(f"Kich thuoc sau t-SNE: {X_tsne.shape}")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='jet', alpha=0.6, s=15)

plt.colorbar(scatter, label='Gia tri chu so (0-9)')
plt.title('t-SNE Visualization of Handwritten Digits')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()
# Kich thuoc ban dau: (1797, 64)
# Kich thuoc sau t-SNE: (1797, 2)