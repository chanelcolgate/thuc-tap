from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import time

X, y = load_digits(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)

start = time.time()
X_tsne = tsne.fit_transform(X_scaled)
end = time.time()

print(f"Hoàn thành trong: {end - start:.2f} giây")
print(f"Kích thước sau khi nén: {X_tsne.shape}")

# kết quả 
# Hoàn thành trong: 5.80 giây
# Kích thước sau khi nén: (1797, 2)