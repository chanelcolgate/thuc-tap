import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = load_wine(return_X_y=True)
target_names = load_wine().target_names
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_scatter(ax, X_data, title):
    colors = ['r', 'b', 'g']
    for label, color in zip([0, 1, 2], colors):
        ax.scatter(X_data[y == label, 0], 
                   X_data[y == label, 1], 
                   c=color, label=target_names[label], alpha=0.7)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

plot_scatter(ax1, X_pca, "PCA: Các nhóm vẫn hơi dính nhau")
plot_scatter(ax2, X_lda, "LDA: Các nhóm tách biệt hoàn toàn")

plt.show()


#  LDA đẩy 3 nhóm (Đỏ, Xanh, Lục) ra xa nhau hơn hẳn PCA