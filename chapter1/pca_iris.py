from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data  
y = iris.target

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X)

print(f"Dữ liệu gốc: {X.shape}")     
print(f"Dữ liệu sau khi ép: {X_reduced.shape}")