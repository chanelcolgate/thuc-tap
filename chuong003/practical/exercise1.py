import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline_no_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
pipeline_with_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
])
pipeline_no_pca.fit(X_train, y_train)
pipeline_with_pca.fit(X_train, y_train)

y_pred_no_pca = pipeline_no_pca.predict(X_test)
y_pred_with_pca = pipeline_with_pca.predict(X_test)
acc_no_pca = accuracy_score(y_test, y_pred_no_pca)
acc_with_pca = accuracy_score(y_test, y_pred_with_pca)

print(f"Do chinh xac khong dung PCA: {acc_no_pca:.4f}")
print(f"Do chinh xac co dung PCA:    {acc_with_pca:.4f}")

# Do chinh xac khong dung PCA: 1.0000
# Do chinh xac co dung PCA:    0.9000