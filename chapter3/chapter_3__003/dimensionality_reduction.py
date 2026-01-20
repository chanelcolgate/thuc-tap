from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = load_wine(return_X_y=True)
n_features = X.shape[1]    
n_classes = len(set(y))     

print(f"Dữ liệu có {n_features} đặc trưng và {n_classes} loại rượu.")

lda_2 = LinearDiscriminantAnalysis(n_components=2)
X_new = lda_2.fit_transform(X, y)
print(f"LDA 2 chiều thành công: {X_new.shape}")

try:
    print(" nén xuống 5 chiều")
    lda_5 = LinearDiscriminantAnalysis(n_components=5)
    X_new_5 = lda_5.fit_transform(X, y)
    print(f"Kết quả thực tế: {X_new_5.shape}")
except Exception as e:
    print(e)


#     Dữ liệu có 13 đặc trưng và 3 loại rượu.
# LDA 2 chiều thành công: (178, 2)
#  nén xuống 5 chiều
# n_components cannot be larger than min(n_features, n_classes - 1).
# nó chỉ giảm đc toosi đa n_classes -1 tức là 3 loại chỉ xuống được 2