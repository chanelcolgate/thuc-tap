from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
acc_org = model.score(X_test, y_test)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y) 

X_train_lda, X_test_lda, y_train, y_test = train_test_split(X_lda, y, test_size=0.3, random_state=42)
model_lda = LogisticRegression()
model_lda.fit(X_train_lda, y_train)
acc_lda = model_lda.score(X_test_lda, y_test)

print(f"Độ chính xác (Gốc - 13 chiều): {acc_org:.4f}")
print(f"Độ chính xác (LDA - 2 chiều):  {acc_lda:.4f}")

if acc_lda >= acc_org:

#     Độ chính xác (Gốc - 13 chiều): 0.9815
# Độ chính xác (LDA - 2 chiều):  1.0000
# => KẾT LUẬN: LDA Chỉ dùng 2 chiều mà kết quả ngang ngửa (hoặc hơn) dùng 13 chiều