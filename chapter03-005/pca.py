import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print(f"Số đặc trưng gốc: {X_train.shape[1]}")
print(f"Số đặc trưng sau PCA (giữ 95% thông tin): {pca.n_components_}")

classifier = LogisticRegression()
classifier.fit(X_train_pca, y_train)

y_pred = classifier.predict(X_test_pca)

print("\n--- Kết quả phân loại ---")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Phương sai riêng lẻ')
plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid', label='Phương sai tích lũy')
plt.ylabel('Tỷ lệ phương sai giải thích')
plt.xlabel('Chỉ số thành phần chính')
plt.legend(loc='best')
plt.title('Phân tích phương sai của PCA')
plt.show()

#kết quả chạy code
#Số đặc trưng gốc: 30
#Số đặc trưng sau PCA (giữ 95% thông tin): 10
--- Kết quả phân loại ---
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        43
           1       0.99      0.99      0.99        71

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114