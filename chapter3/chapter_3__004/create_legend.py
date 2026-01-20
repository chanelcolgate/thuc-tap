import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(X))

plt.figure(figsize=(12, 10))
colors = plt.cm.rainbow(np.linspace(0, 1, 10)) 

for digit in range(10):
    indices = y == digit
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], 
                color=colors[digit], 
                label=f'Số {digit}', 
                alpha=0.6)

plt.legend(fontsize=12, loc='best', title="Các con số")
plt.title("Phân tích cụm t-SNE")
plt.grid(True)
plt.show()

# INSIGHTS TỪ BIỂU ĐỒ
# đám mây số 0 (Màu đỏ): Nó tách biệt hoàn toàn -> Số 0 rất dễ nhận diện.
# đám mây số 1 và số 9: Có thể chúng hơi dính vào nhau.
# Kết luận: t-SNE đã gom nhóm cực tốt các ảnh 64 chiều về 2 chiều mà vẫn giữ được cấu trúc.