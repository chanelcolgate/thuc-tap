import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names 

X_scaled = StandardScaler().fit_transform(X)

print("Đang chạy t-SNE ")
tsne = TSNE(n_components=2, perplexity=30, random_state=42) 
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
colors = ['red', 'green']

for i, target_name in enumerate(target_names):
    indices = y == i
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], 
                c=colors[i], label=target_name, alpha=0.6)

plt.legend(title="Loại u", loc='best')
plt.title("t-SNE Visualization: Breast Cancer Dataset (30 chiều -> 2 chiều)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.show()

# --- NHẬN XÉT ---
# -thấy 2 nhóm (Ác tính và Lành tính) tách ra thành 2 đám mây riêng biệt.
# - Tuy nhiên, ở giữa vẫn có một vùng giao thoa (các điểm xanh đỏ lẫn lộn).
# => Ý nghĩa: Có những ca bệnh rất khó phân biệt, cần bác sĩ chuyên sâu hoặc model phức tạp hơn.