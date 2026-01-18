import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- HÀNH ĐỘNG 1: Chuẩn bị dữ liệu Chữ số ---
# Load tập dữ liệu Digits (chữ số viết tay từ 0-9)
digits = datasets.load_digits()
X = digits.data  # Các đặc trưng (pixel)
y = digits.target # Nhãn (0, 1, 2, ..., 9)

# --- BÀI KIỂM TRA: Tiêu chuẩn hóa dữ liệu ---
# t-SNE tính toán dựa trên khoảng cách, nên các đặc điểm phải cùng thang đo
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# --- HÀNH ĐỘNG 2 & BÀI KIỂM TRA: Tạo quy trình t-SNE ---
# n_components=2: Giảm xuống không gian 2D để vẽ biểu đồ
# perplexity: Có thể hiểu là số lượng láng giềng mà mỗi điểm quan tâm (thường từ 5-50)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_scaled)

# --- HÀNH ĐỘNG 3 & 4: Trực quan hóa và Tạo huyền thoại (Legend) ---
plt.figure(figsize=(10, 8))
# Vẽ biểu đồ phân tán, mỗi màu đại diện cho một chữ số
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.8, edgecolors='none')

# Tạo huyền thoại (Legend) để biết màu nào là số mấy
legend = plt.legend(*scatter.legend_elements(), title="Chữ số", loc="best")
plt.gca().add_artist(legend)

plt.title('Trực quan hóa t-SNE: Các chữ số được phân cụm trong không gian 2D')
plt.xlabel('Trục t-SNE 1')
plt.ylabel('Trục t-SNE 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()