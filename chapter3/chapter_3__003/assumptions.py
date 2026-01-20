import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
import pandas as pd

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

plt.figure(figsize=(10, 6))
for label in [0, 1, 2]:
    sns.kdeplot(df[df['target'] == label]['alcohol'], label=f'Loại {label}', fill=True)

plt.title("Kiểm tra phân phối của 'Alcohol' ")
plt.legend()
plt.show()

# Nhận xét biểu đồ
#  Nếu 3 hình này trông giống cái chuông úp ngược -> Tốt (Thỏa mãn Assumption)."
# Nếu nó méo mó, xiêu vẹo -> LDA sẽ hoạt động kém đi một chút."