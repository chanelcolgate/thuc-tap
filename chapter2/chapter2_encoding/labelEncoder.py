import numpy as np 
from sklearn.preprocessing import LabelEncoder

y_raw = np.array(["Bình thường", "Kém", "Tốt", "Kém", "Tốt", "Xuất sắc"])

print(y_raw)

le = LabelEncoder();
y_encoded = le.fit_transform(y_raw)

print("\n--- KẾT QUẢ LABEL ENCODER (Máy tính hiểu) ---")
print(y_encoded)

for i, label in enumerate(le.classes_):
    print(f"Số {i} đại diện cho: {label}")

so_du_doan = [0, 2, 1]
chu_dich_lai = le.inverse_transform(so_du_doan)
print(f"\nDịch ngược {so_du_doan} -> {chu_dich_lai}")