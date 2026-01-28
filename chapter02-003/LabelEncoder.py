from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

y_labels = ['Táo', 'Cam', 'Xoài', 'Cam', 'Táo']
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)

print("Nhãn sau khi mã hóa số:", y_encoded)
print("Danh sách các lớp máy đã học được:", le.classes_)

prediction_numbers = [1, 0, 2]
original_labels = le.inverse_transform(prediction_numbers)

print("\nDự đoán từ số dịch ra chữ cho con người đọc:")
print(original_labels)