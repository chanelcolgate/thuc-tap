import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

data = {
    'Gio_Hoc': [10, 2, 8, 4, 7, 1],
    'Co_Giay': [38, 40, 39, 42, 37, 41],     
    'Tien_Tieu_Vat': [50, 20, 45, 30, 40, 10] 
}
X = pd.DataFrame(data)
y = [1, 0, 1, 0, 1, 0] # Kết quả Đậu/Trượt

selector_model = RandomForestClassifier(random_state=42)

selector = SelectFromModel(estimator=selector_model, threshold="mean")

selector.fit(X, y)

selected_columns = X.columns[selector.get_support()]
print(f"Cột quan trọng được giữ lại: {selected_columns.tolist()}")

X_important = selector.transform(X)