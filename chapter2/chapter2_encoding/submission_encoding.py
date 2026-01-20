import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

def get_data():
    df = pd.DataFrame({
        'Tuổi': [20, 30, 25, 40, 50, 22, 35, 28],
        'Màu': ['Đỏ', 'Xanh', 'Đỏ', 'Vàng', 'Đỏ', 'Xanh', 'Vàng', 'Đỏ'],
    })
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    return train_test_split(df, y, test_size=0.25, random_state=42)

def cham_diem(X_train, X_test, y_train, y_test, encoder, name):
    preprocessor = ColumnTransformer(
        transformers=[
            ('xu_ly_mau', encoder, ['Màu']),
            ('xu_ly_tuoi', StandardScaler(), ['Tuổi'])
        ]
    )
    
    model = make_pipeline(preprocessor, LogisticRegression())
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    print(f"👉 Phương pháp {name:<20} | Độ chính xác: {score*100:.1f}%")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()
    print("--- SO SÁNH HIỆU QUẢ ENCODING ---\n")

    cham_diem(X_train, X_test, y_train, y_test, 
              OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), "Ordinal (Gán số)")

    cham_diem(X_train, X_test, y_train, y_test, 
              OneHotEncoder(sparse_output=False, handle_unknown='ignore'), "OneHot (Tách cột)")