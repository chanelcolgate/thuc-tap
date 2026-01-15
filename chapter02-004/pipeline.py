import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

data = {
    'Gio_Hoc': [10, 2, 8, 4, np.nan, 1], 
    'Mon_Hoc': ['Toan', 'Van', 'Toan', 'Van', 'Anh', 'Anh'], 
    'Ket_Qua': [1, 0, 1, 0, 1, 0] # 1: Đậu, 0: Trượt
}
df = pd.DataFrame(data)
X = df.drop('Ket_Qua', axis=1)
y = df['Ket_Qua']

preprocessor = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), ['Gio_Hoc']),
    (make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder()), ['Mon_Hoc'])
)

model = make_pipeline(preprocessor, RandomForestClassifier())

model.fit(X, y)

new_student = pd.DataFrame({'Gio_Hoc': [9], 'Mon_Hoc': ['Toan']})
print(f"Kết quả dự đoán: {'Đậu' if model.predict(new_student)[0] == 1 else 'Trượt'}")