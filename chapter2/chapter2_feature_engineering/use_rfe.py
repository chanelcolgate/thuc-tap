import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=500, n_features=10, n_informative=3, 
                           n_redundant=0, random_state=42, shuffle=False)
col_names = [f"Feature_{i}" for i in range(10)]

print(f"--- Dữ liệu gốc: {X.shape} ---")

model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=3, step=1)

rfe.fit(X, y)
print("\n--- KẾT QUẢ RFE ---")
df_result = pd.DataFrame({
    'Tên Cột': col_names,
    'Được Chọn?': rfe.support_,    
    'Thứ Hạng': rfe.ranking_      
})

print(df_result.sort_values(by='Thứ Hạng'))