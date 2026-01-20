import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

X, y = make_classification(n_samples=500, n_features=10, n_informative=2, 
                           n_redundant=0, random_state=42, shuffle=False)

print(f"--- Dữ liệu gốc: {X.shape} (10 cột) ---")
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)

print("Độ quan trọng các cột:", clf.feature_importances_)

selector = SelectFromModel(clf, threshold='mean', prefit=True)
X_new = selector.transform(X)

print(f"\n--- SAU KHI LỌC: {X_new.shape} ---")