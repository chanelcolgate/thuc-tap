import numpy as np 
from sklearn.impute import KNNImputer;

x = np.array([
    [8, 9, 7],
    [7, np.nan, 8],
    [np.nan, 6, 9],
    [5, 5, np.nan],
    [10, 10, 10]     
])

print("--data gốc--");
print(x);

imputer_knn = KNNImputer(n_neighbors=2);

print("sử dụng knn imputer để tính giá trị thay cho chổ đang thiếu");
x_knn = imputer_knn.fit_transform(x);
print(x_knn);

