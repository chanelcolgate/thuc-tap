import numpy as np 
from sklearn.impute import SimpleImputer;

x = np.array([
    [8, 9, 7],
    [7, np.nan, 8],
    [np.nan, 6, 9],
    [5, 5, np.nan],
    [10, 10, 10]     
])

print("--data gốc--");
print(x);

print("-- sử dụng simple imputer để tính giá trị trung bình thay cho chổ đang thiếu --");
imputer_mean = SimpleImputer(strategy='mean');
x_mean = imputer_mean.fit_transform(x);
print(x_mean);