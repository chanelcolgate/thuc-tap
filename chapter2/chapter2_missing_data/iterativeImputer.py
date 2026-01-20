import numpy as np 
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

x = np.array([
    [8, 9, 7],
    [7, np.nan, 8],
    [np.nan, 6, 9],
    [5, 5, np.nan],
    [10, 10, 10]     
])

print("--data gốc--");
print(x);

imputer_iter = IterativeImputer(random_state=0);
print(" sử dụng iterative imputer để tính giá trị thay chổ cần thiếu");
x_iter = imputer_iter.fit_transform(x);
print(x_iter);