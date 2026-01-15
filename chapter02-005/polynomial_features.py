import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[2], [3], [4]]) 

poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)

print("Dữ liệu sau khi nâng bậc đa thức:")
print(X_poly)