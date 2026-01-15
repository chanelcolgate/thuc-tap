import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
np.random.seed(0)
x=2-3*np.random.normal(0,1,20)
y=x-2*(x**2)+np.random.normal(-3,3,20)
x=x[:,np.newaxis]
poly=PolynomialFeatures(degree=2,include_bias=False)
x_poly=poly.fit_transform(x)
model=LinearRegression()
model.fit(x_poly,y)
print(poly.get_feature_names_out(['Phanbon']))