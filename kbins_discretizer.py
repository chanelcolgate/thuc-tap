import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
np.random.seed(0)
x=2-3*np.random.normal(0,1,20)
y=x-2*(x**2)+np.random.normal(-3,3,20)
x=x[:,np.newaxis]
kbin=KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform',subsample=None)
x_binned=kbin.fit_transform(x)
model=LinearRegression()
model.fit(x_binned,y)
print(kbin.bin_edges_)