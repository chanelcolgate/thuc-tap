import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
np.random.seed(0)
x=2-3*np.random.normal(0,1,20)
y=x-2*(x**2)+np.random.normal(-3,3,20)
x=x[:,np.newaxis]
kbin=KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='uniform',subsample=None)
x_binned=kbin.fit_transform(x)
model=LinearRegression()
rfe=RFE(estimator=model, n_features_to_select=3)
x_rfe=rfe.fit_transform(x_binned,y)
print(x_rfe.shape)