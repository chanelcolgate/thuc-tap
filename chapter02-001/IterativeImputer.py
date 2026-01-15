import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

data = {
    'Dien_Tich': [50, 60, 70, 80, 90],
    'Gia_Nha': [110, 130, np.nan, 170, 190] 
}
df = pd.DataFrame(data)

print("Dữ liệu ban đầu:")
print(df)   

imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
df_filled = imputer.fit_transform(df)   
df_result = pd.DataFrame(df_filled, columns=['Dien_Tich', 'Gia_Nha'])

print("\nDữ liệu sau khi xử lý thiếu bằng IterativeImputer:")
print(df_result)