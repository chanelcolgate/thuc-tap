import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
data={
    'color':['red', 'blue', 'green', 'blue', 'red'],
    'size':['S', 'M', 'L', 'XL', 'M'],
}
df=pd.DataFrame(data)
column=ColumnTransformer(
    transformers=[
        ('color_onehot', OneHotEncoder(sparse_output=False), ['color']),
    ],
    remainder='passthrough'
)
encoded_result=column.fit_transform(df)
new_columns=column.get_feature_names_out()
df_final=pd.DataFrame(encoded_result, columns=new_columns)
print(df_final)
