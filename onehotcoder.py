import pandas as pd
from sklearn.preprocessing import OneHotEncoder 
data={
    'color':['red', 'blue', 'green', 'blue', 'red'],
}
df=pd.DataFrame(data)
onehot=OneHotEncoder(sparse_output=False)
onehot_encoded=onehot.fit_transform(df[['color']])
onehot_df=pd.DataFrame(onehot_encoded, columns=onehot.get_feature_names_out(['color']))

print(onehot_df)