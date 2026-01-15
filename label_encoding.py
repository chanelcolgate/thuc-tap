import pandas as pd
from sklearn.preprocessing import LabelEncoder
data={
    'color':['red', 'blue', 'green', 'blue', 'red'],
}
df=pd.DataFrame(data)
label=LabelEncoder()
df['color_encoded']=label.fit_transform(df['color'])
print(df)