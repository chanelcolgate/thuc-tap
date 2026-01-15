import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

data={
    'feature1':np.random.random(100)*100,
    'feature2':np.random.random(100)*50,    
    'target':np.random.choice([0,1],size=100)
}
df=pd.DataFrame(data)
x=df.drop('target', axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
scaler=Normalizer()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
print(x_train_scaled)