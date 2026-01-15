import numpy as np
import pandas as pd
np.random.seed(40)
n_samples=300
bmi=np.random.uniform(20, 40, n_samples)
blood_pressure=2.5*bmi+np.random.normal(30,5,n_samples)
glucose=(1.5*bmi)+(0.8*blood_pressure)+np.random.normal(20,10,n_samples)
data={
    "BMI": bmi,
    "Blood_Pressure": blood_pressure,
    "Glucose": glucose
}
df=pd.DataFrame(data)
for columns in df.columns:
    mask=np.random.random(n_samples) < 0.1
    df.loc[df.sample(frac=0.25).index,columns] = np.nan
print(df)