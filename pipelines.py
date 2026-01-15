import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)
n_samples=200
bmi=np.random.uniform(20, 40, n_samples)
blood_pressure=2.5*bmi+np.random.normal(30,5,n_samples)
glucose=(1.5*bmi)+(0.8*blood_pressure)+np.random.normal(20,10,n_samples)
data={
    "BMI": bmi,
    "Blood_Pressure": blood_pressure,
    "Glucose": glucose
}
df=pd.DataFrame(data)
for col in df.columns:
    df.loc[df.sample(frac=0.25).index, col] = np.nan
medical_pipeline=Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
processed_data=medical_pipeline.fit_transform(df)
processed_df=pd.DataFrame(processed_data, columns=df.columns)
print(processed_df)