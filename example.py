import numpy as np
import pandas as pd

np.random.seed(2024)
n_samples=20
n_features=10
data={
    f"Feature{i+1}": np.random.uniform(0, 100, n_samples)
    for i in range(n_features)
}
df=pd.DataFrame(data)
for column in df.columns:
    mask=np.random.random(n_samples) < 0.2
    df.loc[mask, column] = np.nan
print(df)