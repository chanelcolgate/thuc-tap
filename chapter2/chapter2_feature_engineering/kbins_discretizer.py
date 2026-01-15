import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

X_age = np.array([
    [5], [10],   
    [22], [35], [40], 
    [70], [80]    
])

print("--- 1. TUỔI GỐC ---")
print(X_age.flatten())

est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_binned = est.fit_transform(X_age)

df = pd.DataFrame({'Tuổi_Gốc': X_age.flatten(), 'Nhóm_Tuổi': X_binned.flatten()})
print("\n--- 2. SAU KHI CHIA GIỎ (KBINS) ---")
print(df) 