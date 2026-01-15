import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def get_dataset():
    X = np.array([
        [1, 2, np.nan], 
        [3, 4, 3], 
        [np.nan, 6, 5], 
        [8, 8, 7], 
        [10, np.nan, 9],
        [2, 3, 4],       
        [9, 9, np.nan],
        [5, 5, 5]
    ])
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    return X, y

def score_dataset(X, y, imputer, name):
    model = make_pipeline(imputer, LogisticRegression())
    
    scores = cross_val_score(model, X, y, cv=2, scoring='accuracy')
    
    print(f" Phương pháp {name}: Độ chính xác = {scores.mean():.2f} (hoặc {scores.mean()*100}%)")

if __name__ == "__main__":
    X, y = get_dataset()

    score_dataset(X, y, SimpleImputer(strategy='mean'), "Mean (Trung bình)")

    score_dataset(X, y, KNNImputer(n_neighbors=2), "KNN (Hàng xóm)")

    score_dataset(X, y, IterativeImputer(random_state=0), "Iterative (Suy luận)")
    
    print("\n Đã hoàn thành toàn bộ yêu cầu Ticket!")