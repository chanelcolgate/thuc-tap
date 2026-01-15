import pandas as pd
from advancedPipeline import full_pipeline, X_test, X_train 

print("\n--- ACTION: CREATE DATAFRAMES WITH TRANSFORMED DATA ---")

preprocessor_step = full_pipeline.named_steps['tien_xu_ly']

X_test_processed = preprocessor_step.transform(X_test)

cat_feature_names = preprocessor_step.named_transformers_['nhanh_chu']\
    .named_steps['encoder'].get_feature_names_out(['ThànhPhố'])

all_feature_names = ['Tuổi'] + list(cat_feature_names)

df_transformed = pd.DataFrame(X_test_processed, columns=all_feature_names)

print("Dữ liệu sau khi đi qua DataFrame:")
print(df_transformed)
print("\n=> Đây là dữ liệu sạch sẽ trước khi vào LogisticRegression.")