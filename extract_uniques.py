import pandas as pd
import json

df = pd.read_csv("job_salary_prediction_dataset.csv")
categorical_cols = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']

unique_values = {}
for col in categorical_cols:
    unique_values[col] = sorted(df[col].unique().tolist())

with open('unique_values.json', 'w') as f:
    json.dump(unique_values, f)
