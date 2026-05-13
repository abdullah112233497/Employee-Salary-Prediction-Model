# ================================
# Employee Salary Prediction
# Using Linear Regression
# ================================

# STEP 1 — Import Libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error


# STEP 2 — Load Dataset

df = pd.read_csv("job_salary_prediction_dataset.csv")


# STEP 3 — Select Features (Inputs)

X = df[[
    'job_title',
    'education_level',
    'experience_years',
    'skills_count',
    'industry',
    'company_size',
    'location',
    'remote_work',
    'certifications'
]]


# STEP 4 — Select Target (Output)

y = df['salary']


# STEP 5 — Define Column Types

# Text columns
categorical_cols = [
    'job_title',
    'education_level',
    'industry',
    'company_size',
    'location',
    'remote_work'
]
# Numerical columns
numerical_cols = [
    'experience_years',
    'skills_count',
    'certifications'
]


# STEP 6 — Preprocessing

preprocessor = ColumnTransformer([

    # One Hot Encoding for text columns
    ('cat', OneHotEncoder(drop='first'), categorical_cols),

    # Scaling for numerical columns
    ('num', StandardScaler(), numerical_cols)

])


# STEP 7 — Create Pipeline

model = Pipeline([

    ('preprocessing', preprocessor),

    ('regressor', LinearRegression())

])


# STEP 8 — Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 9 — Train Model

model.fit(X_train, y_train)


# STEP 10 — Make Predictions

y_pred = model.predict(X_test)


# STEP 11 — Check Accuracy

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", r2)

print("Mean Absolute Error:", mae)

# STEP 12 — Save Model
import joblib
joblib.dump(model, 'salary_model.pkl')
print("\nModel saved as salary_model.pkl")
