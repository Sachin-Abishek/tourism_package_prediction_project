# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# for model serialization
import joblib
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/SachinAbi/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/SachinAbi/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/SachinAbi/tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/SachinAbi/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define features
numeric_features = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome']
categorical_features = ['TypeofContact', 'CityTier', 'Occupation', 'Gender', 'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 'OwnCar', 'Designation']

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42, objective="reg:squarederror")

# Define hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [50, 100],
    'xgbregressor__max_depth': [2, 3],
    'xgbregressor__learning_rate': [0.01, 0.05],
    'xgbregressor__colsample_bytree': [0.6, 0.8],
    'xgbregressor__subsample': [0.6, 0.8],
    'xgbregressor__reg_lambda': [0.5, 1],
}

# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Grid search with cross-validation
grid_search = GridSearchCV(
    model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predictions
y_pred_train = best_model.predict(Xtrain)
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Performance:")
print("MAE:", mean_absolute_error(ytrain, y_pred_train))
print("RMSE:", np.sqrt(mean_squared_error(ytrain, y_pred_train)))
print("R²:", r2_score(ytrain, y_pred_train))

print("\nTest Performance:")
print("MAE:", mean_absolute_error(ytest, y_pred_test))
print("RMSE:", np.sqrt(mean_squared_error(ytest, y_pred_test)))
print("R²:", r2_score(ytest, y_pred_test))

# Save best model
joblib.dump(best_model, "tourism_package_prediction_model_v1.joblib")


# Upload to Hugging Face
repo_id = "SachinAbi/tourism_package_prediction_model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="tourism_package_prediction_model_v1.joblib",
    path_in_repo="tourism_package_prediction_model_v1.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
