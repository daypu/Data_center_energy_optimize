import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import time
import joblib  # 导入joblib用于保存模型

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题

# Step 1: Load and prepare the data
file_path = 'PUE数据汇总.xlsx'  # 你的文件路径
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Convert '测量日期' to datetime format
data['测量日期'] = pd.to_datetime(data['测量日期'])

# Extract the target variable (逐时能耗)
target = data['CWHR_2_t_0_5h']

# Extract features (excluding '测量日期' and target column)
features = data.drop(columns=["冷源群控系统实时能耗_单位_kW", '测量日期', 'CWHR_1_t_0_5h','CWHR_2_t_0_5h', 'PUE值'])

# Check for missing values in both features and target
if target.isnull().any():
    print("Target variable contains NaN values. Filling with median.")
    target = target.fillna(target.median())  # 用中位数填补目标变量中的NaN值

if features.isnull().any().any():
    print("Features contain NaN values. Filling with median.")
    features = features.fillna(features.median())  # 用中位数填补特征变量中的NaN值

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 2: Perform K-fold Cross Validation (k=6)
kf = KFold(n_splits=6, shuffle=True, random_state=42)

# Initialize variables to store performance metrics
r2_scores = []
rmse_scores = []
mape_scores = []
times = []


# Define the objective function for Optuna optimization
def objective(trial, X_train, y_train, X_val, y_val):
    # Define the search space for hyperparameters
    num_leaves = trial.suggest_int('num_leaves', 30, 100)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 10)

    # Initialize the LightGBM model with the hyperparameters
    model = lgb.LGBMRegressor(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Predict and calculate RMSE
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return rmse


# Step 3: Perform cross-validation and track the results
for train_index, test_index in kf.split(features_scaled):
    X_train, X_test = features_scaled[train_index], features_scaled[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=10)

    # Get the best parameters from the optimization
    best_params = study.best_params

    # Train the LightGBM model with optimized parameters
    model = lgb.LGBMRegressor(
        num_leaves=best_params['num_leaves'],
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        random_state=42
    )
    # Start time measurement for prediction
    start_time = time.time()

    # Fit the model
    model.fit(X_train, y_train)

    # Predict energy consumption
    predicted = model.predict(X_test)

    # Measure time taken
    end_time = time.time()
    times.append(end_time - start_time)

    # Calculate R², RMSE, and MAPE
    r2 = r2_score(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    mape = np.mean(np.abs((y_test - predicted) / y_test)) * 100  # in percentage

    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mape_scores.append(mape)

# Step 4: Calculate the average performance metrics
avg_r2 = np.mean(r2_scores)
avg_rmse = np.mean(rmse_scores)
avg_mape = np.mean(mape_scores)
avg_time = np.mean(times)

# Step 6: Train the LightGBM model on the entire dataset for final prediction
final_best_params = study.best_params
final_model = lgb.LGBMRegressor(
    num_leaves=final_best_params['num_leaves'],
    learning_rate=final_best_params['learning_rate'],
    n_estimators=final_best_params['n_estimators'],
    max_depth=final_best_params['max_depth'],
    random_state=42
)
final_model.fit(features_scaled, target)

# Save the model to a file
model_filename = 'Lightgbm_model2.pkl'
joblib.dump(final_model, model_filename)
print(f"模型已保存为 {model_filename}")

# Predict with the trained model
predicted_final = final_model.predict(features_scaled)

# Step 7: Create a line plot comparing actual and predicted energy consumption
plt.figure(figsize=(10, 6))

# Line plot for actual temperature
plt.plot(data['测量日期'], target, label='实际温度', color='b', linestyle='-', marker='o')

# Line plot for predicted temperature
plt.plot(data['测量日期'], predicted_final, label='预测温度', color='r', linestyle='--', marker='x')

print("决定系数 (R²):", r2_score(target, predicted_final))
print("均方根误差 (RMSE):", np.sqrt(mean_squared_error(target, predicted_final)))
print("平均绝对百分比误差 (MAPE):", np.mean(np.abs((target - predicted_final) / target)) * 100)
print("每条数据的平均预测时间:", np.mean(times), "秒")
print("参数:", final_best_params)

# Add labels and title in Chinese
plt.xlabel('日期')
plt.ylabel('温度 (℃)')
plt.title('回水管道温度2实际与预测温度的比较')
plt.xticks(rotation=45)
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()
