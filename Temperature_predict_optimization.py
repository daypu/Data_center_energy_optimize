import optuna
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 读取数据
data = pd.read_excel("PUE数据汇总.xlsx")

# 数据预处理
data_clean = data.dropna()
X = data_clean.drop(columns=["冷源群控系统实时能耗_单位_kW","CWHR_1_t_0_5h", "测量日期", "PUE值","CWHR_2_t_0_5h"])
y = data_clean["CWHR_2_t_0_5h"]

# 确保列名是字符串类型并且不包含非法字符
X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X.columns]

# KFold交叉验证
kf = KFold(n_splits=6, shuffle=True, random_state=42)

# 定义模型
models = {
    "CART": DecisionTreeRegressor(),
    "RF": RandomForestRegressor(),
    "AdaBoost": AdaBoostRegressor(),
    "GBDT": GradientBoostingRegressor(),
    "XGBoost": xgb.XGBRegressor(),
    "LightGBM": lgb.LGBMRegressor(),
    "CatBoost": CatBoostRegressor(learning_rate=0.1, iterations=500, depth=5, silent=True)
}


# 目标函数定义
def objective_Cart(trial, X_train, y_train, X_val, y_val):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }

    model = DecisionTreeRegressor(**param)
    model.fit(X_train, y_train)

    # 预测并计算RMSE
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    return rmse


def objective_RF(trial, X_train, y_train, X_val, y_val):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }

    model = RandomForestRegressor(**param)
    model.fit(X_train, y_train)

    # 预测并计算RMSE
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    return rmse


def objective_AdaBoost(trial, X_train, y_train, X_val, y_val):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
    }

    model = AdaBoostRegressor(**param)
    model.fit(X_train, y_train)

    # 预测并计算RMSE
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    return rmse


def objective_GBDT(trial, X_train, y_train, X_val, y_val):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }

    model = GradientBoostingRegressor(**param)
    model.fit(X_train, y_train)

    # 预测并计算RMSE
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    return rmse


def objective_XGBoost(trial, X_train, y_train, X_val, y_val):
    param = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
    }

    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)

    # 预测并计算RMSE
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    return rmse


def objective_LightGBM(trial, X_train, y_train, X_val, y_val):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }

    model = lgb.LGBMRegressor(**param)
    model.fit(X_train, y_train)

    # 预测并计算RMSE
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    return rmse


def objective_CatBoost(trial, X_train, y_train, X_val, y_val):
    param = {
        'iterations': trial.suggest_int('iterations', 50, 200),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-5, 1e2),
    }

    model = CatBoostRegressor(**param, silent=True)
    model.fit(X_train, y_train)

    # 预测并计算RMSE
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    return rmse


# 进行贝叶斯优化并获取最佳参数
def optimize_model(objective, X_train, y_train, X_val, y_val):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=10)
    return study


# 存储每个模型的评价指标
results = {model_name: {"R²": [], "RMSE": [], "MAPE": [], "Time": [], "Prediction Time": []} for model_name in models}

# 在交叉验证之前进行贝叶斯优化，得到每个模型的最佳参数
studies = {}
for model_name in models:
    if model_name == "CART":
        studies[model_name] = optimize_model(objective_Cart, X, y, X, y)
    elif model_name == "RF":
        studies[model_name] = optimize_model(objective_RF, X, y, X, y)
    elif model_name == "AdaBoost":
        studies[model_name] = optimize_model(objective_AdaBoost, X, y, X, y)
    elif model_name == "GBDT":
        studies[model_name] = optimize_model(objective_GBDT, X, y, X, y)
    elif model_name == "XGBoost":
        studies[model_name] = optimize_model(objective_XGBoost, X, y, X, y)
    elif model_name == "LightGBM":
        studies[model_name] = optimize_model(objective_LightGBM, X, y, X, y)
    elif model_name == "CatBoost":
        studies[model_name] = optimize_model(objective_CatBoost, X, y, X, y)

# K折交叉验证
kf = KFold(n_splits=6, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 对每个模型进行训练并评估
    for model_name, model in models.items():
        # 从贝叶斯优化结果中获取最佳参数
        best_params = studies[model_name].best_params
        # 创建新的模型实例并应用最佳参数
        if model_name == "XGBoost":
            model = xgb.XGBRegressor(**best_params)  # 创建新的XGBoost模型
        elif model_name == "CatBoost":
            model = CatBoostRegressor(**best_params, silent=True)  # 创建新的CatBoost模型
        else:
            model.set_params(**best_params)  # 对其他模型设置最佳参数

        model.fit(X_train, y_train)

        # 预测并评估
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # 预测并评估
        predict_start_time = time.time()  # 记录预测开始时间
        y_pred = model.predict(X_test)
        predict_end_time = time.time()  # 记录预测结束时间
        prediction_time = predict_end_time - predict_start_time  # 计算预测时间
        avg_prediction_time = prediction_time / len(X_test)  # 计算平均预测时间

        results[model_name]["R²"].append(r2)
        results[model_name]["RMSE"].append(rmse)
        results[model_name]["MAPE"].append(mape)
        results[model_name]["Prediction Time"].append(prediction_time)

# 计算每个模型的平均性能指标
average_results = {model_name: {
    "R²": np.mean(results[model_name]["R²"]),
    "RMSE": np.mean(results[model_name]["RMSE"]),
    "MAPE": np.mean(results[model_name]["MAPE"]),
    "Prediction Time": np.mean(results[model_name]["Prediction Time"])
} for model_name in models}

# 打印最终结果
average_results_df = pd.DataFrame(average_results).T

# 获取每个模型的参数
model_params = {model_name: model.get_params() for model_name, model in models.items()}

# 打印结果和模型参数
print(average_results_df)
for model_name, params in model_params.items():
    print(f"\n{model_name} Parameters:")
    print(params)
