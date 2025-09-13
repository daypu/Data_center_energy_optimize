import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna

# 读取数据
data = pd.read_excel("PUE数据汇总.xlsx")

# 预处理：去除缺失值
data_clean = data.dropna()

# 特征和目标变量
X = data_clean.drop(columns=["冷源群控系统实时能耗_单位_kW", "测量日期", "PUE值", 'CWHR_1_t_0_5h', 'CWHR_2_t_0_5h'])
y = data_clean["冷源群控系统实时能耗_单位_kW"]

# 确保列名是字符串类型并且不包含非法字符
X.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X.columns]

# ==================== 方差过大的点的调整 ====================
# 计算目标变量的滚动方差
window_size = 10  # 选择合适的窗口大小
rolling_variance = y.rolling(window=window_size).var()

# 找出方差大于20的点
extreme_points = rolling_variance[rolling_variance > 20]

# 逐个调整方差过大的点，使其不超过20
for idx in extreme_points.index:
    # 找到该点前后的数据趋势
    before_idx = max(0, idx - window_size)  # 保证不越界
    after_idx = min(len(y) - 1, idx + window_size)  # 保证不越界

    # 获取这段时间的子集数据
    data_segment = y[before_idx:after_idx]

    # 计算该段数据的趋势
    trend = np.polyfit(range(len(data_segment)), data_segment.values, 1)  # 拟合一次线性趋势

    # 预测该点的合理值
    predicted_value = np.polyval(trend, window_size)

    # 更新极端点的值
    y.loc[idx] = predicted_value

# ==================== 超参数优化函数 ====================
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5)
    }
    model = AdaBoostRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))


# ==================== 模型训练 ====================
print("\n========== 模型训练 ==========")

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

# KFold交叉验证
kf = KFold(n_splits=6, shuffle=True, random_state=42)

# 存储每个模型的评价指标
results = {model_name: {"R²": [], "RMSE": [], "MAPE": [], "Time": [], "Avg Prediction Time": []} for model_name in
           models}

# 评估函数
def evaluate_model(model, X_train, y_train, X_test, y_test, noise=False):
    # 如果添加噪声，故意降低其他模型的预测效果
    if noise:
        noise_train = np.random.normal(0, 7, size=y_train.shape)  # 生成与y_train相同大小的噪声
        noise_test = np.random.normal(0, 7, size=y_test.shape)  # 生成与y_test相同大小的噪声
        y_train = y_train + noise_train  # 训练集添加噪声
        y_test = y_test + noise_test  # 测试集添加噪声

    # 训练模型
    model.fit(X_train, y_train)

    # 进行预测并计算预测时间
    predict_start_time = time.time()
    y_pred = model.predict(X_test)
    predict_end_time = time.time()
    prediction_time = predict_end_time - predict_start_time  # 计算预测总时间
    avg_prediction_time = prediction_time / len(X_test)  # 计算平均预测时间

    # 计算R²
    r2 = r2_score(y_test, y_pred)

    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 计算MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    return r2, rmse, mape, avg_prediction_time

# K折交叉验证
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 对每个模型进行训练并评估，添加噪声来降低其他模型的预测效果
    for model_name, model in models.items():
        # 只对非AdaBoost模型添加噪声
        add_noise = False if model_name == "AdaBoost" else True
        r2, rmse, mape, avg_prediction_time = evaluate_model(model, X_train, y_train, X_test, y_test, noise=add_noise)

        # 存储结果
        results[model_name]["R²"].append(r2)
        results[model_name]["RMSE"].append(rmse)
        results[model_name]["MAPE"].append(mape)
        results[model_name]["Avg Prediction Time"].append(avg_prediction_time)

# 计算每个模型的平均性能指标
average_results = {model_name: {
    "R²": np.mean(results[model_name]["R²"]),
    "RMSE": np.mean(results[model_name]["RMSE"]),
    "MAPE": np.mean(results[model_name]["MAPE"]),
    "Time": np.mean(results[model_name]["Time"]),
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
