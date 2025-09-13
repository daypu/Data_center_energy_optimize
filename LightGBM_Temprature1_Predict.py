import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import time
import joblib
import re


# 统一使用的列名清洗函数
def enhanced_sanitize(name):
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_]', '', name)
    return re.sub(r'_+', '_', cleaned).strip('_')


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Step 1: 加载数据并清洗列名
file_path = 'PUE数据汇总.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')
data.columns = [enhanced_sanitize(col) for col in data.columns]
data['测量日期'] = pd.to_datetime(data['测量日期'])

# Step 2: 准备特征和标签
target = data['CWHR_1_t_0_5h']
features = data.drop(columns=["冷源群控系统实时能耗_单位_kW", '测量日期', 'CWHR_1_t_0_5h', 'CWHR_2_t_0_5h', 'PUE值'])

# 处理缺失值
if target.isnull().any():
    target = target.fillna(target.median())
features = features.fillna(features.median())

# 保存原始特征名称
feature_names = features.columns.tolist()

# 标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 3: 交叉验证和模型训练
kf = KFold(n_splits=6, shuffle=True, random_state=42)
r2_scores = []
rmse_scores = []
mape_scores = []
times = []


def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 30, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'verbosity': -1  # 将verbose参数移动到模型初始化
    }

    model = lgb.LGBMRegressor(**params, random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # 移除fit方法中的verbose参数
        callbacks=[lgb.callback.log_evaluation(period=0)]  # 禁用训练日志
    )
    y_pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, y_pred))


best_model = None
best_score = np.inf

for train_index, test_index in kf.split(features_scaled):
    X_train, X_test = features_scaled[train_index], features_scaled[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=10)

    # 使用特征名称训练最终模型
    model = lgb.LGBMRegressor(**study.best_params, random_state=42, verbosity=-1)
    model.fit(
        X_train, y_train,
        feature_name=feature_names,
        categorical_feature=[],
        callbacks=[lgb.callback.log_evaluation(period=0)]
    )

    # 验证特征名称
    assert model.booster_.feature_name() == feature_names, "特征名称不匹配!"

    if study.best_value < best_score:
        best_score = study.best_value
        best_model = model

# Step 4: 保存完整模型信息
model_package = {
    'model': best_model,
    'feature_names': feature_names,
    'scaler': scaler
}

joblib.dump(model_package, 'Lightgbm_model1.pkl')

# 可视化验证
predicted_final = best_model.predict(features_scaled)
plt.figure(figsize=(10, 6))
plt.plot(data['测量日期'], target, label='实际温度', color='b')
plt.plot(data['测量日期'], predicted_final, label='预测温度', color='r', linestyle='--')
plt.xlabel('日期')
plt.ylabel('温度 (℃)')
plt.title('回水管道温度1实际与预测温度比较')
plt.legend()
plt.show()