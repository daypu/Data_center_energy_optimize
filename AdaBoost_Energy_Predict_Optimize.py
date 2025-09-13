import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import time
import joblib

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 数据预处理 ====================
print("\n========== 数据预处理 ==========")
file_path = '总数据副本.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 日期转换与列处理
data['测量日期'] = pd.to_datetime(data['测量日期'])
target = data['冷源群控系统实时能耗_单位_kW']
features = data.drop(columns=['测量日期', '冷源群控系统实时能耗_单位_kW', 'PUE值', 'CWHR_1_t_0_5h', 'CWHR_2_t_0_5h'])

# 缺失值处理
print("正在检测缺失值...")
nan_info = {
    '目标变量缺失数': target.isna().sum(),
    '特征缺失数': features.isna().sum().sum()
}
print(f"目标变量缺失值数量: {nan_info['目标变量缺失数']}")
print(f"特征矩阵缺失值总数: {nan_info['特征缺失数']}")

target.fillna(target.median(), inplace=True)
features.fillna(features.median(), inplace=True)

# 特征标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ==================== 方差过大的点的调整 ====================
# 计算目标变量的滚动方差
window_size = 10  # 选择合适的窗口大小
rolling_variance = target.rolling(window=window_size).var()

# 找出方差大于18的点
extreme_points = rolling_variance[rolling_variance > 20]

# 逐个调整方差过大的点，使其不超过18
for idx in extreme_points.index:
    # 找到该点前后的数据趋势
    before_idx = max(0, idx - window_size)  # 保证不越界
    after_idx = min(len(target) - 1, idx + window_size)  # 保证不越界

    # 获取这段时间的子集数据
    data_segment = target[before_idx:after_idx]

    # 计算该段数据的趋势
    trend = np.polyfit(range(len(data_segment)), data_segment.values, 1)  # 拟合一次线性趋势

    # 预测该点的合理值
    predicted_value = np.polyval(trend, window_size)

    # 更新极端点的值
    target.loc[idx] = predicted_value

# ==================== 模型训练 ====================
print("\n========== 模型训练 ==========")


# 优化目标函数
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5)
    }
    model = AdaBoostRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))


# 超参数优化
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, features_scaled, target, features_scaled, target), n_trials=10)

# 记录最佳参数
current_params = study.best_params
print(f"最佳参数: {current_params}")

# ==================== 全量训练 ====================
print("\n========== 全量模型训练 ==========")
final_model = AdaBoostRegressor(**current_params, random_state=42)
final_model.fit(features_scaled, target)

# 模型保存
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'features': features.columns.tolist()
}, 'adaboost_energy_model.pkl')

# ==================== 最终预测 ====================
predicted_final = final_model.predict(features_scaled)

print(f"R²分数: {r2_score(target, predicted_final):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(target, predicted_final)):.4f}")
print(f"MAPE: {np.mean(np.abs((target - predicted_final) / target)) * 100:.2f}%")

# ==================== 可视化 ====================
plt.figure(figsize=(14, 7))

# 实际值曲线
plt.plot(
    data['测量日期'],
    target,
    label='实际能耗',
    color='#0000ff',
    linestyle='-',
    marker='o',
    markersize=5,
    linewidth=1.5,
    alpha=0.8
)

# 预测值曲线
plt.plot(
    data['测量日期'],
    predicted_final,
    label='预测能耗',
    color='#ff0000',
    linestyle='--',
    marker='s',
    markersize=5,
    linewidth=1.5,
    alpha=0.8
)

# 图表装饰
plt.title('经Optuna参数优化的冷源系统实时能耗实际值与预测值对比\n(AdaBoost Regression)', fontsize=14, pad=20)
plt.xlabel('测量日期', fontsize=12)
plt.ylabel('实时能耗 (kW)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='upper left', frameon=False)
plt.grid(True, linestyle=':', alpha=0.6)

# 紧凑布局
plt.tight_layout()
plt.savefig('energy_comparison_optuna.png', dpi=300)
print("\n对比图已保存为 energy_comparison_optuna.png")
plt.show()
