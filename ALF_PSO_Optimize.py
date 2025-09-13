# -*- coding: utf-8 -*-
import sys
import traceback
import pandas as pd
import numpy as np
import re
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline # Not used directly currently
import lightgbm as lgb
import joblib
import json
from collections import defaultdict
import time # 导入 time 模块用于计时（可选）

# 配置参数
DATA_PATH = "PUE数据汇总_processed.xlsx"
OUTPUT_FILE = "ALF_PSO_调优后的参数_v2.xlsx" # 修改输出文件名以区分
MODEL_PREFIX = "optimized_model_v2"
META_FILE = "optimized_meta_data_v2.json"

# 变量范围定义
VARIABLE_BOUNDS_ORIGINAL = {
    '冷机冷却侧出水压力PIT_1_CWS_1_压力表_实际值': (2.24, 2.33),
    '冷机冷却侧出水压力PIT_2_CWS_1_压力表_实际值': (0.74, 2.29),
    '冷机冷冻侧出水压力PIT_1_CWHS_1_压力表_实际值': (1.72, 3.67),
    '冷机冷冻侧出水压力PIT_2_CWHS_1_压力表_实际值': (1.69, 3.67),
    '冷机冷冻侧出水温度TIT_1_CWHS_1_温度计_实际值': (12.13, 15.22),
    '冷机冷冻侧出水温度TIT_2_CWHS_1_温度计_实际值': (12.81, 15.30),
    '冷机冷却侧出水温度TIT_1_CWS_1_温度计_实际值': (13.48, 16.88),
    '冷机冷却侧出水温度TIT_2_CWS_1_温度计_实际值': (14.52, 17.39),
    'CV1_1_比例阀门_开度反馈显示': (0.04, 99.81),
    'CV1_3_比例阀门_开度反馈显示': (0.13, 0.17),
    'CV10_1_比例阀门_开度反馈显示': (0.17, 99.48),
    'CV10_2_比例阀门_开度反馈显示': (0.08, 0.10),
    'CV11_1_比例阀门_开度反馈显示': (100.21, 100.35),
    'CV11_2_比例阀门_开度反馈显示': (100.24, 100.36),
    'CV14_1_比例阀门_开度反馈显示': (0.31, 0.35),
    'CV14_2_比例阀门_开度反馈显示': (0.03, 0.07),
    'CV15_1_比例阀门_开度反馈显示': (0.13, 0.16),
    'CV15_2_比例阀门_开度反馈显示': (0.04, 0.08),
    'CV2_1_比例阀门_开度反馈显示': (0.02, 0.04),
    'CV2_3_比例阀门_开度反馈显示': (0.0, 0.02),
    'CV9_1_比例阀门_开度反馈显示': (0.0, 0.03),
    'CV9_2_比例阀门_开度反馈显示': (0.0, 0.02),
    # 明确指定开关阀门变量及其 0/1 边界
    'V1_6_开关阀门_开到位反馈': (0, 1),
    'V1_7_开关阀门_开到位反馈': (0, 1),
    'V16_1_开关阀门_开到位反馈': (0, 1),
    'V16_2_开关阀门_开到位反馈': (0, 1)
}

# --- 函数和类定义 ---

def enhanced_sanitize(name):
    """清洗列名特殊字符"""
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_]+', '_', str(name))
    cleaned = re.sub(r'^_+|_+$', '', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned

VARIABLE_BOUNDS = {enhanced_sanitize(k): v for k, v in VARIABLE_BOUNDS_ORIGINAL.items()}

# 识别需要处理为整数 (0/1) 的变量列名 (清洗后的)
INTEGER_COLS_SANITIZED = [
    enhanced_sanitize('V1_6_开关阀门_开到位反馈'),
    enhanced_sanitize('V1_7_开关阀门_开到位反馈'),
    enhanced_sanitize('V16_1_开关阀门_开到位反馈'),
    enhanced_sanitize('V16_2_开关阀门_开到位反馈')
]


def load_data():
    """加载并预处理数据"""
    try:
        print(f"开始加载数据: {DATA_PATH}...")
        start_time = time.time()
        df = pd.read_excel(DATA_PATH)
        load_time = time.time() - start_time
        print(f"Excel 文件加载完成，耗时: {load_time:.2f} 秒。原始形状: {df.shape}")

        original_columns = df.columns.tolist()
        df.columns = [enhanced_sanitize(col) for col in df.columns]
        sanitized_columns = df.columns.tolist()
        print("列名已清洗。")

        target_temp1_col = enhanced_sanitize('CWHR_1_t_0.5h')
        target_temp2_col = enhanced_sanitize('CWHR_2_t_0.5h')
        target_energy_col = enhanced_sanitize('冷源群控系统实时能耗_单位_kW')
        pue_col_sanitized = enhanced_sanitize('PUE值') # 清洗 PUE 列名
        date_col = enhanced_sanitize('测量日期')
        print(f"目标列: Temp1='{target_temp1_col}', Temp2='{target_temp2_col}', Energy='{target_energy_col}', Date='{date_col}'")

        variable_cols = list(VARIABLE_BOUNDS.keys())
        # 检查所有整数列是否都在可调参数列中
        missing_int_cols = [col for col in INTEGER_COLS_SANITIZED if col not in variable_cols]
        if missing_int_cols:
            raise ValueError(f"配置错误：指定的整数列 {missing_int_cols} 不在可调参数列表 (variable_cols) 中。")

        required_cols = variable_cols + [target_temp1_col, target_temp2_col, target_energy_col, date_col]
        print(f"共 {len(variable_cols)} 个可调参数列，其中 {len(INTEGER_COLS_SANITIZED)} 个将处理为整数。")

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             raise ValueError(f"数据文件中缺少以下必须的列 (已尝试清洗名称): {missing_cols}。\n可用列: {sanitized_columns}")

        # 定义温度模型特征: 所有非排除列 (*** 已排除 PUE ***)
        exclude_cols = [date_col, target_energy_col, target_temp1_col, target_temp2_col, pue_col_sanitized]
        temp_features = [col for col in df.columns if col not in exclude_cols]
        temp1_features = temp_features.copy() # 两个温度模型使用相同的特征集
        temp2_features = temp_features.copy()
        print(f"温度模型将使用 {len(temp_features)} 个特征 (已排除 PUE值)。")

        # 检查可调参数是否都在温度特征里 (可能有些不在，这是正常的)
        missing_vars_in_temp = [col for col in variable_cols if col not in temp_features]
        if missing_vars_in_temp:
             print(f"提示: 以下 {len(missing_vars_in_temp)} 个可调参数列不在温度模型特征集中 (这可能是预期行为): {missing_vars_in_temp}")

        print("开始数据类型转换和NaN处理...")
        start_preprocess_time = time.time()
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        rows_before_drop = len(df_numeric)
        cols_to_check_na = list(set(required_cols + temp_features))
        df_filtered = df_numeric.dropna(subset=cols_to_check_na)
        rows_after_drop = len(df_filtered)
        preprocess_time = time.time() - start_preprocess_time
        print(f"数据类型转换和NaN处理完成，耗时: {preprocess_time:.2f} 秒。")
        print(f"原始行数: {rows_before_drop}, 基于关键列去除NaN后剩余行数: {rows_after_drop}")

        if df_filtered.empty:
            raise ValueError("处理和删除NaN后，数据为空，无法继续。请检查原始数据质量和 'dropna' 的列范围。")
        print(f"最终用于模型训练和优化的数据集大小: {df_filtered.shape}")

        for col in INTEGER_COLS_SANITIZED:
             if col in df_filtered.columns:
                  unique_vals = df_filtered[col].unique()
                  if not all(v == 0 or v == 1 for v in unique_vals if not pd.isna(v)):
                       print(f"警告: 整数列 '{col}' 在加载和清理后包含非 0/1 的值: {unique_vals}。请检查数据源。")

        if date_col in df_filtered.columns:
            try:
                df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])
                print(f"'{date_col}' 列已转换为日期时间类型。")
            except Exception as e:
                print(f"警告: 转换 '{date_col}' 列为日期时间类型失败: {e}。将保持原始格式。")

        return (
            df_filtered, variable_cols, temp1_features, temp2_features,
            target_temp1_col, target_temp2_col, target_energy_col, date_col
        )
    except FileNotFoundError:
        print(f"错误: 数据文件未找到于路径 {DATA_PATH}")
        sys.exit(1)
    except ValueError as ve:
        print(f"数据加载/处理错误: {str(ve)}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"数据加载时发生未知错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


class HyperparameterOptimizer:
    """Optuna超参数优化器"""
    @staticmethod
    def optimize_lgbm(trial, X, y):
        """LGBM的Optuna目标函数"""
        if X.size == 0 or y.size == 0:
            trial.report(float('inf'), step=0)
            return float('inf')
        params = {
            'objective': 'regression_l1', 'metric': 'mae',
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 700),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 0.5, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 0.5, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42, 'n_jobs': -1, 'verbosity': -1
        }
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(X, y)
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            return mse
        except Exception as e:
            # print(f"Optuna LGBM 试验失败: {e}") # 保持安静
            return float('inf')

    @staticmethod
    def optimize_adaboost(trial, X, y):
        """AdaBoost的Optuna目标函数 (带 base_estimator 调优)"""
        if X.size == 0 or y.size == 0:
            trial.report(float('inf'), step=0)
            return float('inf')
        base_estimator_max_depth = trial.suggest_int('base_max_depth', 2, 10)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
        }
        try:
            base_estimator = DecisionTreeRegressor(max_depth=base_estimator_max_depth, random_state=42)
            model = AdaBoostRegressor(estimator=base_estimator, **params, random_state=42)
            model.fit(X, y)
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            return mse
        except Exception as e:
            # print(f"Optuna AdaBoost 试验失败: {e}") # 保持安静
            return float('inf')


def train_models(X_temp1, X_temp2, y_temp1, y_temp2, X_energy, y_energy):
    """训练温度模型和能耗模型，并进行超参数优化"""
    models = {}
    scalers = {}
    feature_lists = {}
    try:
        # --- 温度模型 1 (LGBM) ---
        print("开始温度模型1 (LGBM) 的超参数优化...")
        start_opt1 = time.time()
        study1 = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        # 减少 Optuna 自身的输出
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study1.optimize(lambda trial: HyperparameterOptimizer.optimize_lgbm(trial, X_temp1, y_temp1), n_trials=50, n_jobs=-1, show_progress_bar=False)
        optuna.logging.set_verbosity(optuna.logging.INFO) # 恢复默认日志级别
        opt1_time = time.time() - start_opt1
        if not hasattr(study1, 'best_params') or not study1.best_params:
            raise RuntimeError("温度模型1的Optuna优化失败，未能找到最佳参数。")
        print(f"温度模型1 优化完成，耗时: {opt1_time:.2f} 秒。最佳参数: {study1.best_params} (MSE: {study1.best_value:.4f})")
        model_temp1 = lgb.LGBMRegressor(**study1.best_params, objective='regression_l1', metric='mae', random_state=42, n_jobs=-1, verbosity=-1)
        model_temp1.fit(X_temp1, y_temp1)
        models['temp1'] = model_temp1
        feature_lists['temp1'] = X_temp1.columns.tolist()
        print("温度模型1 最终训练完成。")

        # --- 温度模型 2 (LGBM) ---
        print("\n开始温度模型2 (LGBM) 的超参数优化...")
        start_opt2 = time.time()
        study2 = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study2.optimize(lambda trial: HyperparameterOptimizer.optimize_lgbm(trial, X_temp2, y_temp2), n_trials=50, n_jobs=-1, show_progress_bar=False)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        opt2_time = time.time() - start_opt2
        if not hasattr(study2, 'best_params') or not study2.best_params:
             raise RuntimeError("温度模型2的Optuna优化失败，未能找到最佳参数。")
        print(f"温度模型2 优化完成，耗时: {opt2_time:.2f} 秒。最佳参数: {study2.best_params} (MSE: {study2.best_value:.4f})")
        model_temp2 = lgb.LGBMRegressor(**study2.best_params, objective='regression_l1', metric='mae', random_state=42, n_jobs=-1, verbosity=-1)
        model_temp2.fit(X_temp2, y_temp2)
        models['temp2'] = model_temp2
        feature_lists['temp2'] = X_temp2.columns.tolist()
        print("温度模型2 最终训练完成。")

        # --- 能耗模型 (AdaBoost) ---
        energy_features = X_energy.columns.tolist()
        feature_lists['energy'] = energy_features
        print(f"\n能耗模型 (AdaBoost) 将使用以下 {len(energy_features)} 个特征: {energy_features}")

        print("\n对能耗模型特征进行标准化...")
        energy_scaler = StandardScaler()
        X_energy_scaled = energy_scaler.fit_transform(X_energy)
        scalers['energy'] = energy_scaler
        print("特征标准化完成。Scaler 已保存。")

        # print("\n能耗目标变量 (y_energy) 统计信息:") # 静默
        # print(y_energy.describe())
        if y_energy.isnull().any():
             print(f"警告: 能耗目标变量 y_energy 中包含 {y_energy.isnull().sum()} 个 NaN 值！")
        # print("-" * 30)

        print("\n开始能耗模型 (AdaBoost) 的超参数优化 (使用标准化特征)...")
        start_opt3 = time.time()
        study3 = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study3.optimize(lambda trial: HyperparameterOptimizer.optimize_adaboost(trial, X_energy_scaled, y_energy), n_trials=50, n_jobs=-1, show_progress_bar=False)
        optuna.logging.set_verbosity(optuna.logging.INFO)
        opt3_time = time.time() - start_opt3
        if not hasattr(study3, 'best_params') or not study3.best_params:
             raise RuntimeError("能耗模型的Optuna优化失败，未能找到最佳参数。")
        print(f"能耗模型 优化完成，耗时: {opt3_time:.2f} 秒。最佳参数: {study3.best_params} (MSE: {study3.best_value:.4f})")

        final_base_depth = study3.best_params['base_max_depth']
        final_base_estimator = DecisionTreeRegressor(max_depth=final_base_depth, random_state=42)
        # print(f"创建基础决策树估计器，max_depth={final_base_depth}") # 静默

        adaboost_params = {
            key: value for key, value in study3.best_params.items()
            if key != 'base_max_depth'
        }
        # print(f"传递给 AdaBoostRegressor 的参数: {adaboost_params}") # 静默

        model_energy = AdaBoostRegressor(
            estimator=final_base_estimator,
            **adaboost_params,
            random_state=42
        )
        model_energy.fit(X_energy_scaled, y_energy)
        models['energy'] = model_energy
        print("能耗模型 最终训练完成。")

        return models, scalers, feature_lists

    except Exception as e:
        print(f"模型训练过程中发生错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

class ALF_PSO:
    """自适应学习因子PSO优化器 (处理混合整数变量)"""
    def __init__(self, models, scalers, feature_lists, data_row_series, config):
        self.model_temp1 = models['temp1']
        self.model_temp2 = models['temp2']
        self.model_energy = models['energy']
        self.energy_scaler = scalers['energy']
        self.temp1_features = feature_lists['temp1']
        self.temp2_features = feature_lists['temp2']
        self.energy_features = feature_lists['energy']

        self.original_row = data_row_series.copy()
        self.variable_cols = config['variable_cols']
        self.target_energy_col = config['target_energy_col']
        self.variable_bounds_dict = config['variable_bounds']

        self.const_temp_values = {
            col: self.original_row[col] for col in self.temp1_features
            if col not in self.variable_cols and col in self.original_row.index
        }
        self.const_energy_values = {
            col: self.original_row[col] for col in self.energy_features
            if col not in self.variable_cols and col in self.original_row.index
        }

        self.original_energy = self.original_row[self.target_energy_col]
        self.best_energy = np.inf
        self.best_params_vec = None
        self.dim = len(self.variable_cols)

        missing_in_row = [col for col in self.variable_cols if col not in self.original_row.index]
        if missing_in_row:
             raise ValueError(f"初始化ALF_PSO时, data_row_series缺少列: {missing_in_row}")

        try:
            self.lb = np.array([self.variable_bounds_dict[col][0] for col in self.variable_cols])
            self.ub = np.array([self.variable_bounds_dict[col][1] for col in self.variable_cols])
        except KeyError as e:
            raise ValueError(f"配置错误：在 variable_bounds 中找不到列 '{e}' 的边界定义。")
        except IndexError as e:
             raise ValueError(f"配置错误：列 '{self.variable_cols[len(self.lb)]}' 的边界定义格式不正确 (应为元组或列表)。")

        # 识别整数列的索引
        self.integer_indices = [i for i, col in enumerate(self.variable_cols) if col in INTEGER_COLS_SANITIZED]
        # *** 修改点：移除这里的打印 ***
        # if self.integer_indices:
        #      print(f"初始化ALF_PSO: 将处理以下索引的变量为整数(0/1): {self.integer_indices}")

    def _round_integer_vars(self, x_vec):
        """将向量中指定索引的值四舍五入到0或1"""
        if not self.integer_indices:
            return x_vec
        x_rounded = x_vec.copy()
        for idx in self.integer_indices:
            x_rounded[idx] = 1.0 if x_rounded[idx] >= 0.5 else 0.0
        return x_rounded

    def _create_input_df(self, x_vec, feature_list, const_values):
        """通用的创建模型输入DataFrame的辅助函数"""
        x_processed = self._round_integer_vars(x_vec)
        input_data = {col: x_processed[i] for i, col in enumerate(self.variable_cols)}
        input_data.update(const_values)
        missing_features = [f for f in feature_list if f not in input_data]
        if missing_features:
             补充成功 = True
             for mf in missing_features:
                 if mf in self.original_row.index:
                     input_data[mf] = self.original_row[mf]
                 else:
                     # print(f"  错误: 无法为 _create_input_df 补充缺失特征 '{mf}', 在原始数据中也找不到。") # 静默
                     补充成功 = False
                     break
             if not 补充成功:
                 return None

        try:
            input_data_filtered = {k: v for k, v in input_data.items() if k in feature_list}
            input_df = pd.DataFrame([input_data_filtered])
            input_df = input_df[feature_list]
            return input_df
        except KeyError as e:
            # print(f"  错误: _create_input_df 创建 DataFrame 时 KeyError: {e}。需要列: {feature_list}, 实际列: {list(input_data_filtered.keys())}") # 静默
            return None
        except Exception as e:
            # print(f"  错误: _create_input_df 创建 DataFrame 时发生未知错误: {e}") # 静默
            return None


    def objective_function(self, x_vec):
        """PSO的目标函数，评估给定参数组合(x_vec)的能耗"""
        try:
            if np.any(x_vec < self.lb) or np.any(x_vec > self.ub):
                return np.inf

            # 温度预测和约束检查
            temp1_input_df = self._create_input_df(x_vec, self.temp1_features, self.const_temp_values)
            if temp1_input_df is None: return np.inf
            temp2_input_df = self._create_input_df(x_vec, self.temp2_features, self.const_temp_values)
            if temp2_input_df is None: return np.inf

            pred_temp1 = self.model_temp1.predict(temp1_input_df)[0]
            pred_temp2 = self.model_temp2.predict(temp2_input_df)[0]

            temp_constraint_penalty = 0
            max_temp1_limit = 21.55
            max_temp2_limit = 20.91
            if pred_temp1 > max_temp1_limit: temp_constraint_penalty += (pred_temp1 - max_temp1_limit) * 1e6
            if pred_temp2 > max_temp2_limit: temp_constraint_penalty += (pred_temp2 - max_temp2_limit) * 1e6
            if temp_constraint_penalty > 0: return temp_constraint_penalty + 1e9

            # 能量预测
            energy_input_df = self._create_input_df(x_vec, self.energy_features, self.const_energy_values)
            if energy_input_df is None: return np.inf

            try:
                energy_input_df_reordered = energy_input_df[self.energy_features]
                energy_input_scaled = self.energy_scaler.transform(energy_input_df_reordered)
            except ValueError as e:
                # print(f"错误: Scaler transform失败: {e}.") # 静默
                return np.inf
            except Exception as e:
                # print(f"错误: Scaler transform时发生未知错误: {e}") # 静默
                return np.inf

            pred_energy = self.model_energy.predict(energy_input_scaled)[0]

            # 计算最终目标值
            if pred_energy >= self.original_energy:
                objective_val = self.original_energy + abs(pred_energy - self.original_energy) + 1e-6
                return objective_val
            else:
                return pred_energy

        except Exception as e:
            # print(f"!!!!!!!!!! objective_function 内部发生严重错误: {e}") # 静默
            return np.inf


    def optimize(self, swarmsize=27, maxiter=10, w=0.729, c1_range=(0.5, 2.5), c2_range=(0.5, 2.5)):
        """执行PSO优化过程"""
        c1_min, c1_max = c1_range
        c2_min, c2_max = c2_range

        swarm_pos = np.random.uniform(self.lb, self.ub, (swarmsize, self.dim))
        for i in range(swarmsize):
            swarm_pos[i] = self._round_integer_vars(swarm_pos[i])

        vel_range = (self.ub - self.lb) * 0.1
        swarm_vel = np.random.uniform(-vel_range, vel_range, (swarmsize, self.dim))

        swarm_pbest_pos = swarm_pos.copy()
        swarm_pbest_val = np.array([self.objective_function(p) for p in swarm_pos])

        if np.all(np.isinf(swarm_pbest_val)):
            # print(f"警告: 初始化时所有粒子的评估值均为无穷大。可能无法找到有效解。原始能耗: {self.original_energy:.4f}") # 静默
            gbest_idx = 0
            gbest_val = np.inf
        else:
            gbest_idx = np.nanargmin(swarm_pbest_val)
            gbest_val = swarm_pbest_val[gbest_idx]

        gbest_pos = swarm_pbest_pos[gbest_idx].copy()

        for iter_num in range(maxiter):
            c1 = c1_max - (c1_max - c1_min) * iter_num / maxiter
            c2 = c2_min + (c2_max - c2_min) * iter_num / maxiter

            r1 = np.random.random((swarmsize, self.dim))
            r2 = np.random.random((swarmsize, self.dim))

            swarm_vel = w * swarm_vel + \
                        c1 * r1 * (swarm_pbest_pos - swarm_pos) + \
                        c2 * r2 * (gbest_pos - swarm_pos)
            swarm_pos = swarm_pos + swarm_vel
            swarm_pos = np.clip(swarm_pos, self.lb, self.ub)
            swarm_pos_processed = np.array([self._round_integer_vars(p) for p in swarm_pos])
            current_val = np.array([self.objective_function(p) for p in swarm_pos_processed])

            update_mask = current_val < swarm_pbest_val
            swarm_pbest_pos[update_mask] = swarm_pos_processed[update_mask]
            swarm_pbest_val[update_mask] = current_val[update_mask]

            if np.any(update_mask):
                iter_best_idx = np.nanargmin(swarm_pbest_val)
                if swarm_pbest_val[iter_best_idx] < gbest_val:
                    gbest_pos = swarm_pbest_pos[iter_best_idx].copy()
                    gbest_val = swarm_pbest_val[iter_best_idx]

        if gbest_val < self.original_energy - 1e-9 and np.isfinite(gbest_val):
            self.best_energy = gbest_val
            self.best_params_vec = gbest_pos
            return self.best_params_vec
        else:
             return None


def save_results(results, variable_cols, date_col, target_energy_col, output_file):
    """保存优化结果到Excel文件"""
    if not results:
        print("没有结果可以保存。")
        return
    print(f"\n准备保存 {len(results)} 条结果到 {output_file}...")
    try:
        start_save_time = time.time()
        output_df = pd.DataFrame(results)

        print("开始格式化结果数据...")
        for col in variable_cols:
            if col not in output_df.columns:
                # print(f"警告: 结果中缺少列 '{col}'，跳过格式化。") # 静默
                continue
            if col in INTEGER_COLS_SANITIZED:
                 output_df[col] = pd.to_numeric(output_df[col], errors='coerce').round().astype('Int64')
            elif '开关阀门' in col:
                output_df[col] = pd.to_numeric(output_df[col], errors='coerce').round().astype('Int64')
            elif output_df[col].dtype in [np.float64, np.float32, float]:
                 output_df[col] = pd.to_numeric(output_df[col], errors='coerce')
                 output_df[col] = output_df[col].apply(lambda x: round(x, 4) if pd.notna(x) and np.isfinite(x) else x)


        if date_col in output_df.columns:
            try:
                output_df[date_col] = pd.to_datetime(output_df[date_col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                # print(f"警告: 格式化日期列 '{date_col}' 时出错: {e}。将保留原始格式或 NaT。") # 静默
                pass

        energy_cols_to_format = [target_energy_col, '优化后能耗']
        for ecol in energy_cols_to_format:
             if ecol in output_df.columns:
                 output_df[ecol] = pd.to_numeric(output_df[ecol], errors='coerce')
                 output_df[ecol] = output_df[ecol].apply(lambda x: round(x, 4) if pd.notna(x) and np.isfinite(x) else x)
        print("数据格式化完成。")

        print("开始排序列...")
        ordered_cols = [date_col] if date_col in output_df.columns else []
        ordered_cols += [col for col in variable_cols if col in output_df.columns]
        if target_energy_col in output_df.columns: ordered_cols.append(target_energy_col)
        if '优化后能耗' in output_df.columns: ordered_cols.append('优化后能耗')

        remaining_cols = [col for col in output_df.columns if col not in ordered_cols]
        final_cols = ordered_cols + remaining_cols
        output_df = output_df[final_cols]
        print(f"列已排序。最终列数: {len(final_cols)}")

        output_df.to_excel(output_file, index=False, engine='openpyxl')
        save_time = time.time() - start_save_time
        print(f"优化结果已成功保存至: {output_file}，耗时: {save_time:.2f} 秒。")

    except ImportError:
         print("错误: 需要安装 'openpyxl' 库来写入 Excel 文件。请运行: pip install openpyxl")
    except Exception as e:
        print(f"保存结果到Excel时发生错误: {e}")
        traceback.print_exc()


# --- NpEncoder 类 (用于JSON序列化NumPy类型) ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating):
            if np.isinf(obj) or np.isnan(obj): return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            cleaned_list = [self.default(x) for x in obj.tolist()]
            return cleaned_list
        elif isinstance(obj, pd.Timestamp): return obj.isoformat()
        elif pd.isna(obj): return None
        return super(NpEncoder, self).default(obj)


# --- 主函数 ---
def main():
    overall_start_time = time.time()
    try:
        # --- 1. 加载和预处理数据 ---
        print("="*20 + " 1. 数据加载与预处理 " + "="*20)
        (df_filtered, variable_cols, temp1_features, temp2_features,
         target_temp1_col, target_temp2_col, target_energy_col, date_col) = load_data()

        # --- 2. 准备模型训练数据 ---
        print("\n" + "="*20 + " 2. 准备模型训练数据 " + "="*20)
        X_temp1 = df_filtered[temp1_features]
        y_temp1 = df_filtered[target_temp1_col]
        X_temp2 = df_filtered[temp2_features]
        y_temp2 = df_filtered[target_temp2_col]
        energy_features_for_training = variable_cols
        X_energy_train = df_filtered[energy_features_for_training]
        y_energy = df_filtered[target_energy_col]

        if X_temp1.empty or y_temp1.empty or X_temp2.empty or y_temp2.empty or X_energy_train.empty or y_energy.empty:
            raise ValueError("关键模型输入(X)或输出(y)数据为空，无法继续训练。请检查 load_data 中的 dropna 逻辑。")
        print(f"训练数据准备完成:")
        print(f"  Temp1: X({X_temp1.shape}), y({y_temp1.shape}) - Features: {len(temp1_features)}")
        print(f"  Temp2: X({X_temp2.shape}), y({y_temp2.shape}) - Features: {len(temp2_features)}")
        print(f"  Energy: X({X_energy_train.shape}), y({y_energy.shape}) - Features: {len(energy_features_for_training)}")

        # --- 3. 模型训练与超参数优化 ---
        print("\n" + "="*20 + " 3. 模型训练与优化 " + "="*20)
        training_start_time = time.time()
        models, scalers, trained_feature_lists = train_models(
            X_temp1, X_temp2, y_temp1, y_temp2, X_energy_train, y_energy)
        training_time = time.time() - training_start_time
        print(f"\n所有模型训练完成，总耗时: {training_time:.2f} 秒。")

        # --- 4. 保存模型、Scaler 和配置 ---
        print("\n" + "="*20 + " 4. 保存训练产物 " + "="*20)
        config = {
            'variable_cols': variable_cols,
            'temp1_features': trained_feature_lists['temp1'],
            'temp2_features': trained_feature_lists['temp2'],
            'energy_features': trained_feature_lists['energy'],
            'target_energy_col': target_energy_col,
            'variable_bounds': VARIABLE_BOUNDS,
            'date_col': date_col,
            'integer_cols': INTEGER_COLS_SANITIZED,
            'training_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            joblib.dump((models, scalers, trained_feature_lists), f"{MODEL_PREFIX}_ensemble.pkl")
            print(f"模型、Scaler和特征列表已保存至: {MODEL_PREFIX}_ensemble.pkl")
            with open(META_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4, cls=NpEncoder)
            print(f"配置元数据已保存至: {META_FILE}")
        except Exception as e:
             print(f"保存模型/Scaler或配置文件时出错: {e}")
             traceback.print_exc()

        # --- 5. 逐行参数优化 ---
        print("\n" + "="*20 + " 5. 逐行参数优化 (ALF-PSO) " + "="*20)

        # *** 修改点：在循环外打印整数索引信息 (如果存在) ***
        integer_indices_for_info = [i for i, col in enumerate(variable_cols) if col in INTEGER_COLS_SANITIZED]
        if integer_indices_for_info:
            print(f"优化器提示: 将处理以下索引的变量为整数(0/1): {integer_indices_for_info}")
        # ************************************************

        results = []
        total_rows = len(df_filtered)
        processed_count = 0
        optimized_count = 0
        optimization_start_time = time.time()

        optimizer_config = config.copy()

        for idx in df_filtered.index:
            processed_count += 1
            # *** 修改点：移除原进度打印，或根据需要保留/修改 ***
            # if processed_count == 1 or processed_count % 50 == 0 or processed_count == total_rows:
            #      # ... (原进度打印逻辑) ...
            #      pass # 暂时移除或保持静默

            current_row_data = df_filtered.loc[idx]
            original_energy = current_row_data.get(target_energy_col, np.nan)
            current_date = current_row_data.get(date_col, None)

            if pd.isna(original_energy) or not np.isfinite(original_energy):
                result_row = {date_col: current_date}
                for col in variable_cols: result_row[col] = current_row_data.get(col, np.nan)
                result_row[target_energy_col] = original_energy
                result_row['优化后能耗'] = original_energy
                results.append(result_row)
                # *** 修改点：打印无效能耗信息 ***
                print(f"记录 {processed_count}/{total_rows} (索引: {idx}): 原始能耗无效 ({original_energy})，跳过优化。")
                continue

            try:
                optimizer = ALF_PSO(
                    models=models, scalers=scalers, feature_lists=trained_feature_lists,
                    data_row_series=current_row_data, config=optimizer_config
                )
                optimized_params_vec = optimizer.optimize(swarmsize=27, maxiter=10)
            except Exception as opt_err:
                # print(f"  记录 {processed_count} (索引: {idx}): 初始化或执行优化器时出错: {opt_err}") # 静默
                optimized_params_vec = None

            result_row = {date_col: current_date}
            final_optimized_energy = original_energy # 默认等于原始能耗

            if optimized_params_vec is None:
                # 未找到更优解或优化出错，保留原始参数
                for col in variable_cols: result_row[col] = current_row_data.get(col, np.nan)
                result_row[target_energy_col] = original_energy
                result_row['优化后能耗'] = original_energy # 优化后能耗等于原始能耗
                # final_optimized_energy 保持为 original_energy
            else:
                # 优化成功
                optimized_count += 1
                for i, col in enumerate(variable_cols):
                    result_row[col] = optimized_params_vec[i]
                result_row[target_energy_col] = original_energy
                result_row['优化后能耗'] = optimizer.best_energy
                final_optimized_energy = optimizer.best_energy # 更新最终能耗

            results.append(result_row)

            # *** 修改点：统一打印优化结果 ***
            print(f"记录 {processed_count}/{total_rows} (索引: {idx}): "
                  f"原始能耗={original_energy:.4f} -> "
                  f"优化后能耗={final_optimized_energy:.4f}")
            # *********************************

        optimization_time = time.time() - optimization_start_time
        print(f"\n逐行参数优化完成。总耗时: {optimization_time:.2f} 秒。")
        final_success_rate = (optimized_count / (total_rows - pd.isna(df_filtered[target_energy_col]).sum()) * 100) if (total_rows - pd.isna(df_filtered[target_energy_col]).sum()) > 0 else 0 # 修正成功率计算，排除无效行
        print(f"总共处理 {total_rows} 条记录 (其中 {pd.isna(df_filtered[target_energy_col]).sum()} 条原始能耗无效)，有效记录中 {optimized_count} 条成功优化 (优化率: {final_success_rate:.2f}%)。")


        # --- 6. 保存最终结果 ---
        print("\n" + "="*20 + " 6. 保存优化结果 " + "="*20)
        save_results(results, variable_cols, date_col, target_energy_col, OUTPUT_FILE)

    # --- 异常处理 ---
    except FileNotFoundError as fnf_err:
        print(f"\n错误: 关键文件未找到: {fnf_err}")
        sys.exit(1)
    except ValueError as val_err:
        print(f"\n错误: 数据或配置无效: {val_err}")
        traceback.print_exc()
        sys.exit(1)
    except RuntimeError as run_err:
        print(f"\n错误: 运行时发生错误 (例如模型训练失败): {run_err}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n!!!!!!!!!! 程序主流程发生未捕获的严重错误 !!!!!!!!!!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("详细追溯信息:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        overall_end_time = time.time()
        total_runtime = overall_end_time - overall_start_time
        print(f"\n程序执行完毕。总耗时: {total_runtime:.2f} 秒 ({total_runtime/60:.2f} 分钟)。")

if __name__ == "__main__":
    main()