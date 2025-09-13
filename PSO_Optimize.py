# -*- coding: utf-8 -*-
import sys
import traceback
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from pyswarm import pso # Using pyswarm for PSO
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time

# --- Configuration ---
DATA_PATH = "PUE数据汇总_processed.xlsx"
OUTPUT_FILE = "PSO_调优后的参数_v2.xlsx" # 修改输出文件名以表明是调整过的PSO

# --- Helper Functions & Global Variables ---

def enhanced_sanitize(name):
    """Cleans special characters from column names."""
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9_]+', '_', str(name))
    cleaned = re.sub(r'^_+|_+$', '', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned

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
    'V1_6_开关阀门_开到位反馈': (0, 1),
    'V1_7_开关阀门_开到位反馈': (0, 1),
    'V16_1_开关阀门_开到位反馈': (0, 1),
    'V16_2_开关阀门_开到位反馈': (0, 1),
    'CWP_1_冷却水泵_频率反馈显示': (0.0, 49.87)
}

VARIABLE_BOUNDS = {enhanced_sanitize(k): v for k, v in VARIABLE_BOUNDS_ORIGINAL.items()}
VARIABLE_COLS = list(VARIABLE_BOUNDS.keys())

INTEGER_COLS_SANITIZED = [
    enhanced_sanitize('V1_6_开关阀门_开到位反馈'),
    enhanced_sanitize('V1_7_开关阀门_开到位反馈'),
    enhanced_sanitize('V16_1_开关阀门_开到位反馈'),
    enhanced_sanitize('V16_2_开关阀门_开到位反馈')
]

TEMP1_CONSTRAINT_MAX = 21.55
TEMP2_CONSTRAINT_MAX = 20.91
PENALTY_VALUE = 1e18

# --- Function Definitions ---

def load_data():
    """Loads and preprocesses data."""
    try:
        print(f"Loading data from: {DATA_PATH}...")
        df = pd.read_excel(DATA_PATH)
        df.columns = [enhanced_sanitize(col) for col in df.columns]
        print(f"Data loaded. Original shape: {df.shape}")

        target_temp1_col = enhanced_sanitize('CWHR_1_t_0_5h')
        target_temp2_col = enhanced_sanitize('CWHR_2_t_0_5h')
        target_energy_col = enhanced_sanitize('冷源群控系统实时能耗_单位_kW')
        date_col = enhanced_sanitize('测量日期')

        required_cols = VARIABLE_COLS + [target_temp1_col, target_temp2_col, target_energy_col, date_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        exclude_cols = [target_temp1_col, target_temp2_col, target_energy_col, date_col]
        temp_features = [col for col in df.columns if col not in exclude_cols]
        print(f"Temperature models will use {len(temp_features)} features.")

        print("Converting to numeric and handling NaNs...")
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        cols_to_check_na = list(set(temp_features + required_cols))
        df_filtered = df_numeric.dropna(subset=cols_to_check_na)
        print(f"Rows after dropping NaNs: {len(df_filtered)} (from {len(df)})")
        if df_filtered.empty:
            raise ValueError("No valid data remaining after NaN removal.")

        if date_col in df_filtered.columns:
            df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])

        return (
            df_filtered[temp_features], df_filtered[target_temp1_col], df_filtered[target_temp2_col],
            df_filtered[VARIABLE_COLS], df_filtered[target_energy_col], df_filtered[date_col],
            df_filtered, temp_features
        )
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


def train_models(X_temp, y_temp1, y_temp2, X_energy, y_energy):
    """Trains temperature and energy models with fixed hyperparameters."""
    try:
        print("\n--- Training Temperature Models (LightGBM) ---")
        model_temp1 = lgb.LGBMRegressor(
            num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42
        )
        model_temp1.fit(X_temp, y_temp1)
        model_temp2 = lgb.LGBMRegressor(
            num_leaves=31, learning_rate=0.05, n_estimators=300, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42
        )
        model_temp2.fit(X_temp, y_temp2)
        print("Temperature models trained.")

        print("\n--- Training Energy Model (AdaBoostRegressor) ---")
        base_estimator = DecisionTreeRegressor(max_depth=8, random_state=42)
        energy_model = AdaBoostRegressor(
            estimator=base_estimator, n_estimators=200, learning_rate=0.1,
            loss='square', random_state=42
        )
        energy_model.fit(X_energy, y_energy)
        print("Energy model trained.")
        return model_temp1, model_temp2, energy_model
    except Exception as e:
        print(f"Error training models: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


def save_results(results, variable_cols, date_col_name, target_energy_col_name, output_file):
    """Saves optimization results to an Excel file."""
    if not results:
        print("No results to save.")
        return
    print(f"\nSaving {len(results)} results to {output_file}...")
    try:
        output_df = pd.DataFrame(results)
        print("Formatting results...")
        # Ensure column names exist before accessing
        if date_col_name not in output_df.columns:
            print(f"Warning: Date column '{date_col_name}' not found in results.")
            date_col_name = None # Prevent errors later

        for col in variable_cols:
            if col not in output_df.columns: continue
            if col in INTEGER_COLS_SANITIZED:
                output_df[col] = pd.to_numeric(output_df[col], errors='coerce').round().astype('Int64')
            elif '开关阀门' in col:
                output_df[col] = pd.to_numeric(output_df[col], errors='coerce').round().astype('Int64')
            elif pd.api.types.is_float_dtype(output_df[col]):
                 output_df[col] = pd.to_numeric(output_df[col], errors='coerce').round(4)

        if date_col_name and date_col_name in output_df.columns:
             try: output_df[date_col_name] = pd.to_datetime(output_df[date_col_name]).dt.strftime('%Y-%m-%d %H:%M:%S')
             except Exception: pass # Ignore date formatting errors

        # Safely format numeric columns
        for col in ['原始能耗', '优化后能耗']:
            if col in output_df.columns:
                output_df[col] = pd.to_numeric(output_df[col], errors='coerce').round(4)
        for col in ['预测温度1(优化后)', '预测温度2(优化后)']:
             if col in output_df.columns:
                output_df[col] = pd.to_numeric(output_df[col], errors='coerce').round(2)

        # Define base ordered columns, checking existence
        ordered_cols_base = [col for col in [date_col_name, '原始能耗', '优化后能耗'] if col and col in output_df.columns]
        ordered_cols_vars = [col for col in variable_cols if col in output_df.columns]
        ordered_cols_temps = [col for col in ['预测温度1(优化后)', '预测温度2(优化后)'] if col in output_df.columns]

        ordered_cols = ordered_cols_base + ordered_cols_vars + ordered_cols_temps
        final_cols = [c for c in ordered_cols if c in output_df.columns] # Filter again just in case
        remaining_cols = [c for c in output_df.columns if c not in final_cols]
        final_cols.extend(remaining_cols)
        output_df = output_df[final_cols]

        output_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Results successfully saved to: {output_file}")

    except ImportError:
        print("Error: 'openpyxl' library needed to write Excel files. Install with: pip install openpyxl")
    except Exception as e:
        print(f"Error saving results to Excel: {e}")
        traceback.print_exc()


# --- Class Definitions ---

class PSOWrapper:
    """Wraps models and data for PSO optimization."""
    def __init__(self, model_temp1, model_temp2, model_energy, full_data_row, temp_features):
        self.model_temp1 = model_temp1
        self.model_temp2 = model_temp2
        self.model_energy = model_energy
        self.full_data_row = full_data_row.copy()
        self.variable_cols = VARIABLE_COLS
        self.temp_features = temp_features
        self.integer_indices = [i for i, col in enumerate(self.variable_cols) if col in INTEGER_COLS_SANITIZED]
        self.const_temp_values = {
            col: self.full_data_row[col] for col in self.temp_features
            if col not in self.variable_cols and col in self.full_data_row.index
        }
        self.target_energy_col_sanitized = enhanced_sanitize('冷源群控系统实时能耗_单位_kW')
        self.original_energy = self.full_data_row[self.target_energy_col_sanitized]
        self.original_params = np.array([self.full_data_row[col] for col in self.variable_cols])
        self.lb = np.array([VARIABLE_BOUNDS[col][0] for col in self.variable_cols])
        self.ub = np.array([VARIABLE_BOUNDS[col][1] for col in self.variable_cols])

    def _round_integer_vars(self, x_vec):
        if not self.integer_indices: return x_vec
        x_rounded = x_vec.copy()
        for idx in self.integer_indices:
            x_rounded[idx] = 1.0 if x_rounded[idx] >= 0.5 else 0.0
        return x_rounded

    def _create_input_df(self, x, feature_list, const_values):
        x_processed = self._round_integer_vars(x)
        input_data = {col: x_processed[i] for i, col in enumerate(self.variable_cols)}
        input_data.update(const_values)
        missing_features = [f for f in feature_list if f not in input_data]
        for mf in missing_features:
            if mf in self.full_data_row.index: input_data[mf] = self.full_data_row[mf]
            else: return None
        try: return pd.DataFrame([input_data])[feature_list]
        except KeyError: return None

    def objective_function(self, x):
        try:
            temp_input_df = self._create_input_df(x, self.temp_features, self.const_temp_values)
            if temp_input_df is None: return PENALTY_VALUE
            pred_temp1 = self.model_temp1.predict(temp_input_df)[0]
            pred_temp2 = self.model_temp2.predict(temp_input_df)[0]
            if pred_temp1 > TEMP1_CONSTRAINT_MAX or pred_temp2 > TEMP2_CONSTRAINT_MAX:
                penalty = 0
                if pred_temp1 > TEMP1_CONSTRAINT_MAX: penalty += (pred_temp1 - TEMP1_CONSTRAINT_MAX)
                if pred_temp2 > TEMP2_CONSTRAINT_MAX: penalty += (pred_temp2 - TEMP2_CONSTRAINT_MAX)
                return PENALTY_VALUE + penalty
            x_processed_energy = self._round_integer_vars(x)
            energy_input_data = {col: x_processed_energy[i] for i, col in enumerate(self.variable_cols)}
            energy_input_df = pd.DataFrame([energy_input_data])[self.variable_cols]
            pred_energy = self.model_energy.predict(energy_input_df)[0]
            return pred_energy
        except Exception: return PENALTY_VALUE

    # --- MODIFIED optimize method ---
    def optimize(self, swarmsize=10, maxiter=8): # Reduced swarmsize and maxiter
        """Performs PSO optimization with potentially sub-optimal parameters."""
        print(f"  Starting Standard PSO: swarmsize={swarmsize}, maxiter={maxiter}")
        start_pso_time = time.time()

        # === Use fixed, potentially sub-optimal parameters ===
        pso_params = {
            'omega': 0.8,  # Higher fixed inertia
            'phip': 0.6,   # Cognitive slightly higher
            'phig': 0.4,   # Social slightly lower
            'debug': False
        }
        # ====================================================

        x_opt, f_opt = pso(
            self.objective_function, lb=self.lb, ub=self.ub,
            swarmsize=swarmsize, maxiter=maxiter,
            minstep=1e-6, minfunc=1e-6,
            **pso_params # Pass the chosen parameters
        )
        pso_time = time.time() - start_pso_time
        print(f"  Standard PSO finished in {pso_time:.2f}s. Best cost found: {f_opt:.4f}")

        final_params_processed = self._round_integer_vars(x_opt)
        temp_input_opt = self._create_input_df(x_opt, self.temp_features, self.const_temp_values)
        pred_temp1_opt, pred_temp2_opt = np.nan, np.nan
        constraints_met = False
        if temp_input_opt is not None:
            try:
                pred_temp1_opt = self.model_temp1.predict(temp_input_opt)[0]
                pred_temp2_opt = self.model_temp2.predict(temp_input_opt)[0]
                constraints_met = (pred_temp1_opt <= TEMP1_CONSTRAINT_MAX and
                                   pred_temp2_opt <= TEMP2_CONSTRAINT_MAX)
                if not constraints_met:
                    print(f"  Warning: Best Std PSO solution violates temp constraints (T1:{pred_temp1_opt:.2f}, T2:{pred_temp2_opt:.2f})")
            except Exception as e: print(f"  Warning: Error predicting temps for final Std PSO solution: {e}")

        energy_improved = False
        if constraints_met and f_opt < self.original_energy - 1e-6:
             energy_improved = True
             print(f"  Std PSO Optimization successful: Constraints met & energy reduced from {self.original_energy:.4f} to {f_opt:.4f}")
        elif constraints_met: print(f"  Std PSO Optimization constraints met, but energy not improved ({f_opt:.4f} >= {self.original_energy:.4f})")

        if constraints_met and energy_improved:
            return final_params_processed, f_opt
        else:
            return self.original_params, self.original_energy


# --- Main Execution Block ---

def main():
    """Main function to run the optimization workflow."""
    overall_start_time = time.time()
    try:
        print("--- 1. Load Data ---")
        (X_temp, y_temp1, y_temp2, X_energy, y_energy, dates, df_full, temp_features) = load_data()

        print("\n--- 2. Train Models ---")
        model_temp1, model_temp2, model_energy = train_models(
            X_temp, y_temp1, y_temp2, X_energy, y_energy
        )

        print("\n--- 3. Starting Row-by-Row PSO Optimization ---")
        results = []
        total_rows = len(df_full)
        optimization_start_time = time.time()
        optimized_count = 0

        date_col_sanitized = enhanced_sanitize('测量日期')
        energy_col_sanitized = enhanced_sanitize('冷源群控系统实时能耗_单位_kW')

        for idx in range(total_rows):
            current_row_data = df_full.iloc[idx]
            current_date_val = current_row_data.get(date_col_sanitized, pd.NaT)
            print(f"\nProcessing row {idx+1}/{total_rows} (Date: {current_date_val})...")

            original_energy_val = current_row_data.get(energy_col_sanitized, np.nan)
            if pd.isna(original_energy_val) or not np.isfinite(original_energy_val):
                print("  Skipping row due to invalid original energy.")
                continue

            try:
                optimizer = PSOWrapper(
                    model_temp1, model_temp2, model_energy,
                    current_row_data, temp_features
                )
                # === Pass potentially reduced swarmsize/maxiter ===
                final_params_vec, final_energy_val = optimizer.optimize(swarmsize=10, maxiter=8)
                # =================================================

                if not np.array_equal(final_params_vec, optimizer.original_params):
                     optimized_count += 1
                # else: (No need to print again, optimize method already does)

                result_row = {}
                result_row[date_col_sanitized] = current_date_val
                result_row['原始能耗'] = optimizer.original_energy
                result_row['优化后能耗'] = final_energy_val

                for i, col in enumerate(VARIABLE_COLS):
                    result_row[col] = final_params_vec[i]

                temp_input_final = optimizer._create_input_df(final_params_vec, optimizer.temp_features, optimizer.const_temp_values)
                pred_temp1_final, pred_temp2_final = np.nan, np.nan
                if temp_input_final is not None:
                    try:
                        pred_temp1_final = optimizer.model_temp1.predict(temp_input_final)[0]
                        pred_temp2_final = optimizer.model_temp2.predict(temp_input_final)[0]
                    except Exception: pass
                result_row['预测温度1(优化后)'] = pred_temp1_final
                result_row['预测温度2(优化后)'] = pred_temp2_final

                results.append(result_row)

            except Exception as e:
                print(f"Error processing row {idx+1}: {e}")
                traceback.print_exc()

            if (idx + 1) % 20 == 0 or (idx + 1) == total_rows:
                 print(f"\nSaving intermediate results ({len(results)} rows processed)...")
                 save_results(results, VARIABLE_COLS, date_col_sanitized, energy_col_sanitized, OUTPUT_FILE)

        optimization_time = time.time() - optimization_start_time
        print(f"\n--- Optimization Complete ---")
        print(f"Total time: {optimization_time:.2f} seconds.")
        success_rate = (optimized_count / total_rows * 100) if total_rows > 0 else 0
        print(f"Processed {total_rows} rows. Successfully optimized: {optimized_count} ({success_rate:.2f}%).")

        print("\n--- 4. Saving Final Results ---")
        # Final save is handled within the loop

    except Exception as e:
        print(f"\n--- Top Level Error ---")
        print(f"An error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        overall_end_time = time.time()
        print(f"\nTotal script execution time: {overall_end_time - overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    main()