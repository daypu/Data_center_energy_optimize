import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # 用于可视化（可选）
# 确保已安装 pandas, numpy, matplotlib, openpyxl:
# pip install pandas numpy matplotlib openpyxl

# --- 1. 配置参数 ---
file_path = 'PUE数据汇总.xlsx' # 你的 Excel 文件路径
column_name = '冷源群控系统实时能耗_单位_kW' # 需要处理的列名
threshold = 20.0 # 偏差阈值
window_size = 10 # 计算移动平均的窗口大小 (可以调整, 之前你设置为10)
output_file_path = 'PUE数据汇总_processed.xlsx' # 输出的 Excel 文件名

# *** 重要修改：指定要读取的工作表名称 ***
# 请根据你的 Excel 文件实际情况修改 'Sheet1' 为正确的工作表名
target_sheet_name = 'Sheet1' # <--- 不再使用 None，而是指定确切的工作表名

# --- 2. 加载数据 ---
try:
    # 使用 read_excel 读取 .xlsx 文件中指定的工作表
    # 需要安装 openpyxl (pip install openpyxl)
    print(f"正在读取 Excel 文件: {file_path} (工作表: {target_sheet_name})...")
    if target_sheet_name is None:
        print("错误：未指定目标工作表名 (target_sheet_name 不能为 None)。请设置正确的表名。")
        exit()
    df = pd.read_excel(file_path, sheet_name=target_sheet_name, engine='openpyxl')
    # 成功读取后，df 应该是一个 DataFrame
    print(f"工作表 '{target_sheet_name}' 读取成功，数据包含 {df.shape[0]} 行 和 {df.shape[1]} 列。")

except FileNotFoundError:
    print(f"错误：文件未找到 {file_path}")
    exit()
except ImportError:
     print("\n错误：读取/写入 Excel 文件需要 'openpyxl' 库。请先安装 (pip install openpyxl)。")
     exit()
except ValueError as ve:
    # 捕获 sheet_name 不存在的错误
    if "Worksheet" in str(ve) and "does not exist" in str(ve):
         print(f"\n错误：指定的工作表名称 '{target_sheet_name}' 在文件 '{file_path}' 中不存在。")
         print("请检查 Excel 文件或修改代码中的 target_sheet_name。")
    else:
         print(f"读取 Excel 文件时发生值错误: {ve}") # 其他 ValueError
    exit()
except Exception as e:
    # 捕获其他可能的读取错误
    print(f"读取 Excel 文件 '{file_path}' 时出错: {e}")
    exit()

# --- 从这里开始的代码应该可以正常工作了 ---

# 检查列是否存在
if column_name not in df.columns:
    print(f"错误：列 '{column_name}' 在 Excel 文件的工作表 '{target_sheet_name}' 中未找到。")
    print(f"可用的列有: {df.columns.tolist()}")
    exit()

# 确保目标列是数值类型，非数值转为 NaN
# 制作一个副本用于可视化原始数据
try:
    original_data_for_plot = df[column_name].copy()
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
except KeyError:
     print(f"错误：处理列 '{column_name}' 时发生 Key 错误，请检查列名是否准确无误。")
     exit()

initial_nan_count = df[column_name].isna().sum()
if initial_nan_count > 0:
    print(f"注意：目标列 '{column_name}' 包含 {initial_nan_count} 个初始 NaN 值。")

# --- 3. 计算趋势线 (移动平均) ---
print("正在计算趋势线和偏差...")
df['_trend'] = df[column_name].rolling(window=window_size, center=True, min_periods=1).mean()
df['_trend'] = df['_trend'].fillna(method='bfill').fillna(method='ffill')

# --- 4. 计算偏差 ---
df['_deviation'] = df[column_name] - df['_trend']

# --- 5. 识别并调整极端偏差，直接修改目标列 ---
print("正在识别并调整极端偏差...")
outlier_condition = (df['_deviation'].abs() > threshold) & df[column_name].notna() & df['_trend'].notna() & df['_deviation'].notna()
outlier_indices = df[outlier_condition].index

original_outlier_values = df.loc[outlier_indices, column_name].copy()
df.loc[outlier_indices, column_name] = df.loc[outlier_indices, '_trend'] + np.sign(df.loc[outlier_indices, '_deviation']) * threshold
adjusted_outlier_values = df.loc[outlier_indices, column_name].copy()

print("数据处理完成。")
print(f"共找到并调整了 {len(outlier_indices)} 个极端偏差点。")

# --- 6. 清理临时列 ---
print("正在清理临时计算列...")
df = df.drop(columns=['_trend', '_deviation'])

# --- 7. 保存结果到 Excel 文件 ---
print(f"正在保存处理后的数据到 Excel 文件: {output_file_path}...")
try:
    df.to_excel(output_file_path, index=False, engine='openpyxl')
    print(f"处理后的数据已成功保存。")
except ImportError:
     print("\n错误：写入 Excel 文件需要 'openpyxl' 库。请先安装 (pip install openpyxl)。")
except Exception as e:
    print(f"\n保存 Excel 文件时出错: {e}")

# --- 8. 可视化对比 (可选) ---
print("正在生成可视化图表...")
try:
    plt.figure(figsize=(15, 7))
    plt.plot(original_data_for_plot.index, original_data_for_plot, label='原始数据', alpha=0.6, linestyle=':')
    trend_for_plot = original_data_for_plot.rolling(window=window_size, center=True, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill')
    plt.plot(trend_for_plot.index, trend_for_plot, label=f'趋势线 (移动平均, w={window_size})', color='orange', linewidth=2)
    plt.plot(df.index, df[column_name], label='调整后数据 (最终列)', alpha=0.8, color='green')

    if not outlier_indices.empty:
        plt.scatter(outlier_indices, original_outlier_values, color='red', label='原始极端点值', s=50, zorder=5)
        plt.scatter(outlier_indices, adjusted_outlier_values, color='purple', label='调整后极端点值', s=50, marker='x', zorder=5)
    else:
        print("未找到极端点，图表中不标记。")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(f'"{column_name}" 数据处理前后对比 (阈值={threshold}, 窗口={window_size})')
    plt.xlabel('数据点索引 (时间顺序)')
    plt.ylabel('能耗 (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    print("图表已生成，请查看弹出的窗口。")
    plt.show()

except Exception as e:
    print(f"\n生成可视化图表时出错: {e}")
    print("请检查是否安装了 matplotlib 库，以及系统是否支持中文字体。")

print("\n脚本执行完毕。")