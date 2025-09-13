import pandas as pd
# import seaborn as sns # Seaborn 在此脚本中未被使用，可以注释掉或删除
import matplotlib.pyplot as plt
import numpy as np

# 可能需要安装 openpyxl 库来支持写入 Excel: pip install openpyxl

# --- 用户指定的列 ---
variable_cols= [
    "冷机冷却侧出水压力PIT_1_CWS_1_压力表_实际值",
    "冷机冷却侧出水压力PIT_2_CWS_1_压力表_实际值",
    "冷机冷冻侧出水压力PIT_1_CWHS_1_压力表_实际值",
    "冷机冷冻侧出水压力PIT_2_CWHS_1_压力表_实际值",
    "冷机冷冻侧出水温度TIT_1_CWHS_1_温度计_实际值",
    "冷机冷冻侧出水温度TIT_2_CWHS_1_温度计_实际值",
    "冷机冷却侧出水温度TIT_1_CWS_1_温度计_实际值",
    "冷机冷却侧出水温度TIT_2_CWS_1_温度计_实际值",
    "CV1_1_比例阀门_开度反馈显示",
    "CV1_3_比例阀门_开度反馈显示",
    "CV10_1_比例阀门_开度反馈显示",
    "CV10_2_比例阀门_开度反馈显示",
    "CV11_1_比例阀门_开度反馈显示",
    "CV11_2_比例阀门_开度反馈显示",
    "CV14_1_比例阀门_开度反馈显示",
    "CV14_2_比例阀门_开度反馈显示",
    "CV15_1_比例阀门_开度反馈显示",
    "CV15_2_比例阀门_开度反馈显示",
    "CV2_1_比例阀门_开度反馈显示",
    "CV2_3_比例阀门_开度反馈显示",
    "CV9_1_比例阀门_开度反馈显示",
    "CV9_2_比例阀门_开度反馈显示",
    "V1_6_开关阀门_开到位反馈",
    "V1_7_开关阀门_开到位反馈",
    "V16_1_开关阀门_开到位反馈",
    "V16_2_开关阀门_开到位反馈",
    "CWP_1_冷却水泵_频率反馈显示",
    "冷冻侧回水环网温度TIT_6_CWHR_1_温度计_实际值",
    "冷冻侧回水环网温度TIT_6_CWHR_2_温度计_实际值",
    "冷机冷冻侧进水温度TIT_1_CWHR_3_温度计_实际值",
    "冷机冷冻侧进水温度TIT_2_CWHR_3_温度计_实际值",
    "冷机冷却侧进水温度TIT_1_CWR_4_温度计_实际值",
    "冷机冷却侧进水温度TIT_2_CWR_4_温度计_实际值",
    "冷冻侧回水环网流量FIT_6_CWHR_1_单向流量传感器_实际值",
    "冷冻侧回水环网流量FIT_6_CWHR_2_单向流量传感器_实际值",
    "冷冻侧末端支路回水压力PIT_6_CWHR_1_压力表_实际值",
    "冷冻侧末端支路回水压力PIT_6_CWHR_2_压力表_实际值",
    "冷冻侧末端支路回水压力PIT_6_CWHR_3_压力表_实际值",
    "冷冻侧末端支路回水压力PIT_6_CWHR_4_压力表_实际值",
    "冷冻侧回水环网压力PIT_6_CWHR_5_压力表_实际值",
    "冷冻侧回水环网压力PIT_6_CWHR_6_压力表_实际值",
    "室外环境温度Tw",
    "室外环境湿度_w",
    "冷冻侧末端支路供水压力PIT_6_CWHS_1_压力表_实际值",
    "冷冻侧末端支路供水压力PIT_6_CWHS_2_压力表_实际值",
    "冷冻侧末端支路供水压力PIT_6_CWHS_3_压力表_实际值",
    "冷冻侧末端支路供水压力PIT_6_CWHS_4_压力表_实际值",
    "冷冻侧供水环网压力PIT_6_CWHS_5_压力表_实际值",
    "冷冻侧供水环网压力PIT_6_CWHS_6_压力表_实际值",
    "冷冻侧供水环网压力PIT_6_CWHS_7_压力表_实际值",
    "冷冻侧供水环网压力PIT_6_CWHS_8_压力表_实际值",
    "冷冻侧供水环网温度TIT_6_CWHS_1_温度计_实际值",
    "冷冻侧供水环网温度TIT_6_CWHS_2_温度计_实际值",
    "冷冻侧供水环网温度TIT_6_CWHS_3_温度计_实际值",
    "冷冻侧供水环网温度TIT_6_CWHS_4_温度计_实际值",
    "Eit实时能耗_单位_kW"
]
# --- 结束用户指定的列 ---

# --- 文件读取 ---
file_path = 'PUE数据汇总_processed.xlsx'
sheet_name = 'Sheet1'

try:
    data = pd.read_excel(file_path, sheet_name=sheet_name)
except FileNotFoundError:
    print(f"错误：文件未找到，请确认路径 '{file_path}' 是否正确。")
    exit()
except ValueError as ve:
    if "Worksheet named" in str(ve) and sheet_name in str(ve):
        print(f"错误：在 Excel 文件 '{file_path}' 中未找到名为 '{sheet_name}' 的工作表。")
        try:
            print("尝试读取第一个工作表...")
            data = pd.read_excel(file_path, sheet_name=0)
            print(f"成功读取了第一个工作表（索引为0）。")
        except Exception as e_first:
            print(f"尝试读取第一个工作表失败: {e_first}")
            exit()
    else:
        print(f"读取 Excel 文件时出错: {ve}")
        exit()
except Exception as e:
    print(f"读取 Excel 文件时发生未知错误: {e}")
    exit()
# --- 结束文件读取 ---

# --- 数据清理与准备 ---
columns_to_drop = ["测量日期", "PUE", "CWHR_1_t_0.5h", "CWHR_2_t_0.5h"]
existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
if existing_columns_to_drop:
    data_cleaned = data.drop(columns=existing_columns_to_drop)
else:
    data_cleaned = data.copy()

# --- 修改：修正目标列名称中的下划线和括号 ---
# 检查多个可能的名称，或使用您确认的准确名称
possible_energy_cols = [
    '冷源群控系统实时能耗_单位_kW',  # 您之前代码中的名称
    '冷源群控系统实时能耗（单位：kW）'  # 您最初描述中的名称
]
energy_column = None
for col_name in possible_energy_cols:
    if col_name in data_cleaned.columns:
        energy_column = col_name
        print(f"找到目标能耗列: '{energy_column}'")
        break

if energy_column is None:
    print(f"错误：未能找到目标能耗列。请检查以下可能的名称是否在您的数据中：{possible_energy_cols}")
    print("可用的列名：", data_cleaned.columns.tolist())
    exit()

missing_vars = [col for col in variable_cols if col not in data_cleaned.columns]
if missing_vars:
    print(f"警告：以下指定的变量列不在数据中，将被忽略：{missing_vars}")
    variable_cols = [col for col in variable_cols if col in data_cleaned.columns]

if not variable_cols:
    print("错误：所有指定的变量列都不在数据中，无法进行相关性分析。")
    exit()

cols_for_corr = variable_cols + [energy_column]
data_filtered = data_cleaned[cols_for_corr].copy()

for col in data_filtered.columns:
    data_filtered[col] = pd.to_numeric(data_filtered[col], errors='coerce')

nan_rows_before = data_filtered.isnull().any(axis=1).sum()
if nan_rows_before > 0:
    print(f"警告：数据中发现 {nan_rows_before} 行包含非数值或缺失值，这些行将被删除以计算相关性。")
    data_filtered = data_filtered.dropna()

if data_filtered.empty or len(data_filtered) < 2:
    print("错误：在处理非数值和缺失值后，没有足够的数据（至少需要2行）进行相关性分析。")
    exit()
# --- 结束数据清理与准备 ---

# --- 相关性计算 ---
correlation_matrix_filtered = data_filtered.corr()

if energy_column not in correlation_matrix_filtered:
    print(f"错误：目标列 '{energy_column}' 可能由于全是NaN值或其他原因，无法计算相关性。")
    exit()

correlation_with_energy = correlation_matrix_filtered[energy_column].sort_values(ascending=False)
# --- 结束相关性计算 ---

# --- 字体和样式设置 ---
# 设置Matplotlib的字体为SimHei（黑体），以支持中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    print("已设置字体为 SimHei")
except:
    print("警告：未找到 SimHei 字体，尝试设置为 'Microsoft YaHei'。")
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        print("已设置字体为 Microsoft YaHei")
    except:
        print("警告：也未找到 'Microsoft YaHei' 字体。请确保系统安装了支持中文的字体。")

# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# --- 修改：增大全局字体大小 ---
plt.rcParams.update({'font.size': 12})  # 设置基础字体大小，可调整
# --- 结束字体和样式设置 ---


# --- 绘图部分 ---
# 准备绘图数据（移除能耗列自身的相关性）
plot_data = correlation_with_energy.drop(energy_column)

if plot_data.empty:
    print("信息：没有其他指定变量与目标能耗列的相关性可供绘制。")
else:
    # --- 修改：增大图形尺寸和各元素字体大小 ---
    plt.figure(figsize=(15, 10))  # 增大图形尺寸
    bars = plot_data.plot(kind='bar', color='skyblue')
    plt.title(f"冷源系统设备变量与能耗的相关性分析", fontsize=16)  # 增大标题字号
    plt.ylabel("相关系数 (Pearson Correlation Coefficient)", fontsize=14)  # 增大Y轴标签字号
    plt.xlabel("数据中心运行参数", fontsize=14)  # 增大X轴标签字号
    plt.xticks(rotation=45, ha='right', fontsize=10)  # 调整X轴刻度字号和旋转角度

    # 在每个柱子上方标记相关系数
    for patch in bars.patches:
        height = patch.get_height()
        va_pos = 'bottom' if height >= 0 else 'top'
        y_pos = height + 0.01 if height >= 0 else height - 0.04  # 微调垂直位置
        plt.text(patch.get_x() + patch.get_width() / 2, y_pos, f'{height:.2f}',
                 ha='center', va=va_pos, fontsize=9)  # 增大标注字号

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # 尽可能自动调整布局

    # --- 修改：保存图片而不是显示 ---
    output_plot_filename = 'Energy_correlation_plot.png'
    try:
        plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG
        print(f"相关性分析图已保存为: {output_plot_filename}")
    except Exception as e_save:
        print(f"保存图片时出错: {e_save}")
    # plt.show() # 注释掉原来的显示命令
    plt.close()  # 关闭图形，释放内存
    # --- 结束绘图修改 ---

# --- 新增：导出相关系数表格 ---
# 将 plot_data (Series) 转换为 DataFrame
# plot_data 的 index 是变量名，values 是相关系数
df_table = pd.DataFrame({
    '冷源群控系统设备变量名': plot_data.index,
    '与能耗的相关系数': plot_data.values
})

# 定义输出表格文件名
output_table_filename = 'Energy_correlation_table.xlsx'
try:
    # 使用 to_excel 保存为 Excel 文件，不包含索引列
    df_table.to_excel(output_table_filename, index=False)
    print(f"相关系数表格已保存为: {output_table_filename}")
except Exception as e_excel:
    print(f"导出表格到 Excel 时出错: {e_excel}")
    # 可以尝试导出为 CSV 作为备选
    output_table_filename_csv = 'correlation_table.csv'
    try:
        df_table.to_csv(output_table_filename_csv, index=False, encoding='utf-8-sig')  # utf-8-sig 通常能被Excel正确识别中文
        print(f"尝试导出表格为 CSV 文件成功: {output_table_filename_csv}")
    except Exception as e_csv:
        print(f"导出表格到 CSV 时也出错: {e_csv}")
# --- 结束表格导出 ---

print("脚本执行完毕。")
