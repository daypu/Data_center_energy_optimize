import pandas as pd
import numpy as np

# 原有代码保持不变
file_path = 'PUE数据汇总.xlsx'
excel_data = pd.ExcelFile(file_path)
data = excel_data.parse('Sheet1')

# 温度数据处理部分保持不变
temperature_data = data['冷冻侧回水环网温度TIT_6_CWHR_1_温度计_实际值'].dropna().values

energy_min = 700
energy_max = 1000
temp_min, temp_max = np.min(temperature_data), np.max(temperature_data)

m = (energy_max - energy_min) / (temp_max - temp_min)
b = energy_min - m * temp_min
print(f"斜率 (m): {m}")
print(f"截距 (b): {b}")

new_energy_consumption = m * temperature_data + b

# 添加噪声部分保持不变
mean = 0
std_dev = 20
gaussian_noise = np.random.normal(mean, std_dev, size=new_energy_consumption.shape)
new_energy_consumption_with_noise = new_energy_consumption + gaussian_noise

correlation = np.corrcoef(new_energy_consumption_with_noise, temperature_data)[0, 1]
print(f"生成的能耗数据与温度数据的相关系数: {correlation}")

# 新增EIT数据加载和处理
try:
    # 尝试从现有列加载EIT数据
    eit_series = data['Eit实时能耗_单位_kW']
except KeyError:
    # 如果列名不同，这里可以添加其他处理逻辑
    raise ValueError("未找到Eit实时能耗数据列，请确认列名是否正确")

eit_data = eit_series.dropna().values

# 数据一致性校验
if len(new_energy_consumption_with_noise) != len(eit_data):
    raise ValueError(f"数据长度不匹配：冷源能耗长度={len(new_energy_consumption_with_noise)}，EIT长度={len(eit_data)}")

# 添加PUE计算逻辑
def calculate_pue(energy, eit):
    """计算PUE值，包含除零保护和异常处理"""
    with np.errstate(divide='raise'):
        pue = (energy + eit) / eit
    return pue

# 转换为numpy数组进行计算
try:
    pue_values = calculate_pue(new_energy_consumption_with_noise, eit_data)
except FloatingPointError as e:
    print(f"计算错误：{e}")
    # 使用np.array_like替代np_like生成NaN数组
    pue_values = np.full_like(new_energy_consumption_with_noise, np.nan)  # **修正点**

# 添加PUE列到DataFrame
data['PUE值'] = pd.Series(pue_values, index=data.index)

# 数据保存前校验
invalid_pue = data[data['PUE值'] < 1]  # PUE理论上应大于1
if not invalid_pue.empty:
    print(f"警告：检测到{len(invalid_pue)}个无效的PUE值（<1），建议检查数据：")
    print(invalid_pue[['冷源群控系统实时能耗_单位_kW', 'Eit实时能耗_单位_kW', 'PUE值']].head())

# 保存更新后的数据
data.to_excel('PUE数据汇总.xlsx', index=False)
print("数据更新完成，已包含PUE值列")

# 显示关键结果
print("\n示例数据验证：")
print(data[['冷冻侧回水环网温度TIT_6_CWHR_1_温度计_实际值',
          '冷源群控系统实时能耗_单位_kW',
          'Eit实时能耗_单位_kW',
          'PUE值']].head(5))