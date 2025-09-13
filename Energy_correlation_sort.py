import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取用户上传的文件
file_path = 'PUE数据汇总.xlsx'
data = pd.read_excel(file_path)

# 设置Matplotlib的字体为SimHei（黑体），以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 提取能耗数据列
temperature_column = '冷源群控系统实时能耗_单位_kW'
correlation_with_energy = data.corr()[temperature_column].sort_values(ascending=False)

# 获取排序后的列名
sorted_columns = correlation_with_energy.drop(temperature_column).index

# 根据相关系数排序每列的数据
sorted_data = data[sorted_columns]

# 获取相关系数并添加到排序后的数据下方
correlation_values = correlation_with_energy[sorted_columns]
correlation_row = pd.DataFrame(correlation_values).T  # 将相关系数转化为行

# 将相关系数行添加到数据的末尾
sorted_data_with_correlation = pd.concat([sorted_data, correlation_row], ignore_index=True)

# 将排序后的数据和相关系数保存到新的Excel文件
sorted_data_with_correlation.to_excel('排序后的能耗数据.xlsx', index=False)

# 绘制与能耗数据的相关性条形图
plt.figure(figsize=(10, 6))
bars = correlation_with_energy.drop(temperature_column).plot(kind='bar', color='lightblue')
plt.title(f"各个特征与 {temperature_column} 的相关性")
plt.ylabel("相关系数")
plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，防止重叠

# 在每个柱子上方标记相关系数
for patch in bars.patches:
    height = patch.get_height()
    plt.text(patch.get_x() + patch.get_width() / 2, height + 0.02, f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()  # 自动调整子图参数，避免标签被遮挡
plt.show()
