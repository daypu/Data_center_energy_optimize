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
temperature_column = 'CWHR_2_t_0_5h'
correlation_with_energy = data.corr()[temperature_column].sort_values(ascending=False)

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
