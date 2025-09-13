# 数据中心能耗优化算法研究

​        该项目代码中的数据集为"PUE数据汇总_processed.xlsx"，是对原始数据PUE数据汇总.xlsx的数据预处理后的数据集，用于整个项目的预测和优化，其中本项目用AdaBoost的能耗预测模型和LightGBM的温度预测模型一起用于ALF-PSO的参数粒子群优化，旨在找出比原始参数组合和能耗更低能耗的更优参数组合，AdaBoost的预测用于ALF-PSO的目标函数，LightGBM温度预测模型用于ALF-PSO优化途中的温度约束限制，如果调优后的变量预测温度值大于限定值，则会拒绝该参数组合。经过ALF-PSO算法不断地寻找，最终找出一组更优的解来对数据中心冷源群控系统的设备参数进行进一步的优化。

​        主要运行文件有两个，一个是对全数据集"PUE数据汇总_processed.xlsx"的预测优化，"ALF_PSO_Optimize.py"负责运行，它会保存预测模型（optimized_model_v2_ensemble.pkl）和参数表（optimized_meta_data_v2.json），以用于对单条数据进行预测。优化后导出文件为"ALF_PSO_调优后的参数_v2.xlsx"，该数据再通过运行" Red_signal.py"进行逐行标红，会得到"ALF-PSO__调优后的参数_v2标红.xlsx"，这是对优化后的数据和原始数据进行对比，会将优化项标记为红色，方便进行数据分析。

​		另一个是对单条数据进行的优化，数据集名称叫"测试用的一条数据.xlsx"，该数据里第一行是变量名，第二行是选取的一行数据，"One_data_Optimize.py"负责执行，优化后导出文件"单条数据优化结果.xlsx"，"One_data_Optimize.py"需要在"ALF_PSO_Optimize.py"运行后才能执行，因为"One_data_Optimize.py"必须包含ALF_PSO_Optimize.py生成的元数据组（"optimized_meta_data_v2.json"）和预测模型（"optimized_model_v2_ensemble.pkl"）。

​       其他python文件是阶段性的数据分析文件和过程性文件，一般用于数据分析。

​		整体运行次序就是先运行ALF_PSO_Optimize.py，保存预测模型和参数表，再运行One_data_Optimize.py对单条数据进行优化。
