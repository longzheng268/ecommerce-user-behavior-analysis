# 基于数据挖掘的电商平台用户行为分析

E-commerce User Behavior Analysis with Data Mining

## 项目简介

本项目是数据科学与大数据本科专业（2021级）数据挖掘综合训练项目，基于Python对电商平台用户行为数据进行全面分析，涵盖数据清洗、统计分析、关联规则挖掘、用户聚类、时间序列预测、机器学习预测和营销效果评估等模块。

## 项目结构

```
├── generate_data.py              # 模拟数据生成
├── main.py                       # 主运行脚本（一键执行全部分析）
├── modules/
│   ├── data_cleaning.py          # 模块1：数据清洗与预处理
│   ├── statistical_analysis.py   # 模块2：统计分析（频率/分布/趋势）
│   ├── apriori_mining.py         # 模块3：Apriori关联规则挖掘
│   ├── kmeans_clustering.py      # 模块4：K-means用户聚类分析
│   ├── arima_forecast.py         # 模块5：ARIMA时间序列预测
│   ├── ml_prediction.py          # 模块6：随机森林 + 神经网络预测
│   └── marketing_analysis.py     # 模块7：营销活动效果分析 + A/B测试
├── data/                         # 数据文件
│   ├── users.csv                 # 用户表（2000条）
│   ├── products.csv              # 商品表（500条）
│   ├── behavior.csv              # 行为日志（20万条）
│   ├── orders.csv                # 订单表（1万条）
│   ├── campaigns.csv             # 营销活动（5个）
│   └── interaction_matrix.csv    # 用户-商品交互矩阵
└── output/                       # 分析结果
    ├── 01_data_cleaning.png      # 数据清洗可视化
    ├── 02_statistical_analysis.png # 统计分析图表
    ├── 03_apriori_rules.png      # Apriori关联规则
    ├── 04_kmeans_clustering.png  # K-means聚类结果
    ├── 05_arima_forecast.png     # ARIMA时序预测
    ├── 06_ml_prediction.png      # 机器学习预测结果
    ├── 07_marketing_ab.png       # 营销活动+A/B测试
    └── analysis_report.md        # 汇总分析报告
```

## 技术栈

- **语言**: Python 3.11+
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **机器学习**: Scikit-learn (Random Forest, MLP, K-Means)
- **时间序列**: Statsmodels (ARIMA)
- **关联规则**: MLxtend (Apriori)

## 快速开始

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels mlxtend

# 3. 一键运行全部分析
python main.py
```

## 模块说明

### 模块1：数据清洗与预处理
- 处理缺失值（中位数填充、众数填充、删除）
- 处理异常值（年龄<0、会话时长<0）
- 统一数据格式（日期、类别编码）
- 构建用户-商品交互矩阵 (2000×500)

### 模块2：统计分析
- 行为类型频率分布（浏览/搜索/加购/收藏/购买）
- 24小时行为热力分布
- 周内行为趋势
- RFM用户价值分析
- 品类转化率分析

### 模块3：Apriori关联规则挖掘
- 基于用户购买事务构建关联规则
- 计算支持度、置信度、Lift值
- 发现商品捆绑销售机会

### 模块4：K-means用户聚类
- 肘部法则 + 轮廓系数确定最优K值
- PCA降维可视化
- 用户群体特征雷达图
- 聚类命名与业务解读

### 模块5：ARIMA时间序列预测
- ADF平稳性检验
- 自动参数搜索 (p, d, q)
- 未来30天购买趋势预测
- 残差诊断（ACF/PACF）

### 模块6：机器学习预测
- 随机森林分类器（特征重要性分析）
- MLP神经网络（多层感知机）
- ROC曲线、混淆矩阵、AUC评估
- 模型性能对比

### 模块7：营销活动效果分析
- 活动前后购买量/转化率对比
- A/B测试统计检验（独立样本t检验）
- 不同营销策略效果对比

## 关键发现

| 指标 | 结果 |
|---|---|
| 浏览→购买转化率 | 11.30% |
| K-means最优聚类数 | 2类用户群 |
| ARIMA模型 | ARIMA(2,0,2), RMSE=7.60 |
| 随机森林 AUC | 0.6245 |
| 神经网络 AUC | 0.6237 |
| 618大促购买提升 | 155.2% |
| 双11购买提升 | 124.4% |
| 推荐营销策略 | 折扣策略 |

## 数据安全与伦理说明

本项目使用模拟数据，不涉及真实用户隐私。在实际电商场景中，需注意：
- 用户数据脱敏与匿名化
- 遵守《个人信息保护法》
- 数据访问权限最小化原则
- 算法公平性与透明度

## 参考文献

[1] 卢滔等. Python数据挖掘入门进阶与实用案例分析[M]. 机械工业出版社, 2023.
[2] 王磊，邱江涛. Python数据挖掘实战[M]. 人民邮电出版社, 2023.
[3] Pang-Ning Tan, Michael Steinbach, Vipin Kumar. 数据挖掘导论[M]. 人民邮电出版社, 2019.
[4] Jiawei Han, Micheline Kamber. 数据挖掘：概念与技术[M]. 机械工业出版社, 2021.
[5] Sebastian Raschka. Python机器学习[M]. 机械工业出版社, 2022.

## License

MIT License
