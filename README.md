<div align="center">

# 🛒 基于数据挖掘的电商平台用户行为分析

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*数据科学与大数据 · 数据挖掘综合训练*

</div>

---

## 📊 功能模块

| 模块 | 算法 | 说明 |
|------|------|------|
| 数据清洗 | — | 缺失值/异常值处理，交互矩阵构建 |
| 统计分析 | RFM | 行为分布、时段趋势、品类转化率 |
| 关联规则 | Apriori | 商品关联关系挖掘（144条规则） |
| 用户聚类 | K-means | 用户分群 + 雷达图可视化 |
| 趋势预测 | ARIMA | 30天购买量预测（RMSE=7.60） |
| 购买预测 | RF + MLP | 随机森林 / 神经网络（AUC≈0.62） |
| 营销评估 | A/B Test | 活动效果对比 + 统计检验 |
| 可视化大屏 | ECharts | 暗色主题交互式数据看板 |

---

## 🚀 快速开始

```bash
git clone https://github.com/longzheng268/ecommerce-user-behavior-analysis.git
cd ecommerce-user-behavior-analysis

python -m venv venv
source venv/bin/activate   # Windows 用 venv\Scripts\activate

pip install -r requirements.txt

python main.py    # 命令行分析 → output/
python app.py     # Web看板 → http://localhost:5000
````

---

## 🧱 架构设计

```
app.py                  # Flask入口
config.py               # 配置管理

api/routes.py           # REST API（10个端点）

modules/
├── data_cleaning.py
├── statistical_analysis.py
├── apriori_mining.py
├── kmeans_clustering.py
├── arima_forecast.py
├── ml_prediction.py
└── marketing_analysis.py

utils/
├── data_loader.py
└── visualization.py

templates/             # 前端页面
static/                # CSS/JS
data/                  # 原始数据
output/                # 输出结果
docs/                  # 文档
```

---

## 📈 核心结果

| 指标       | 值                 |
| -------- | ----------------- |
| 浏览→购买转化率 | 11.30%            |
| 关联规则数量   | 144               |
| 用户分群     | K=2（高价值 / 普通）     |
| ARIMA模型  | (2,0,2)，RMSE=7.60 |
| 618大促提升  | +155.2%           |
| 推荐策略     | 折扣优先于优惠券          |

---

## 📚 文档

* [综合训练说明书](docs/training_report.md)
* [分析汇总报告](output/analysis_report.md)

---

## 📜 License

MIT License · © longzheng268
