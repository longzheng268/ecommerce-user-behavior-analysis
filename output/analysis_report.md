
# 电商用户行为数据分析报告

## 1. 数据概览
- 用户数: 2000
- 商品数: 500
- 行为记录: 199800
- 订单数: 10000
- 营销活动: 5

## 2. 统计分析关键发现
- 浏览→购买转化率: 11.30%
- 用户行为分布: 浏览占比最高，购买占比最低

## 3. 关联规则挖掘
- 发现 144 条关联规则
- 可用于商品捆绑销售和推荐系统

## 4. 用户聚类分析
- 最优聚类数: 2
- 用户群体: 高价值活跃用户, 普通用户群1

## 5. 购买趋势预测 (ARIMA(2, 0, 2))
- 模型RMSE: 7.60
- 模型MAE: 5.39
- 预测未来30天购买趋势

## 6. 机器学习预测
- 随机森林准确率: 0.6626, AUC: 0.6245
- 神经网络准确率: 0.6644, AUC: 0.6237

## 7. 营销活动效果
- 推荐策略: B
- 详见各活动对比图表

## 8. 生成图表清单
1. output/01_data_cleaning.png - 数据清洗可视化
2. output/02_statistical_analysis.png - 统计分析图表
3. output/03_apriori_rules.png - Apriori关联规则
4. output/04_kmeans_clustering.png - K-means聚类分析
5. output/05_arima_forecast.png - ARIMA时间序列预测
6. output/06_ml_prediction.png - 机器学习预测结果
7. output/07_marketing_ab.png - 营销活动+A/B测试分析
