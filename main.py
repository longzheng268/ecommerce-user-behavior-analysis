#!/usr/bin/env python3
"""
电商用户行为数据分析 - 主运行脚本
依次运行所有分析模块，生成完整报告
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_cleaning import load_and_clean
from modules.statistical_analysis import statistical_analysis
from modules.apriori_mining import apriori_mining
from modules.kmeans_clustering import kmeans_clustering
from modules.arima_forecast import arima_forecast
from modules.ml_prediction import ml_prediction
from modules.marketing_analysis import marketing_analysis

import pandas as pd
import numpy as np


def main():
    start_time = time.time()
    print("🚀 电商用户行为数据分析系统")
    print("=" * 60)

    # 确保数据存在
    if not os.path.exists("data/users.csv"):
        print("生成模拟数据...")
        import generate_data

    # ========== 模块1: 数据清洗 ==========
    df_users, df_products, df_behavior, df_orders, df_campaigns, interaction_matrix = load_and_clean()

    # ========== 模块2: 统计分析 ==========
    user_activity, rfm, stat_results = statistical_analysis(df_users, df_behavior, df_orders, df_products)

    # ========== 模块3: Apriori关联规则 ==========
    rules = apriori_mining(df_orders, df_products)

    # ========== 模块4: K-means聚类 ==========
    user_features, cluster_summary, cluster_names = kmeans_clustering(
        df_users, df_behavior, df_orders, user_activity, rfm
    )

    # ========== 模块5: ARIMA时序预测 ==========
    forecast, forecast_ci, arima_metrics = arima_forecast(df_behavior, df_orders)

    # ========== 模块6: 机器学习预测 ==========
    ml_results = ml_prediction(df_behavior, df_users, df_products)

    # ========== 模块7: 营销活动分析 + A/B测试 ==========
    camp_results, best_strategy = marketing_analysis(df_behavior, df_campaigns, df_orders)

    # ========== 生成汇总报告 ==========
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 全部分析完成！ 耗时: {elapsed:.1f}秒")
    print("=" * 60)

    report = f"""
# 电商用户行为数据分析报告

## 1. 数据概览
- 用户数: {len(df_users)}
- 商品数: {len(df_products)}
- 行为记录: {len(df_behavior)}
- 订单数: {len(df_orders)}
- 营销活动: {len(df_campaigns)}

## 2. 统计分析关键发现
- 浏览→购买转化率: {stat_results.get('conversion_rate', 0):.2f}%
- 用户行为分布: 浏览占比最高，购买占比最低

## 3. 关联规则挖掘
- 发现 {len(rules) if rules is not None and len(rules) > 0 else 0} 条关联规则
- 可用于商品捆绑销售和推荐系统

## 4. 用户聚类分析
- 最优聚类数: {len(cluster_names)}
- 用户群体: {', '.join(cluster_names.values())}

## 5. 购买趋势预测 (ARIMA{arima_metrics['order']})
- 模型RMSE: {arima_metrics['rmse']:.2f}
- 模型MAE: {arima_metrics['mae']:.2f}
- 预测未来30天购买趋势

## 6. 机器学习预测
- 随机森林准确率: {ml_results['rf']['accuracy']:.4f}, AUC: {ml_results['rf']['auc']:.4f}
- 神经网络准确率: {ml_results['mlp']['accuracy']:.4f}, AUC: {ml_results['mlp']['auc']:.4f}

## 7. 营销活动效果
- 推荐策略: {best_strategy}
- 详见各活动对比图表

## 8. 生成图表清单
1. output/01_data_cleaning.png - 数据清洗可视化
2. output/02_statistical_analysis.png - 统计分析图表
3. output/03_apriori_rules.png - Apriori关联规则
4. output/04_kmeans_clustering.png - K-means聚类分析
5. output/05_arima_forecast.png - ARIMA时间序列预测
6. output/06_ml_prediction.png - 机器学习预测结果
7. output/07_marketing_ab.png - 营销活动+A/B测试分析
"""

    with open("output/analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📄 报告已保存: output/analysis_report.md")


if __name__ == "__main__":
    main()
