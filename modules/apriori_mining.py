"""
模块3：Apriori关联规则挖掘
- 发现用户购买商品之间的关联关系
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os

import config
from utils.visualization import setup_chinese_fonts, save_chart
setup_chinese_fonts()



def apriori_mining(df_orders, df_products):
    print("\n" + "=" * 60)
    print("模块3：Apriori关联规则挖掘")
    print("=" * 60)

    # 按订单聚合商品（同一用户同一天的购买视为一次交易）
    df_orders["date"] = pd.to_datetime(df_orders["timestamp"]).dt.date
    basket = df_orders.groupby(["user_id", "date"])["product_id"].apply(list).reset_index()
    transactions = basket["product_id"].tolist()

    # 过滤掉只有1件商品的交易
    transactions = [t for t in transactions if len(t) >= 2]
    print(f"有效交易数: {len(transactions)}")

    if len(transactions) < 10:
        print("交易数不足，使用商品品类替代进行关联分析")
        df_orders_full = df_orders.merge(df_products[["product_id", "category"]], on="product_id")
        basket = df_orders_full.groupby(["user_id", "date"])["category"].apply(list).reset_index()
        transactions = [t for t in basket["category"].tolist() if len(t) >= 2]
        item_col = "category"
        print(f"品类交易数: {len(transactions)}")
    else:
        item_col = "product_id"

    # TransactionEncoder
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_te = pd.DataFrame(te_array, columns=te.columns_)

    # Apriori
    frequent_itemsets = apriori(df_te, min_support=0.01, use_colnames=True)
    print(f"\n频繁项集数量: {len(frequent_itemsets)}")

    if len(frequent_itemsets) == 0:
        print("无频繁项集，降低min_support重试")
        frequent_itemsets = apriori(df_te, min_support=0.005, use_colnames=True)
        print(f"重试后频繁项集数量: {len(frequent_itemsets)}")

    # 关联规则
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
        rules = rules.sort_values("lift", ascending=False)
        print(f"\n关联规则数量: {len(rules)}")
        if len(rules) > 0:
            print("\nTop 10 关联规则:")
            display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
            available = [c for c in display_cols if c in rules.columns]
            print(rules[available].head(10).to_string())
    else:
        rules = pd.DataFrame()

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) 频繁项集支持度分布
    if len(frequent_itemsets) > 0:
        top_items = frequent_itemsets.nlargest(15, "support")
        item_labels = [str(list(s))[:20] for s in top_items["itemsets"]]
        axes[0].barh(range(len(top_items)), top_items["support"].values, color="steelblue")
        axes[0].set_yticks(range(len(top_items)))
        axes[0].set_yticklabels(item_labels, fontsize=8)
        axes[0].set_title("Top 15 频繁项集支持度")
        axes[0].invert_yaxis()
    else:
        axes[0].text(0.5, 0.5, "No frequent itemsets", ha="center")

    # (2) 关联规则散点图
    if len(rules) > 0:
        sample = rules.head(50)
        axes[1].scatter(sample["support"], sample["confidence"],
                        c=sample["lift"], cmap="YlOrRd", alpha=0.7, s=50)
        axes[1].set_xlabel("Support")
        axes[1].set_ylabel("Confidence")
        axes[1].set_title("关联规则 (颜色=Lift)")
        plt.colorbar(axes[1].collections[0], ax=axes[1], label="Lift")
    else:
        axes[1].text(0.5, 0.5, "No rules found", ha="center")

    # (3) Lift分布
    if len(rules) > 0:
        axes[2].hist(rules["lift"], bins=30, color="coral", alpha=0.7)
        axes[2].set_title("关联规则Lift值分布")
        axes[2].set_xlabel("Lift")
    else:
        axes[2].text(0.5, 0.5, "No rules found", ha="center")

    plt.tight_layout()
    plt.savefig(f"{config.config.OUTPUT_DIR}/03_apriori_rules.png", dpi=150)
    plt.close()
    print(f"\n✅ 图表已保存: {config.OUTPUT_DIR}/03_apriori_rules.png")

    return rules if len(rules) > 0 else pd.DataFrame()
