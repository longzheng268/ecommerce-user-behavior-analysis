"""
模块2：统计分析
- 用户行为频率、分布、趋势
- 用户画像分析
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUTPUT_DIR = "output"


def statistical_analysis(df_users, df_behavior, df_orders, df_products):
    print("\n" + "=" * 60)
    print("模块2：统计分析")
    print("=" * 60)

    results = {}

    # ---- 1. 行为频率统计 ----
    behavior_freq = df_behavior["behavior"].value_counts()
    print("\n[行为频率]")
    print(behavior_freq)
    results["behavior_freq"] = behavior_freq.to_dict()

    # 购买转化率
    total_views = len(df_behavior[df_behavior["behavior"] == "浏览"])
    total_purchases = len(df_behavior[df_behavior["behavior"] == "购买"])
    conversion_rate = total_purchases / total_views * 100 if total_views > 0 else 0
    print(f"\n浏览→购买转化率: {conversion_rate:.2f}%")
    results["conversion_rate"] = conversion_rate

    # ---- 2. 用户活跃度分析 ----
    user_activity = df_behavior.groupby("user_id").agg(
        total_actions=("behavior", "count"),
        unique_products=("product_id", "nunique"),
        purchase_count=("behavior", lambda x: (x == "购买").sum()),
        avg_duration=("session_duration", "mean")
    ).reset_index()

    print(f"\n[用户活跃度] 人均行为次数: {user_activity['total_actions'].mean():.1f}")
    print(f"  人均购买次数: {user_activity['purchase_count'].mean():.2f}")
    print(f"  人均浏览商品数: {user_activity['unique_products'].mean():.1f}")

    # ---- 3. 时段分布 ----
    df_behavior["hour"] = pd.to_datetime(df_behavior["timestamp"]).dt.hour
    hourly = df_behavior.groupby("hour").size()

    # ---- 4. 品类分析 ----
    category_behavior = df_behavior.merge(df_products[["product_id", "category"]], on="product_id")
    cat_stats = category_behavior.groupby("category").agg(
        total_views=("behavior", lambda x: (x == "浏览").sum()),
        total_purchases=("behavior", lambda x: (x == "购买").sum())
    )
    cat_stats["conversion"] = (cat_stats["total_purchases"] / cat_stats["total_views"] * 100).round(2)

    # ---- 5. 周趋势 ----
    df_behavior["weekday"] = pd.to_datetime(df_behavior["timestamp"]).dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_counts = df_behavior.groupby("weekday").size().reindex(weekday_order)

    # ---- 6. RFM分析 ----
    ref_date = pd.to_datetime(df_behavior["timestamp"]).max()
    rfm = df_behavior[df_behavior["behavior"] == "购买"].groupby("user_id").agg(
        recency=("timestamp", lambda x: (ref_date - pd.to_datetime(x)).min().days),
        frequency=("behavior", "count"),
    ).reset_index()

    orders_by_user = df_orders.groupby("user_id")["total_amount"].sum().reset_index(name="monetary")
    rfm = rfm.merge(orders_by_user, on="user_id", how="left").fillna(0)

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) 行为分布饼图
    axes[0, 0].pie(behavior_freq.values, labels=behavior_freq.index,
                   autopct="%1.1f%%", colors=sns.color_palette("Set2"))
    axes[0, 0].set_title("用户行为类型分布")

    # (2) 时段分布
    axes[0, 1].bar(hourly.index, hourly.values, color="steelblue", alpha=0.8)
    axes[0, 1].set_title("24小时行为分布")
    axes[0, 1].set_xlabel("小时")
    axes[0, 1].set_ylabel("行为次数")

    # (3) 品类转化率
    top_cats = cat_stats.sort_values("conversion", ascending=True)
    axes[0, 2].barh(top_cats.index, top_cats["conversion"], color="coral")
    axes[0, 2].set_title("各品类浏览→购买转化率(%)")
    axes[0, 2].set_xlabel("转化率(%)")

    # (4) 周趋势
    axes[1, 0].bar(range(7), weekday_counts.values, color="mediumpurple")
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=30)
    axes[1, 0].set_title("每周各天行为量")

    # (5) 用户活跃度分布
    axes[1, 1].hist(user_activity["total_actions"], bins=50, color="seagreen", alpha=0.7)
    axes[1, 1].set_title("用户行为次数分布")
    axes[1, 1].set_xlabel("行为次数")
    axes[1, 1].set_xlim(0, user_activity["total_actions"].quantile(0.95))

    # (6) RFM - R vs F散点
    sample = rfm.sample(min(500, len(rfm)), random_state=42)
    axes[1, 2].scatter(sample["recency"], sample["frequency"], c=sample["monetary"],
                       cmap="YlOrRd", alpha=0.6, s=20)
    axes[1, 2].set_xlabel("Recency (天)")
    axes[1, 2].set_ylabel("Frequency (次)")
    axes[1, 2].set_title("RFM分析 (颜色=Monetary)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_statistical_analysis.png", dpi=150)
    plt.close()
    print(f"\n✅ 图表已保存: {OUTPUT_DIR}/02_statistical_analysis.png")

    return user_activity, rfm, results
