"""
模块7：营销活动效果分析 + A/B测试
- 营销活动前后对比
- A/B测试统计检验
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import os

import config
from utils.visualization import setup_chinese_fonts, save_chart
setup_chinese_fonts()



def marketing_analysis(df_behavior, df_campaigns, df_orders):
    print("\n" + "=" * 60)
    print("模块7：营销活动效果分析 + A/B测试")
    print("=" * 60)

    df_behavior["timestamp"] = pd.to_datetime(df_behavior["timestamp"])
    df_campaigns["start_date"] = pd.to_datetime(df_campaigns["start_date"])
    df_campaigns["end_date"] = pd.to_datetime(df_campaigns["end_date"])

    campaign_results = []

    # ---- 每个营销活动前后对比 ----
    for _, camp in df_campaigns.iterrows():
        cid = camp["campaign_id"]
        name = camp["name"]
        start = camp["start_date"]
        end = camp["end_date"]
        period_days = (end - start).days

        # 活动期间
        during = df_behavior[(df_behavior["timestamp"] >= start) & (df_behavior["timestamp"] <= end)]
        # 活动前同期
        before = df_behavior[(df_behavior["timestamp"] >= start - pd.Timedelta(days=period_days)) &
                             (df_behavior["timestamp"] < start)]
        # 活动后同期
        after = df_behavior[(df_behavior["timestamp"] > end) &
                            (df_behavior["timestamp"] <= end + pd.Timedelta(days=period_days))]

        def calc_metrics(df):
            if len(df) == 0:
                return {"total": 0, "purchases": 0, "buyers": 0, "conversion": 0, "daily_avg": 0}
            purchases = (df["behavior"] == "购买").sum()
            buyers = df[df["behavior"] == "购买"]["user_id"].nunique()
            total_users = df["user_id"].nunique()
            days = max(1, (df["timestamp"].max() - df["timestamp"].min()).days + 1)
            return {
                "total": len(df),
                "purchases": purchases,
                "buyers": buyers,
                "conversion": purchases / len(df) * 100 if len(df) > 0 else 0,
                "daily_avg": purchases / days
            }

        m_before = calc_metrics(before)
        m_during = calc_metrics(during)
        m_after = calc_metrics(after)

        # 提升率
        if m_before["purchases"] > 0:
            lift = (m_during["purchases"] - m_before["purchases"]) / m_before["purchases"] * 100
        else:
            lift = 0

        result = {
            "campaign": name,
            "type": camp["type"],
            "group": camp["group"],
            "before_purchases": m_before["purchases"],
            "during_purchases": m_during["purchases"],
            "after_purchases": m_after["purchases"],
            "before_conversion": m_before["conversion"],
            "during_conversion": m_during["conversion"],
            "lift_pct": lift,
            "before_buyers": m_before["buyers"],
            "during_buyers": m_during["buyers"]
        }
        campaign_results.append(result)
        print(f"\n[{name}] 活动前→中 购买量: {m_before['purchases']}→{m_during['purchases']} "
              f"(提升 {lift:.1f}%), 转化率: {m_before['conversion']:.2f}%→{m_during['conversion']:.2f}%")

    df_camp_results = pd.DataFrame(campaign_results)

    # ---- A/B测试 ----
    print("\n" + "=" * 40)
    print("A/B测试分析")
    print("=" * 40)

    group_a = df_camp_results[df_camp_results["group"] == "A"]
    group_b = df_camp_results[df_camp_results["group"] == "B"]

    print(f"\nA组 (优惠券/满减): {len(group_a)} 个活动")
    print(f"B组 (折扣): {len(group_b)} 个活动")

    # 转化率t检验
    if len(group_a) > 1 and len(group_b) > 1:
        t_stat, p_value = stats.ttest_ind(group_a["during_conversion"], group_b["during_conversion"])
        print(f"\n转化率独立样本t检验:")
        print(f"  A组平均转化率: {group_a['during_conversion'].mean():.2f}%")
        print(f"  B组平均转化率: {group_b['during_conversion'].mean():.2f}%")
        print(f"  t统计量: {t_stat:.4f}")
        print(f"  p值: {p_value:.4f}")
        print(f"  {'统计显著' if p_value < 0.05 else '统计不显著'} (α=0.05)")

        best_group = "B" if group_b["during_conversion"].mean() > group_a["during_conversion"].mean() else "A"
        print(f"\n✅ 推荐策略: {'折扣策略' if best_group == 'B' else '优惠券/满减策略'}")
    else:
        # 单组对比
        a_lift = group_a["lift_pct"].mean() if len(group_a) > 0 else 0
        b_lift = group_b["lift_pct"].mean() if len(group_b) > 0 else 0
        print(f"  A组平均购买提升: {a_lift:.1f}%")
        print(f"  B组平均购买提升: {b_lift:.1f}%")
        best_group = "B" if b_lift > a_lift else "A"
        t_stat, p_value = 0, 1
        print(f"\n✅ 推荐策略: {'折扣策略' if best_group == 'B' else '优惠券/满减策略'}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (1) 各活动购买量前后对比
    x = range(len(df_camp_results))
    axes[0, 0].bar([i - 0.2 for i in x], df_camp_results["before_purchases"], 0.35, label="活动前", color="gray")
    axes[0, 0].bar([i + 0.2 for i in x], df_camp_results["during_purchases"], 0.35, label="活动中", color="coral")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(df_camp_results["campaign"], rotation=30, fontsize=8)
    axes[0, 0].set_title("各营销活动购买量对比")
    axes[0, 0].legend()

    # (2) 购买提升率
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in df_camp_results["lift_pct"]]
    axes[0, 1].barh(df_camp_results["campaign"], df_camp_results["lift_pct"], color=colors)
    axes[0, 1].set_title("各活动购买提升率(%)")
    axes[0, 1].axvline(x=0, color="black", linewidth=0.5)
    axes[0, 1].set_xlabel("提升率(%)")

    # (3) A/B组对比
    if len(group_a) > 0 and len(group_b) > 0:
        ab_data = pd.DataFrame({
            "Group": ["A (优惠券/满减)", "B (折扣)"],
            "Avg Conversion": [group_a["during_conversion"].mean(), group_b["during_conversion"].mean()],
            "Avg Lift": [group_a["lift_pct"].mean(), group_b["lift_pct"].mean()]
        })
        x_ab = np.arange(2)
        axes[1, 0].bar(x_ab - 0.2, ab_data["Avg Conversion"], 0.35, label="转化率(%)", color="steelblue")
        ax_twin = axes[1, 0].twinx()
        ax_twin.bar(x_ab + 0.2, ab_data["Avg Lift"], 0.35, label="购买提升(%)", color="coral", alpha=0.7)
        axes[1, 0].set_xticks(x_ab)
        axes[1, 0].set_xticklabels(ab_data["Group"])
        axes[1, 0].set_ylabel("转化率(%)", color="steelblue")
        ax_twin.set_ylabel("购买提升(%)", color="coral")
        axes[1, 0].set_title(f"A/B测试对比 (p={p_value:.4f})")
        axes[1, 0].legend(loc="upper left", fontsize=8)
        ax_twin.legend(loc="upper right", fontsize=8)

    # (4) 活动类型效果
    type_effect = df_camp_results.groupby("type")["lift_pct"].mean().sort_values(ascending=True)
    axes[1, 1].barh(type_effect.index, type_effect.values, color=sns.color_palette("Set2", len(type_effect)))
    axes[1, 1].set_title("不同营销类型平均购买提升率(%)")
    axes[1, 1].set_xlabel("提升率(%)")

    plt.tight_layout()
    plt.savefig(f"{config.config.OUTPUT_DIR}/07_marketing_ab.png", dpi=150)
    plt.close()
    print(f"\n✅ 图表已保存: {config.OUTPUT_DIR}/07_marketing_ab.png")

    return df_camp_results, best_group
