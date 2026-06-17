"""
模块1：数据清洗与预处理
- 处理缺失值、异常值
- 统一数据格式
- 构建用户-商品交互矩阵
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import config
from utils.visualization import setup_chinese_fonts, save_chart
setup_chinese_fonts()




def load_and_clean():
    """加载原始数据并执行清洗"""
    print("=" * 60)
    print("模块1：数据清洗与预处理")
    print("=" * 60)

    # ---- 加载 ----
    df_users = pd.read_csv("data/users.csv")
    df_products = pd.read_csv("data/products.csv")
    df_behavior = pd.read_csv("data/behavior.csv")
    df_orders = pd.read_csv("data/orders.csv")
    df_campaigns = pd.read_csv("data/campaigns.csv")

    stats = {"原始数据": {}}
    for name, df in [("users", df_users), ("products", df_products),
                     ("behavior", df_behavior), ("orders", df_orders)]:
        stats["原始数据"][name] = len(df)

    # ========== 用户表清洗 ==========
    print("\n[用户表] 缺失值统计：")
    print(df_users.isnull().sum())

    # 年龄缺失 -> 中位数填充
    median_age = df_users.loc[df_users["age"] > 0, "age"].median()
    df_users.loc[df_users["age"].isnull(), "age"] = median_age
    # 年龄异常值(<0 或 >100) -> clip
    df_users["age"] = df_users["age"].clip(18, 65).astype(int)
    # 城市空值 -> "未知"
    df_users["city"] = df_users["city"].replace("", "未知").fillna("未知")
    # 性别缺失 -> "未知"
    df_users["gender"] = df_users["gender"].fillna("未知")
    # 日期格式统一
    df_users["register_date"] = pd.to_datetime(df_users["register_date"], errors="coerce")

    print(f"\n[用户表] 清洗后：{len(df_users)} 行, 缺失: {df_users.isnull().sum().sum()}")

    # ========== 行为日志清洗 ==========
    print("\n[行为日志] 缺失值统计：")
    print(df_behavior.isnull().sum())

    # 时间戳缺失 -> 删除（无法恢复）
    df_behavior = df_behavior.dropna(subset=["timestamp"])
    df_behavior["timestamp"] = pd.to_datetime(df_behavior["timestamp"], errors="coerce")
    df_behavior = df_behavior.dropna(subset=["timestamp"])

    # 会话时长异常(<0) -> 取绝对值
    df_behavior.loc[df_behavior["session_duration"] < 0, "session_duration"] = \
        df_behavior.loc[df_behavior["session_duration"] < 0, "session_duration"].abs()

    # 设备缺失
    df_behavior["device"] = df_behavior["device"].fillna("未知")

    print(f"\n[行为日志] 清洗后：{len(df_behavior)} 行")

    # ========== 订单表清洗 ==========
    print("\n[订单表] 缺失值统计：")
    print(df_orders.isnull().sum())

    # 金额缺失 -> 用单价*数量重算
    if "price" not in df_orders.columns:
        df_orders = df_orders.merge(df_products[["product_id", "price"]], on="product_id", how="left")
    mask = df_orders["total_amount"].isnull()
    df_orders.loc[mask, "total_amount"] = (df_orders.loc[mask, "quantity"] * df_orders.loc[mask, "price"]).round(2)

    print(f"\n[订单表] 清洗后：{len(df_orders)} 行")

    # ========== 构建用户-商品交互矩阵 ==========
    print("\n构建用户-商品交互矩阵...")
    interaction = df_behavior.groupby(["user_id", "product_id"]).size().reset_index(name="interaction_count")
    interaction_matrix = interaction.pivot_table(
        index="user_id", columns="product_id", values="interaction_count", fill_value=0
    )
    print(f"交互矩阵大小: {interaction_matrix.shape}")
    interaction_matrix.to_csv("data/interaction_matrix.csv")

    # ========== 可视化：清洗前后对比 ==========
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 年龄分布
    axes[0].hist(df_users["age"], bins=30, color="steelblue", alpha=0.7, edgecolor="white")
    axes[0].set_title("用户年龄分布（清洗后）")
    axes[0].set_xlabel("年龄")
    axes[0].set_ylabel("人数")

    # 行为类型分布
    behavior_counts = df_behavior["behavior"].value_counts()
    axes[1].bar(behavior_counts.index, behavior_counts.values, color=sns.color_palette("Set2"))
    axes[1].set_title("用户行为类型分布")
    axes[1].set_ylabel("次数")

    # 每日行为趋势
    daily = df_behavior.set_index("timestamp").resample("D").size()
    axes[2].plot(daily.index, daily.values, linewidth=0.8, color="coral")
    axes[2].set_title("每日行为总量趋势")
    axes[2].set_xlabel("日期")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{config.config.OUTPUT_DIR}/01_data_cleaning.png", dpi=150)
    plt.close()
    print(f"\n✅ 图表已保存: {config.OUTPUT_DIR}/01_data_cleaning.png")

    return df_users, df_products, df_behavior, df_orders, df_campaigns, interaction_matrix


if __name__ == "__main__":
    load_and_clean()
