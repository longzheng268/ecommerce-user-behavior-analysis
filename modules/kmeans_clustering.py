"""
模块4：K-means用户聚类分析
- 基于RFM + 行为特征进行用户聚类
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUTPUT_DIR = "output"


def kmeans_clustering(df_users, df_behavior, df_orders, user_activity, rfm):
    print("\n" + "=" * 60)
    print("模块4：K-means用户聚类分析")
    print("=" * 60)

    # 合并用户特征
    user_features = rfm.merge(user_activity[["user_id", "total_actions", "unique_products", "avg_duration"]],
                               on="user_id", how="left")
    user_features = user_features.merge(df_users[["user_id", "age", "vip_level"]], on="user_id", how="left")
    user_features = user_features.fillna(0)

    # 特征列
    feature_cols = ["recency", "frequency", "monetary", "total_actions", "unique_products", "avg_duration", "age", "vip_level"]
    X = user_features[feature_cols].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 肘部法则 + 轮廓系数
    inertias = []
    sil_scores = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    best_k = list(K_range)[np.argmax(sil_scores)]
    print(f"\n最佳聚类数 K = {best_k} (轮廓系数: {max(sil_scores):.4f})")

    # 最终聚类
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    user_features["cluster"] = km_final.fit_predict(X_scaled)

    # 聚类中心分析
    cluster_summary = user_features.groupby("cluster")[feature_cols].mean().round(2)
    print("\n[聚类中心特征]")
    print(cluster_summary.to_string())

    # 聚类标签命名
    cluster_names = {}
    for c in range(best_k):
        row = cluster_summary.loc[c]
        if row["frequency"] > cluster_summary["frequency"].mean() and row["monetary"] > cluster_summary["monetary"].mean():
            cluster_names[c] = "高价值活跃用户"
        elif row["recency"] > cluster_summary["recency"].mean() * 1.5:
            cluster_names[c] = "流失风险用户"
        elif row["total_actions"] > cluster_summary["total_actions"].mean():
            cluster_names[c] = "高频浏览用户"
        else:
            cluster_names[c] = f"普通用户群{c}"

    user_features["cluster_name"] = user_features["cluster"].map(cluster_names)
    print("\n[聚类命名]")
    for k, v in cluster_names.items():
        count = (user_features["cluster"] == k).sum()
        print(f"  簇{k}: {v} ({count}人)")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) 肘部法则
    axes[0, 0].plot(K_range, inertias, "bo-")
    axes[0, 0].set_title("肘部法则")
    axes[0, 0].set_xlabel("K")
    axes[0, 0].set_ylabel("Inertia")

    # (2) 轮廓系数
    axes[0, 1].plot(K_range, sil_scores, "ro-")
    axes[0, 1].set_title("轮廓系数")
    axes[0, 1].set_xlabel("K")
    axes[0, 1].set_ylabel("Silhouette Score")

    # (3) PCA 2D散点
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    scatter = axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1],
                                  c=user_features["cluster"], cmap="Set2", alpha=0.5, s=10)
    axes[0, 2].set_title("PCA降维聚类结果")
    axes[0, 2].set_xlabel("PC1")
    axes[0, 2].set_ylabel("PC2")

    # (4) 各簇人数
    cluster_counts = user_features["cluster_name"].value_counts()
    axes[1, 0].barh(cluster_counts.index, cluster_counts.values, color=sns.color_palette("Set2", best_k))
    axes[1, 0].set_title("各用户群人数")

    # (5) 雷达图 - 各簇特征对比
    categories_radar = ["recency", "frequency", "monetary", "total_actions", "unique_products"]
    normalized = cluster_summary[categories_radar]
    normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-8)
    angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=False).tolist()
    angles += angles[:1]
    ax_radar = fig.add_subplot(2, 3, 5, polar=True)
    axes[1, 1].set_visible(False)
    for c in range(best_k):
        values = normalized.loc[c].tolist()
        values += values[:1]
        ax_radar.plot(angles, values, "o-", label=cluster_names[c], linewidth=1.5)
        ax_radar.fill(angles, values, alpha=0.1)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(["R", "F", "M", "行为量", "商品数"], fontsize=8)
    ax_radar.set_title("用户群特征雷达图", pad=20)
    ax_radar.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1))

    # (6) 各簇平均消费
    cluster_monetary = user_features.groupby("cluster_name")["monetary"].mean().sort_values()
    axes[1, 2].barh(cluster_monetary.index, cluster_monetary.values, color="coral")
    axes[1, 2].set_title("各用户群平均消费金额")
    axes[1, 2].set_xlabel("平均消费(元)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_kmeans_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 图表已保存: {OUTPUT_DIR}/04_kmeans_clustering.png")

    return user_features, cluster_summary, cluster_names
