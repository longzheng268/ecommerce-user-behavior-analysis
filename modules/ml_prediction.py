"""
模块6：机器学习预测
- 随机森林预测用户购买行为
- 神经网络预测用户购买行为
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings("ignore")
import os

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUTPUT_DIR = "output"


def build_features(df_behavior, df_users, df_products):
    """构建用户-商品对特征，标签=是否购买"""
    # 用户特征
    user_feats = df_behavior.groupby("user_id").agg(
        total_actions=("behavior", "count"),
        n_products=("product_id", "nunique"),
        n_purchases=("behavior", lambda x: (x == "购买").sum()),
        avg_duration=("session_duration", "mean"),
        n_devices=("device", "nunique")
    ).reset_index()

    user_feats = user_feats.merge(df_users[["user_id", "age", "gender", "vip_level"]], on="user_id", how="left")
    user_feats["gender"] = LabelEncoder().fit_transform(user_feats["gender"].fillna("未知"))
    user_feats["age"] = user_feats["age"].fillna(user_feats["age"].median())

    # 商品特征
    prod_feats = df_behavior.groupby("product_id").agg(
        prod_total_views=("behavior", "count"),
        prod_purchases=("behavior", lambda x: (x == "购买").sum()),
    ).reset_index()
    prod_feats = prod_feats.merge(df_products[["product_id", "price", "category"]], on="product_id", how="left")
    prod_feats["category_enc"] = LabelEncoder().fit_transform(prod_feats["category"].fillna("未知"))

    # 构建正负样本
    purchases = df_behavior[df_behavior["behavior"] == "购买"][["user_id", "product_id"]].drop_duplicates()
    purchases["label"] = 1

    # 负采样：随机用户-商品对（未购买的）
    all_users = df_behavior["user_id"].unique()
    all_prods = df_behavior["product_id"].unique()
    neg_samples = []
    np.random.seed(42)
    for _ in range(len(purchases) * 2):
        u = np.random.choice(all_users)
        p = np.random.choice(all_prods)
        if not ((purchases["user_id"] == u) & (purchases["product_id"] == p)).any():
            neg_samples.append({"user_id": u, "product_id": p, "label": 0})
    neg_df = pd.DataFrame(neg_samples)

    data = pd.concat([purchases, neg_df], ignore_index=True)

    # 合并特征
    data = data.merge(user_feats, on="user_id", how="left")
    data = data.merge(prod_feats[["product_id", "price", "prod_total_views", "prod_purchases", "category_enc"]],
                      on="product_id", how="left")
    data = data.fillna(0)

    feature_cols = ["total_actions", "n_products", "n_purchases", "avg_duration", "n_devices",
                    "age", "gender", "vip_level", "price", "prod_total_views", "prod_purchases", "category_enc"]
    return data, feature_cols


def ml_prediction(df_behavior, df_users, df_products):
    print("\n" + "=" * 60)
    print("模块6：机器学习预测（随机森林 + 神经网络）")
    print("=" * 60)

    data, feature_cols = build_features(df_behavior, df_users, df_products)
    X = data[feature_cols].values
    y = data["label"].values

    print(f"样本数: {len(data)}, 正样本: {y.sum():.0f}, 负样本: {(1-y).sum():.0f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ---- Random Forest ----
    print("\n[随机森林]")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    rf_proba = rf.predict_proba(X_test_s)[:, 1]
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)
    print(f"  准确率: {rf_acc:.4f}")
    print(f"  AUC: {rf_auc:.4f}")
    print(classification_report(y_test, rf_pred, target_names=["未购买", "购买"]))

    # 特征重要性
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("  特征重要性:")
    for feat, imp in importances.head(5).items():
        print(f"    {feat}: {imp:.4f}")

    # ---- Neural Network (MLP) ----
    print("\n[神经网络 MLP]")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.15)
    mlp.fit(X_train_s, y_train)
    mlp_pred = mlp.predict(X_test_s)
    mlp_proba = mlp.predict_proba(X_test_s)[:, 1]
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_auc = roc_auc_score(y_test, mlp_proba)
    print(f"  准确率: {mlp_acc:.4f}")
    print(f"  AUC: {mlp_auc:.4f}")
    print(classification_report(y_test, mlp_pred, target_names=["未购买", "购买"]))

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (1) 特征重要性
    importances.head(10).plot(kind="barh", ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("随机森林 - 特征重要性 Top10")
    axes[0, 0].invert_yaxis()

    # (2) ROC曲线
    for name, proba in [("Random Forest", rf_proba), ("MLP Neural Net", mlp_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_val = roc_auc_score(y_test, proba)
        axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0, 1].set_title("ROC曲线对比")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].legend()

    # (3) 混淆矩阵 - RF
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[0, 2],
                xticklabels=["未购买", "购买"], yticklabels=["未购买", "购买"])
    axes[0, 2].set_title("随机森林 - 混淆矩阵")

    # (4) 混淆矩阵 - MLP
    cm_mlp = confusion_matrix(y_test, mlp_pred)
    sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Oranges", ax=axes[1, 0],
                xticklabels=["未购买", "购买"], yticklabels=["未购买", "购买"])
    axes[1, 0].set_title("神经网络 - 混淆矩阵")

    # (5) 模型对比
    metrics = {"Accuracy": [rf_acc, mlp_acc], "AUC": [rf_auc, mlp_auc]}
    x = np.arange(len(metrics))
    w = 0.35
    axes[1, 1].bar(x - w/2, [metrics["Accuracy"][0], metrics["AUC"][0]], w, label="Random Forest", color="steelblue")
    axes[1, 1].bar(x + w/2, [metrics["Accuracy"][1], metrics["AUC"][1]], w, label="MLP Neural Net", color="coral")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(["Accuracy", "AUC"])
    axes[1, 1].set_title("模型性能对比")
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0.5, 1.0)

    # (6) MLP训练损失曲线
    if hasattr(mlp, "loss_curve_"):
        axes[1, 2].plot(mlp.loss_curve_, color="red")
        axes[1, 2].set_title("MLP训练损失曲线")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_ml_prediction.png", dpi=150)
    plt.close()
    print(f"\n✅ 图表已保存: {OUTPUT_DIR}/06_ml_prediction.png")

    return {
        "rf": {"accuracy": rf_acc, "auc": rf_auc},
        "mlp": {"accuracy": mlp_acc, "auc": mlp_auc},
        "feature_importance": importances.to_dict()
    }
