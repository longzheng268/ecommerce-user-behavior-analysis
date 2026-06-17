"""
生成模拟电商用户行为数据
包含：用户表、商品表、行为日志、订单表、营销活动表
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# ============ 1. 用户表 ============
N_USERS = 2000
user_ids = [f"U{str(i).zfill(5)}" for i in range(1, N_USERS + 1)]
ages = np.random.normal(32, 10, N_USERS).clip(18, 65).astype(int)
genders = np.random.choice(["男", "女", "未知"], N_USERS, p=[0.48, 0.45, 0.07])
cities = np.random.choice(
    ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "重庆", "西安",
     "苏州", "长沙", "郑州", "青岛", "天津", "厦门", "合肥", "昆明", "大连", "福州"],
    N_USERS
)
reg_dates = [datetime(2024, 1, 1) + timedelta(days=int(d))
             for d in np.random.exponential(180, N_USERS).clip(0, 365)]
vip_levels = np.random.choice([0, 1, 2, 3], N_USERS, p=[0.4, 0.3, 0.2, 0.1])

df_users = pd.DataFrame({
    "user_id": user_ids,
    "age": ages,
    "gender": genders,
    "city": cities,
    "register_date": reg_dates,
    "vip_level": vip_levels
})
# 故意插入缺失值和异常值供清洗
missing_idx = np.random.choice(N_USERS, 60, replace=False)
df_users.loc[missing_idx[:30], "age"] = np.nan
df_users.loc[missing_idx[30:50], "city"] = ""
df_users.loc[missing_idx[50:], "gender"] = np.nan
df_users.loc[np.random.choice(N_USERS, 10), "age"] = -1  # 异常值

# ============ 2. 商品表 ============
N_PRODUCTS = 500
categories = ["电子产品", "服装鞋帽", "食品饮料", "家居用品", "美妆个护",
              "运动户外", "图书文具", "母婴用品", "家电数码", "汽车用品"]
product_ids = [f"P{str(i).zfill(4)}" for i in range(1, N_PRODUCTS + 1)]
product_names = [f"商品_{cat}_{i}" for i, cat in
                 zip(range(N_PRODUCTS), np.random.choice(categories, N_PRODUCTS))]
prices = np.random.lognormal(4, 1, N_PRODUCTS).clip(5, 9999).round(2)

df_products = pd.DataFrame({
    "product_id": product_ids,
    "product_name": product_names,
    "category": np.random.choice(categories, N_PRODUCTS),
    "price": prices,
    "brand": np.random.choice([f"品牌{chr(65+i)}" for i in range(30)], N_PRODUCTS)
})

# ============ 3. 行为日志 ============
N_BEHAVIORS = 200000
behaviors = ["浏览", "搜索", "加购", "收藏", "购买"]
behavior_weights = [0.50, 0.25, 0.12, 0.08, 0.05]

start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31)
date_range_days = (end_date - start_date).days

# 构建带季节性的行为数据（双11/618高峰）
base_timestamps = np.random.uniform(0, date_range_days, N_BEHAVIORS)
timestamps = [start_date + timedelta(days=float(d)) for d in base_timestamps]

# 促销高峰：6月1-18日、11月1-11日权重加倍
promo_mask = []
for ts in timestamps:
    if (ts.month == 6 and 1 <= ts.day <= 18) or (ts.month == 11 and 1 <= ts.day <= 11):
        promo_mask.append(True)
    else:
        promo_mask.append(False)
promo_mask = np.array(promo_mask)

df_behavior = pd.DataFrame({
    "user_id": np.random.choice(user_ids, N_BEHAVIORS),
    "product_id": np.random.choice(product_ids, N_BEHAVIORS),
    "behavior": np.random.choice(behaviors, N_BEHAVIORS, p=behavior_weights),
    "timestamp": timestamps,
    "session_duration": np.random.exponential(300, N_BEHAVIORS).clip(5, 3600).round(1),
    "device": np.random.choice(["手机", "电脑", "平板"], N_BEHAVIORS, p=[0.65, 0.25, 0.10])
})
# 促销期购买比例提升
promo_buy_boost = promo_mask & (df_behavior["behavior"] == "浏览")
boost_idx = df_behavior.index[promo_buy_boost]
replace_idx = np.random.choice(boost_idx, size=int(len(boost_idx) * 0.15), replace=False)
df_behavior.loc[replace_idx, "behavior"] = "购买"

# 故意插入异常数据
df_behavior.loc[np.random.choice(N_BEHAVIORS, 50), "session_duration"] = -100
df_behavior.loc[np.random.choice(N_BEHAVIORS, 200), "timestamp"] = pd.NaT

# ============ 4. 订单表 ============
purchase_rows = df_behavior[df_behavior["behavior"] == "购买"].copy()
purchase_rows = purchase_rows.head(10000)  # 取前1万条购买记录
order_ids = [f"O{str(i).zfill(6)}" for i in range(1, len(purchase_rows) + 1)]
purchase_rows["order_id"] = order_ids
purchase_rows["quantity"] = np.random.choice([1,1,1,2,2,3], len(purchase_rows))
purchase_rows = purchase_rows.merge(df_products[["product_id", "price"]], on="product_id", how="left")
purchase_rows["total_amount"] = (purchase_rows["quantity"] * purchase_rows["price"]).round(2)
purchase_rows["payment"] = np.random.choice(["支付宝", "微信", "银行卡", "花呗"], len(purchase_rows), p=[0.35, 0.35, 0.15, 0.15])
df_orders = purchase_rows[["order_id", "user_id", "product_id", "timestamp", "quantity", "total_amount", "payment"]].copy()
# 插入缺失
df_orders.loc[np.random.choice(len(df_orders), 30), "total_amount"] = np.nan

# ============ 5. 营销活动表 ============
campaigns = [
    {"campaign_id": "C001", "name": "618年中大促", "start_date": datetime(2025, 6, 1), "end_date": datetime(2025, 6, 18), "type": "满减", "discount_rate": 0.15, "group": "B"},
    {"campaign_id": "C002", "name": "双11狂欢", "start_date": datetime(2025, 11, 1), "end_date": datetime(2025, 11, 11), "type": "折扣", "discount_rate": 0.20, "group": "B"},
    {"campaign_id": "C003", "name": "春季新品", "start_date": datetime(2025, 3, 1), "end_date": datetime(2025, 3, 15), "type": "优惠券", "discount_rate": 0.10, "group": "A"},
    {"campaign_id": "C004", "name": "暑期特惠", "start_date": datetime(2025, 7, 15), "end_date": datetime(2025, 8, 15), "type": "满减", "discount_rate": 0.12, "group": "A"},
    {"campaign_id": "C005", "name": "年末清仓", "start_date": datetime(2025, 12, 20), "end_date": datetime(2025, 12, 31), "type": "折扣", "discount_rate": 0.25, "group": "B"},
]
df_campaigns = pd.DataFrame(campaigns)

# ============ 保存 ============
df_users.to_csv("data/users.csv", index=False, encoding="utf-8-sig")
df_products.to_csv("data/products.csv", index=False, encoding="utf-8-sig")
df_behavior.to_csv("data/behavior.csv", index=False, encoding="utf-8-sig")
df_orders.to_csv("data/orders.csv", index=False, encoding="utf-8-sig")
df_campaigns.to_csv("data/campaigns.csv", index=False, encoding="utf-8-sig")

print(f"✅ 数据生成完成！")
print(f"  用户表: {len(df_users)} 行")
print(f"  商品表: {len(df_products)} 行")
print(f"  行为日志: {len(df_behavior)} 行")
print(f"  订单表: {len(df_orders)} 行")
print(f"  营销活动: {len(df_campaigns)} 行")
