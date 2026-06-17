"""
模块5：ARIMA时间序列分析
- 预测用户购物行为的长期趋势
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUTPUT_DIR = "output"


def arima_forecast(df_behavior, df_orders):
    print("\n" + "=" * 60)
    print("模块5：ARIMA时间序列预测")
    print("=" * 60)

    # 每日购买量时间序列
    df_behavior["timestamp"] = pd.to_datetime(df_behavior["timestamp"])
    daily_purchases = df_behavior[df_behavior["behavior"] == "购买"].set_index("timestamp").resample("D").size()
    daily_purchases = daily_purchases.asfreq("D", fill_value=0)
    print(f"时间序列长度: {len(daily_purchases)} 天")
    print(f"日均购买量: {daily_purchases.mean():.1f}")

    # ADF平稳性检验
    adf_result = adfuller(daily_purchases)
    print(f"\nADF检验统计量: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(f"序列{'平稳' if adf_result[1] < 0.05 else '非平稳'}")

    # 一阶差分
    diff1 = daily_purchases.diff().dropna()
    adf_diff = adfuller(diff1)
    print(f"一阶差分后 p-value: {adf_diff[1]:.4f}")

    # 确定参数 (p, d, q)
    d = 0 if adf_result[1] < 0.05 else 1
    # 简化：尝试几组参数选最优
    best_aic = np.inf
    best_order = (1, d, 1)
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                model = ARIMA(daily_purchases, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue

    print(f"\n最优ARIMA参数: {best_order} (AIC: {best_aic:.2f})")

    # 拟合最终模型
    model_final = ARIMA(daily_purchases, order=best_order)
    fitted_final = model_final.fit()

    # 预测未来30天
    forecast_steps = 30
    forecast = fitted_final.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # 拟合评估
    residuals = fitted_final.resid
    rmse = np.sqrt((residuals ** 2).mean())
    mae = np.abs(residuals).mean()
    print(f"\n模型评估:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (1) 原始序列 + 拟合 + 预测
    ax1 = axes[0, 0]
    ax1.plot(daily_purchases.index, daily_purchases.values, label="实际值", linewidth=0.8, color="steelblue")
    fitted_values = fitted_final.fittedvalues
    ax1.plot(fitted_values.index, fitted_values.values, label="拟合值", linewidth=0.8, color="orange", alpha=0.7)
    future_dates = pd.date_range(start=daily_purchases.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
    ax1.plot(future_dates, forecast_mean.values, label="预测值", color="red", linewidth=1.5)
    ax1.fill_between(future_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color="red", alpha=0.1)
    ax1.set_title(f"ARIMA{best_order} 购买量预测")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=45)

    # (2) 残差诊断
    ax2 = axes[0, 1]
    ax2.plot(residuals.index, residuals.values, linewidth=0.5, color="gray")
    ax2.axhline(y=0, color="red", linestyle="--")
    ax2.set_title("残差序列")

    # (3) ACF
    plot_acf(residuals, ax=axes[1, 0], lags=30)
    axes[1, 0].set_title("残差ACF")

    # (4) PACF
    plot_pacf(residuals, ax=axes[1, 1], lags=30)
    axes[1, 1].set_title("残差PACF")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_arima_forecast.png", dpi=150)
    plt.close()
    print(f"\n✅ 图表已保存: {OUTPUT_DIR}/05_arima_forecast.png")

    # 月度趋势预测报告
    monthly = daily_purchases.resample("ME").sum()
    print("\n[月度购买量]")
    for date, val in monthly.items():
        print(f"  {date.strftime('%Y-%m')}: {val:.0f}")

    return forecast_mean, forecast_ci, {"rmse": rmse, "mae": mae, "order": best_order}
