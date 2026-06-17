"""Shared visualization helpers for matplotlib charts."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config


def setup_chinese_fonts():
    """Configure matplotlib for Chinese font support."""
    plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save_chart(fig, filename):
    """Save a matplotlib figure to config.OUTPUT_DIR with dpi=150."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
