"""Centralized configuration for the e-commerce analysis project."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Flask
DEBUG = False
HOST = "0.0.0.0"
PORT = 5000

# Chart color palettes
PALETTE_1 = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de"]
PALETTE_2 = ["#c23531", "#2f4554", "#61a0a8", "#d48265", "#91c7ae"]
PALETTE_3 = ["#37A2DA", "#32C5E9", "#67E0E3", "#9FE6F0", "#FFDB5C"]
PALETTE_4 = ["#ff7f50", "#87cefa", "#da70d6", "#32cd32", "#6495ed"]
PALETTE_5 = ["#2ec7c9", "#b6a2de", "#5ab1ef", "#ffb980", "#d87a80"]

# Model hyperparameters
RF_N_ESTIMATORS = 100
MLP_LAYERS = (128, 64, 32)
KMEANS_K_RANGE = range(2, 7)
ARIMA_MAX_ORDER = 3

# Data file paths (filenames only, resolved relative to DATA_DIR)
USERS_CSV = "users.csv"
PRODUCTS_CSV = "products.csv"
BEHAVIOR_CSV = "behavior.csv"
ORDERS_CSV = "orders.csv"
CAMPAIGNS_CSV = "campaigns.csv"
