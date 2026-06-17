"""Shared data loading utilities with caching and datetime parsing."""
import os
import pandas as pd
import config

_data_cache = {}


def _load_csv(filename):
    """Load a CSV with BOM handling and caching."""
    if filename not in _data_cache:
        path = os.path.join(config.DATA_DIR, filename)
        _data_cache[filename] = pd.read_csv(path, encoding="utf-8-sig")
    return _data_cache[filename].copy()


def load_users() -> pd.DataFrame:
    return _load_csv(config.USERS_CSV)


def load_products() -> pd.DataFrame:
    return _load_csv(config.PRODUCTS_CSV)


def load_behavior() -> pd.DataFrame:
    df = _load_csv(config.BEHAVIOR_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_orders() -> pd.DataFrame:
    df = _load_csv(config.ORDERS_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_campaigns() -> pd.DataFrame:
    df = _load_csv(config.CAMPAIGNS_CSV)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df


def load_all() -> dict:
    return {
        "users": load_users(),
        "products": load_products(),
        "behavior": load_behavior(),
        "orders": load_orders(),
        "campaigns": load_campaigns(),
    }
