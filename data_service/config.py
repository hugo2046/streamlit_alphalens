"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-14 23:17:04
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-14 23:18:13
FilePath: 
Description: 数据加载设置
"""
from typing import Dict


DB_CONN: Dict = dict(host="localhost", port=8848, username="admin", password="123456")
FACTOR_TABLE_NAME: str = "vol_battle"
PRICE_TABLE_NAME: str = "stock"
FACTPR_DB_PATH: str = "dfs://factor_data"
PRICE_DB_PATH: str = "dfs://daily_data"

CSV_PATH: Dict = {"price_path": "data/price.csv", "factor_path": "data/factor.csv"}
