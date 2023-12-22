"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-14 13:57:46
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-14 13:58:04
FilePath: 
Description: 
"""
import pandas as pd




def datetime2str(watch_dt: pd.Timestamp, fmt: str = "%Y-%m-%d") -> str:
    return pd.to_datetime(watch_dt).strftime(fmt)



