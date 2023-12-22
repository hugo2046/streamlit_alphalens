"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-15 09:31:26
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-18 15:48:11
FilePath:
Description: 
"""
from typing import List

import pandas as pd
import streamlit as st
from alphalens.utils import get_clean_factor_and_forward_returns
from data_service import DataLoader
from alphalens.streamit_tears import create_full_tear_sheet

from ..utils import capture_warnings


# 数据准备
@capture_warnings
def fetch_factor_data(loader: DataLoader) -> pd.DataFrame:
    factor_names: List[str] = st.session_state["alphlens_params"]["factor_name"]

    if len(factor_names) > 1:
        st.warning("分析模块仅分析单个因子,不能多选!", icon="warning")

    start_dt, end_dt = st.session_state["alphlens_params"]["date_range"]

    factor_data: pd.DataFrame = loader.get_factor_data(start_dt, end_dt, factor_names)
    codes: List[str] = factor_data["code"].unique().tolist()
    price: pd.DataFrame = loader.get_stock_price(codes, start_dt, end_dt)
    pricing: pd.DataFrame = pd.pivot_table(
        price, index="trade_date", columns="code", values="vwap"
    )
    quantiles: int = st.session_state["alphlens_params"]["quantiles"]
    periods: List[int] = st.session_state["alphlens_params"]["periods"]
    max_loss: float = st.session_state["alphlens_params"]["max_loss"]

    return get_clean_factor_and_forward_returns(
        factor_data.set_index(["trade_date", "code"])[factor_names]
        .sort_index()
        .dropna(),
        prices=pricing.shift(-1),
        quantiles=quantiles,
        periods=periods,
        max_loss=max_loss,
    )


# Step 4: 因子分析
def factor_report(loader: DataLoader):

    st.header("因子分析报告")

    if not st.session_state["alphlens_params"]:
        st.warning("请先选择因子!", icon="🚨")
        st.stop()
    
    factor: pd.DataFrame = fetch_factor_data(loader)

    create_full_tear_sheet(factor)
