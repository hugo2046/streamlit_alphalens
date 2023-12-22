"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-18 16:26:10
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-18 17:02:45
FilePath: 
Description: 
"""
from typing import List

import pandas as pd
import qlib
import streamlit as st
from data_service import DataLoader
from qlib.data.dataset.loader import StaticDataLoader

qlib.init(provider_uri=".../data", region="cn")


def get_processor_params(key: str):
    ls = st.session_state.get(key, [])
    ls_now = [item for item in ls if item]
    return ls_now


def get_factor_names_frome_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in ["code", "trade_date", "next_ret"]]


def get_data_handler_config(loader: DataLoader):
    st.session_state["periods"]
    learn_processors: List = get_processor_params("learn_processors") 
    infer_processors: List = get_processor_params("infer_processors")

    factor_names: List[str] = st.session_state["alphlens_params"]["factor_name"]

    if len(factor_names) > 1:
        st.warning("分析模块仅分析单个因子,不能多选!", icon="warning")

    start_dt, end_dt = st.session_state["alphlens_params"]["date_range"]

    factor_data: pd.DataFrame = loader.get_factor_data(start_dt, end_dt, factor_names)
