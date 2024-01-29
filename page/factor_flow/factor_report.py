"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-15 09:31:26
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-18 15:48:11
FilePath:
Description: 
"""
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac

# from alphalens.streamit_tears import create_full_tear_sheet
# from streamlit_utils.tear import create_full_tear_sheet
# from alphalens.utils import get_clean_factor_and_forward_returns
from analyzer.streamlit_analyze import FactorAnalyzer, create_full_tear_sheet
from data_service import Loader
from joblib import Parallel, delayed

from ..utils import capture_warnings, exception_catcher

if "alphlens_params" not in st.session_state:
    st.session_state["alphlens_params"] = {}


def get_input_factor_Name():
    if "alphlens_params" not in st.session_state:
        raise ValueError("st.session_state中不存在alphlens_params关键字")

    return st.session_state["alphlens_params"].get("factor_name", None)


def prepare_params(loader) -> Dict:
    if "alphlens_params" not in st.session_state:
        raise ValueError("st.session_state中不存在alphlens_params关键字")

    params = st.session_state["alphlens_params"]

    start_dt, end_dt = params["date_range"]
    factor_names: List[str] = get_input_factor_Name()
    factor_data: pd.DataFrame = loader.get_factor_data(
        factor_names, start_dt=start_dt, end_dt=end_dt
    )
    factor_data["trade_date"] = pd.to_datetime(factor_data["trade_date"])
    codes: List[str] = factor_data["code"].unique().tolist()
    price: pd.DataFrame = loader.get_stock_price(
        codes, start_dt=start_dt, end_dt=end_dt, fields=st.session_state["price_type"]
    )
    pricing: pd.DataFrame = pd.pivot_table(
        price, index="trade_date", columns="code", values=st.session_state["price_type"]
    )
    quantiles: int = params["quantiles"]
    periods: List[int] = params["periods"]
    max_loss: float = params["max_loss"]

    return dict(
        factor=factor_data,
        prices=pricing,
        quantiles=quantiles,
        periods=periods,
        max_loss=max_loss,
        factor_names=factor_names,
    )


def load_analyzer(factor_name: str, params: Dict) -> Tuple[str, FactorAnalyzer]:
    factor_ser: pd.Series = (
        params["factor"]
        .set_index(["trade_date", "code"])
        .query("factor_name==@factor_name")["value"]
        .sort_index()
        .dropna()
    )

    if params["prices"].empty or factor_ser.empty:
        raise ValueError(f"{factor_name} 数据为空!")

    factor_analyzer = FactorAnalyzer(
        factor_ser,
        prices=params["prices"].shift(-1),
        quantiles=params["quantiles"],
        periods=params["periods"],
        max_loss=params["max_loss"],
    )

    st.toast(f"{factor_name}因子分析完毕!", icon="🎉")
    return (factor_name, factor_analyzer)


# 数据准备
# @capture_warnings
def fetch_factor_data(loader: Loader, mult: bool = False) -> FactorAnalyzer:
    params: Dict = prepare_params(loader)
    factor_names: List[str] = params["factor_names"]
    factor_name: str = factor_names[0]

    if not mult and len(factor_names) > 1:
        st.warning("分析模块仅分析单个因子,不能多选!", icon="🚨")
        st.stop()

    if mult:
        with Parallel(n_jobs=2) as parallel:
            factor_analyzer = parallel(
                delayed(load_analyzer)(factor_name, params)
                for factor_name in factor_names
            )
        factor_analyzer: Dict = dict(factor_analyzer)
    else:
        factor_analyzer = load_analyzer(factor_name, params)[1]

    return factor_analyzer


# Step 4: 因子分析
@exception_catcher
def factor_report(loader: Loader):
    if (
        st.session_state.get("alphlens_params", None) is None
        or st.session_state["alphlens_params"].get("factor_name", None) is None
    ):
        st.warning("请先选择因子!", icon="🚨")
        st.stop()

    status_placeholder = st.empty()

    with status_placeholder.status("因子分析中...", expanded=False) as status:

        factor_analyze: FactorAnalyzer = fetch_factor_data(loader)

        status.update(label="分析完毕!", state="complete", expanded=True)

    status_placeholder.empty()

    factor_name: List[str] = st.session_state["alphlens_params"].get("factor_name")

    if isinstance(factor_name, list) & len(factor_name) > 0:
        factor_name: str = factor_name[0]
    # create_full_tear_sheet(factor)

    st.header("📰因子分析报告", divider=True)
    create_full_tear_sheet(factor_analyze, factor_name)
