"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2024-01-08 14:52:22
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2024-01-08 15:04:29
FilePath: 
Description: 
"""
from typing import Dict, List, Tuple
import numpy as np
import streamlit as st
from analyzer.streamlit_analyze import FactorAnalyzer
from data_service import Loader
import pandas as pd
from ..factor_flow.factor_report import fetch_factor_data
from ..factor_flow.factor_select import factor_selector
from .utils import (
    merge_table,
    highlight_by_group,
    get_factor_board,
)


if "alphlens_params" not in st.session_state:
    st.session_state["alphlens_params"] = {}


def mult_factor_report(loader: Loader) -> FactorAnalyzer:
    if (
        st.session_state.get("alphlens_params", None) is None
        or st.session_state["alphlens_params"].get("factor_name", None) is None
    ):
        st.warning("请先选择因子!", icon="🚨")
        st.stop()

    status_placeholder = st.empty()

    with status_placeholder.status("因子分析中...", expanded=False) as status:
        analyze_dict: Dict = fetch_factor_data(loader, True)
        status.update(label="分析完毕!", state="complete", expanded=True)

    status_placeholder.empty()
    st.toast("分析完毕!", icon="🎉")

    def style_negative(v):
        return np.where(v.values > 0, "color:red;", "color:green;")

    # 画时序
    board: pd.DataFrame = get_factor_board(analyze_dict)
    subset: List[str] = [
        "Annualvolatility",
        "CumReturn",
        "AnnualReturn",
        "MaxDrawdown",
        "SharpeRatio",
        "MinimumQuantileAverageReturn",
        "MaximumQuantileAverageReturn",
        "IR",
        "IC Mean",
    ]
    column_config = {
        "CumulativeReturnsVeiws": st.column_config.LineChartColumn(
            "CumulativeReturnsVeiws"
        ),
        "TopDownCumulativeReturnsVeiws": st.column_config.LineChartColumn(
            "TopDownCumulativeReturnsVeiws"
        ),
        "CumulativeInformationCoefficient": st.column_config.LineChartColumn(
            "CumulativeInformationCoefficient"
        ),
    }
    st.markdown("**因子看板**")
    st.dataframe(
        board.style.apply(style_negative, subset=subset[:-2]).format(
            "{:.3%}", subset=subset[:-2]
        ).format("{:.3}",subset=subset[-2:]),
        column_config=column_config,
    )

    # 画table
    tables_names: Tuple = {
        "plot_returns_table": "因子收益表",
        "plot_tstats_table": "T统计量表",
        "plot_information_table": "信息比率(IC)相关表",
    }

    dfs: Dict = {}
    for table_method, table_name in tables_names.items():
        for k, factor_analyze in analyze_dict.items():
            dfs[k] = getattr(factor_analyze, table_method)(make_pretty=False)

        all_factor_frame: pd.DataFrame = merge_table(dfs)
        all_factor_frame: pd.DataFrame = all_factor_frame.stack(level=0)
        all_factor_frame.index.names = ["Metric", "Period"]
        st.markdown(f"**{table_name}**")

        if all_factor_frame.shape[1] > 0:
            # st.markdown(
            #     highlight_by_group(all_factor_frame).to_html(), unsafe_allow_html=True
            # )

            st.dataframe(
                all_factor_frame.style.background_gradient(
                    cmap="RdYlGn_r", axis=1
                ).format(precision=3)
            )
        else:
            st.dataframe(all_factor_frame.style.format(precision=3))


def main(params: Dict):
    factor_selector(params["factor_names"])
    mult_factor_report(params["loader"])
