"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-15 09:31:26
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-15 10:06:35
FilePath: 
Description: Step 1: 因子选择 
"""
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from streamlit_utils.utils import datetime2str
from ..utils import view_alphalens_params
from annotated_text import annotated_text


if "alphlens_params" not in st.session_state:
    st.session_state["alphlens_params"] = {}


# Step 1: 因子选择
def factor_selector(factor_names: List[str]):
    """设置因子及回测参数"""

    if not factor_names:
        st.warning("因子列表为空，请检查数据", icon="warning")

    st.header("🧬因子选择")

    with st.form("alphalens_params"):

        st.subheader("🪢选择所需因子", divider="gray")

        sel_factors: List[str] = st.multiselect(
            "选择因子", factor_names, default=factor_names[0], help="选择所需因子一个或多个"
        )

        st.subheader("⚙️Alphalens参数设置", divider="gray")

    
        data_range: Tuple[pd.Timestamp, pd.Timestamp] = st.slider(
            "回测范围选择",
            value=[
                pd.to_datetime("2018-01-01").date(),
                pd.to_datetime("2022-12-31").date(),
            ],
            min_value=pd.to_datetime("2014-01-01").date(),
            max_value=pd.to_datetime("2024-01-12").date(),
            format="YYYY/MM/DD",
        )

        quantiles: int = st.number_input(
            "分组设置(quantiles)", value=10, step=1, min_value=5, max_value=20
        )
        periods: str = st.text_input(
            "期间设置(periods)", value="1,5,10", help="以逗号分隔(注意需要在英语输入法下输入)"
        )
        max_loss: float = st.number_input(
            "最大损失设置(max_loss)",
            value=0.35,
            step=0.01,
            min_value=0.0,
            max_value=1.0,
            help="以逗号分隔(注意需要在英语输入法下输入)",
        )

        submitted = st.form_submit_button("提交参数")

        if submitted:
            # 转换数据格式
            data_range: Tuple[str, str] = tuple(
                datetime2str(data_range, "%Y-%m-%d").tolist()
            )
            quantiles: int = int(quantiles)
            periods: Tuple[int] = tuple(map(int, periods.split(",")))
            max_loss: float = float(max_loss)

            alphalens_params: Dict = {
                "factor_name": sel_factors,
                "date_range": data_range,
                "quantiles": quantiles,
                "periods": periods,
                "max_loss": max_loss,
            }

            st.session_state["alphlens_params"] = alphalens_params

            annotated_text(view_alphalens_params(alphalens_params))
