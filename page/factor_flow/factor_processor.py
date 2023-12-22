"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-18 13:35:40
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-18 13:47:28
FilePath: 
Description: Step 2: 因子预处理

"""
from typing import Dict, List,Tuple
import pandas as pd
import streamlit as st

from ..utils import hash_text

DEFAULT: Dict = {
    "DropnaProcessor": {"params": {"fields_group": None}},
    "DropnaLabel": {"params": {"fields_group": "label"}},
    "DropCol": {"params": {"col_list": "[]"}},
    "FilterCol": {"params": {"fields_group": "feature", "col_list": []}},
    "TanhProcess": {},
    "ProcessInf": {},
    "Fillna": {"params": {"fields_group": None, "fillna_value": 0.0}},
    "MinMaxNorm": {"params": {"fields_group": None}},
    "ZScoreNorm": {"params": {"fields_group": None}},
    "RobustZScoreNorm": {"params": {"fields_group": None, "clip_outlier": "true"}},
    "CSZScoreNorm": {"params": {"fields_group": None, "method": "zscore"}},
    "CSRankNorm": {"params": {"fields_group": None}},
    "CSZFillna": {"params": {"fields_group": None}},
    "HashStockFormat": {},
    "TimeRangeFlt": {"params": {"freq": "day"}},
}

if "infer_processors" not in st.session_state:
    st.session_state["infer_processors"] = []

if "learn_processors" not in st.session_state:
    st.session_state["learn_processors"] = []

if "periods" not in st.session_state:
    st.session_state["periods"] = {}


def update_session_state(key, value):
    st.session_state[key] = value


def split_timeseries():
    start_dt, end_dt = pd.to_datetime(st.session_state["alphlens_params"]["date_range"])
    all_days:pd.DatetimeIndex = pd.date_range(start_dt, end_dt, freq="D")

    fit_end_time = all_days[int(len(all_days) * 0.6)].date()
    valid_end_time = all_days[int(len(all_days) * 0.8)].date()

    train_periods:Tuple = st.date_input("训练集时间范围划分", (start_dt.date(), fit_end_time))
    valid_periods:Tuple = st.date_input("验证集时间范围划分", (fit_end_time + pd.Timedelta(days=1), valid_end_time))
    test_periods:Tuple = st.date_input("测试集时间范围划分", (valid_end_time + pd.Timedelta(days=1), end_dt.date()))

    update_session_state("periods", {
        "train_periods": train_periods, 
        "valid_periods": valid_periods, 
        "test_periods": test_periods
    })


def create_processor_input_selector(processor_name: str):
    if st.checkbox(str(processor_name), key=processor_name):
        params: Dict = DEFAULT[processor_name].get("params", {})
        processor_dict: Dict = {
            "class": processor_name,
            "module_path": "qlib.data.dataset.processor",
            "kwargs": {},
        }

        if params:
            for k, v in params.items():
                if k == "fields_group":
                    options: List = [None, "label", "feature"]
                    idx: int = options.index(v)
                    selected_option = st.selectbox(
                        k,
                        options,
                        index=idx,
                        key=hash_text("".join([processor_name, k, "a"])),
                    )
                    processor_dict["kwargs"][k] = selected_option
                elif k == "method":
                    options: List = ["zscore", "robust"]
                    idx: int = options.index(v)
                    selected_option = st.selectbox(
                        k,
                        options,
                        index=idx,
                        key=hash_text("".join([processor_name, k, "b"])),
                    )
                    processor_dict["kwargs"][k] = selected_option

                elif k == "fillna_value":
                    selected_option = st.number_input(
                        k,
                        value=v,
                        key=hash_text("".join([processor_name, k, "c"])),
                    )
                    processor_dict["kwargs"][k] = selected_option
                elif k == "clip_outlier":
                    options: List = ["true", "false"]
                    idx: int = options.index(v)
                    selected_option = st.selectbox(
                        k,
                        options,
                        index=idx,
                        key=hash_text("".join([processor_name, k, "d"])),
                    )
                    processor_dict["kwargs"][k] = selected_option

                elif k in ["freq", "col_list"]:
                    selected_option = st.text_input(
                        k,
                        value=v,
                        key=hash_text("".join([processor_name, k, "e"])),
                    )
                    processor_dict["kwargs"][k] = selected_option

        return processor_dict


# Step 2: 因子预处理
def factor_preprocess():
    st.header("🧰因子预处理")
    st.info("此部分是可选项,如果不设置则不对原始因子进行处理.", icon="ℹ️")

    st.subheader("📝训练\测试集的划分" ,divider="gray")

    split_timeseries()

    col1, col2 = st.columns(2)
    update_infer_processors: List[Dict] = []
    update_learn_processors: List[Dict] = []
    with col1:
        st.subheader("📌推理处理器(:gray[Infer_processors])设置" ,divider="gray")
        infer_processors: List[str] = ["MinMaxNorm", "ZScoreNorm", "RobustZScoreNorm"]
        for processor_name in infer_processors:
            s1: Dict = create_processor_input_selector(processor_name)
            update_infer_processors.append(s1)

        update_session_state("infer_processors", update_infer_processors)

    with col2:
        st.subheader("📍学习处理器(:gray[Learn_processors])设置" ,divider="gray")
        Learn_processors: List[str] = [
            "DropnaProcessor",
            "Fillna",
            "CSZFillna",
            "ProcessInf",
            "TanhProcess",
            "CSZScoreNorm",
            "CSRankNorm",
            "DropnaLabel",
            "TimeRangeFlt",
            "DropCol",
            "FilterCol",
            "HashStockFormat",
        ]
        for processor_name in Learn_processors:
            s2: Dict = create_processor_input_selector(processor_name)
            update_learn_processors.append(s2)

        update_session_state("learn_processors", update_learn_processors)
