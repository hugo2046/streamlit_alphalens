"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-15 14:35:52
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-15 14:55:02
FilePath:
Description: 
"""
import functools
import hashlib
import json
import warnings
from typing import Dict, List, Tuple

import streamlit as st
import streamlit_antd_components as sac
from streamlit_lottie import st_lottie


def local_json_lottie(path: str, **kwargs):
    with open(path, "r") as f:
        animation = json.load(f)
    st_lottie(animation, **kwargs)


def hash_text(text: str, algorithm="sha256") -> str:
    if algorithm.lower() == "md5":
        hasher = hashlib.md5()
    elif algorithm.lower() == "sha1":
        hasher = hashlib.sha1()
    else:
        hasher = hashlib.sha256()

    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def capture_warnings(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            result = func(*args, **kwargs)
            for warning in caught_warnings:
                sac.alert(
                    message="Alphalens Alert Message",
                    description=str(warning.message),
                    banner=True,
                    icon="🙊",
                    closable=True,
                )

        return result

    return wrapper

def exception_catcher(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
           
            sac.alert(
                    label="Alphalens Alert Message",
                    description=str(f"Caught an exception in {func.__name__}: {str(e)}"),
                    banner=True,
                    icon="🙊",
                    closable=False,
                )
            st.stop()
            
    return wrapper


def get_step_num(step: str) -> int:
    return int(step.split(" ")[-1])


def view_alphalens_params(alphalens_params: Dict) -> List:
    # 从alphalens_params字典中提取值
    factor_names: List[Tuple] = [
        (name,) for name in alphalens_params.get("factor_name", [])
    ]
    date_range: Tuple[str, str] = alphalens_params.get("date_range", ())
    periods: Tuple[int] = alphalens_params.get("periods", ())
    max_loss: float = alphalens_params.get("max_loss", 0)
    quantiles: int = alphalens_params.get("quantiles", 0)

    # 构造期望的数据结构
    periods_formatted: List[Tuple] = []
    for p in periods:
        periods_formatted.extend([(str(p),), ","])
    periods_formatted.pop()  # 移除最后一个逗号

    return [
        "当前选择的因子为：",
        factor_names,
        "，回测范围为：",
        [(date_range[0],), "至", (date_range[1],)],
        ",Alphalens参数为:quantiles=",
        (str(quantiles),),
        ",periods=",
        periods_formatted,
        ",max_loss=",
        (f"{max_loss:.2f}",),
    ]
