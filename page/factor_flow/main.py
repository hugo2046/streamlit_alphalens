"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-18 15:39:47
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-18 16:49:02
FilePath: 
Description:
"""
import streamlit as st
from typing import Dict
import streamlit_antd_components as sac
from .factor_select import factor_selector
from .factor_processor import factor_preprocess
from .factor_report import factor_report
from ..utils import local_json_lottie


def update_web_tag():
    lottie_local_json: Dict = {
        "step 1": "page/img/step1.json",
        "step 2": "page/img/moon.json",
        "step 3": "page/img/rockets.json",
        "step 4": "page/img/step4.json",
    }

    local_json_lottie(lottie_local_json[st.session_state["step"]], height=200)


def update_step_status(step: str):
    st.session_state["step"] = step


def main(params: Dict):
    update_web_tag()

    with st.expander("因子预处理流程", True):
        items = [
            sac.StepsItem(
                title="step 1",
                subtitle="因子选择",
                description="选择所需因子",
                disabled=False,
            ),
            sac.StepsItem(
                title="step 2",
                subtitle="因子预处理",
                description="对因子进行预处理",
                disabled=False,
            ),
            sac.StepsItem(
                title="step 3",
                subtitle="因子合成",
                description="对因子进行合成",
                disabled=False,
            ),
            sac.StepsItem(
                title="step 4",
                subtitle="因子分析",
                description="对因子进行分析",
                disabled=False,
            ),
        ]

        # 创建步骤
        step = sac.steps(items=items)

    update_step_status(step)

    if step == "step 1":
        factor_selector(params["factor_names"])

    elif step == "step 2":
        if st.session_state["alphlens_params"]:
            factor_preprocess()

        else:
            sac.alert(
                message="Alert Message",
                description="请先完成[Step 1 因子选择]!",
                banner=True,
                icon=True,
                closable=True,
                type="warning",
            )

    elif step == "step 3":
        sac.alert(
            message="Alert Message",
            description="暂未完成...",
            banner=True,
            icon=True,
            closable=True,
            type="warning",
        )

    elif step == "step 4":
        factor_report(params["loader"])


factor_flow: Dict = {"factor": {"main": main}}
