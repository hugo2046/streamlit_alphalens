"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-15 08:54:14
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-15 09:10:41
FilePath: 
Description: 
"""

from typing import List

import streamlit as st
import streamlit_antd_components as sac
from data_service import DolinphdbLoader
from page.factor_flow import FACTOR_FLOW
from page.factor_compare import FACTOR_COMPARE
from page.factor_compare import board_table
from page.overview import overview, account_settings
from page.utils import local_json_lottie

st.set_page_config(layout="wide", page_title="Alphalens-stream App", page_icon="🧊")

st.markdown(
    f"""
    <style>
    .stApp .main .block-container{{
        padding-top:30px
    }}
    .stApp [data-testid='stSidebar']>div:nth-child(1)>div:nth-child(2){{
        padding-top:50px
    }}
    iframe{{
        display:block;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


if "index" not in st.session_state:
    st.session_state["index"] = 0

if "step" not in st.session_state:
    st.session_state["step"] = "step 1"

if "loader_type" not in st.session_state:
    st.session_state["loader_type"] = "ddb"

if "price" not in st.session_state:
    st.session_state["price_type"] = "avg_price"

if "db_or_csv" not in st.session_state:
    st.session_state["db_or_csv"] = True

with st.sidebar.container():
    st.subheader("Workflow")
    kernel = sac.Tag("核心", color="purple", bordered=False)
    menu = sac.menu(
        [
            sac.MenuItem("主页", icon="house"),
            sac.MenuItem(
                "多因子模块",
                icon="buildings",
                type="group",
                children=[
                    sac.MenuItem("因子分析", icon="compass", tag=kernel),
                    sac.MenuItem("因子对比", icon="basket"),
                    sac.MenuItem(
                        "因子看板", icon="joystick", tag=sac.Tag("日度更新", color="green")
                    ),
                ],
            ),
            sac.MenuItem(
                "回测模块",
                icon="bank",
                type="group",
                children=[
                    sac.MenuItem("回测分析", icon="activity"),
                ],
            ),
            sac.MenuItem(type="divider", disabled=True),
            sac.MenuItem(
                "reference",
                icon="box-fill",
                type="group",
                children=[sac.MenuItem("设置", icon="setting")],
            ),
        ],
        index=2,
        format_func="title",
        size="md",
        open_all=True,
    )

    sac.tags(
        [
            sac.Tag("测试阶段 0.0.1"),
            sac.Tag("🧊", color="cyan"),
        ]
    )

with st.container():
    loader = DolinphdbLoader()
    factor_names: List[str] = loader.get_factor_name_list

    if menu == "主页":
        overview()

    elif menu == "因子分析":
        com_ = FACTOR_FLOW.get("factor")
        com_.get("main")({"factor_names": factor_names, "loader": loader})

    elif menu == "因子对比":
        com_ = FACTOR_COMPARE.get("factor")
        com_.get("main")({"factor_names": factor_names, "loader": loader})


    elif menu == "因子看板":
        
        board_table()

    elif menu == "回测分析":
        sac.alert(
            label="Alert Message",
            description="暂未完成...",
            banner=True,
            icon="🚧",
            closable=False,
        )
        local_json_lottie("page/img/deer.json", height=500)

    elif menu == "设置":
        account_settings()

    else:
        sac.alert(
            label="Alert Message",
            description="暂未完成...",
            banner=True,
            icon="🚧",
            closable=False,
        )
