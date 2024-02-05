"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-09-15 14:08:43
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-20 13:27:26
FilePath: 
Description: 
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from .utils import local_json_lottie


def redirect(index=0):
    st.session_state["index"] = index


def overview():
    local_json_lottie("page/img/home.json", height=250)
    st.subheader("介绍", False)
    st.markdown(
        """
    :heart: Alphlens-Streamlit.   
    :heart: 动态的因子分析面板.     
    """
    )


#########################################################################################################
#                                         设置页面
#########################################################################################################

def check_csv_file(file_path):
    # 检查file_path下是否存在factor.csv和price.csv
    if isinstance(file_path, str):
        file_path = Path(file_path).absolute()
    for file_name in ("factor.csv", "price.csv"):
        if not Path(file_path, file_name).exists():
            msg: str = f"""{file_name}不在对应的{str(file_path)}目录下,请上传csv文件;
            或直接将csv放到对应的{str(file_path)}目录下.
            """
            st.error(
                msg,
                icon="📄",
            )
            return False
    return True


def update_csv():
    uploaded_files = st.file_uploader("上传csv文件", type="csv", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        this_csv_path = Path("..data").absolute() / uploaded_file.name
        pd.read_csv(uploaded_file).to_csv(this_csv_path, index=False)
        st.write(f"filename:{uploaded_file.name},储存至:{str(this_csv_path)}")


def account_settings():
    with st.container(border=True):
        st.header("参数设置")

        st.subheader("数据加载设置")

        loader_type: bool = st.toggle("使用DolphinDB数据库", value=st.session_state["db_or_csv"])
        st.session_state["db_or_csv"] = loader_type
        
        if not loader_type:
            path_str = st.text_input("请输入数据文件夹路径", value="./data")
            if not check_csv_file(path_str):
                update_csv()
            else:
                st.success("验证成功有对应csv文件")

        st.session_state["loader_type"] = loader_type

        st.subheader("回测数据使用设置")

        price_type: str = st.selectbox(
            "因子分组回测使用的未来期收益率使用下面的哪个价格进行计算？",
            ("open", "high", "low", "close", "avg_price"),
            index=4,
            placeholder="默认使用avg_price",
        )

        st.session_state["price_type"] = price_type

        st.divider()
        mst:str = f"""
        数据加载设置: {"**DolphinDB**" if loader_type else "**CSV**"};
        因子分组回测使用的未来期收益率使用 **{price_type}** 进行测试! :sunglasses:
        """
        st.write(mst)
