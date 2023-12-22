"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-09-15 14:08:43
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-20 13:27:26
FilePath: 
Description: 
"""
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
