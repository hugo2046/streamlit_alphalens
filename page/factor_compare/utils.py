"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2024-01-08 17:01:40
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2024-01-08 21:01:11
FilePath: 
Description: 
"""
import re
from typing import Dict, List

import empyrical as ep
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from analyzer.streamlit_analyze import FactorAnalyzer


def merge_table(dfs: Dict, axis: int = 1) -> pd.DataFrame:
    new_df: List[pd.DataFrame] = []
    for k, df in dfs.items():
        sorted_columns: pd.Index = sorted(
            df.columns, key=lambda x: int(x.split("_")[1])
        )
        df: pd.DataFrame = df[sorted_columns]
        df.columns: pd.DataFrame = pd.MultiIndex.from_product([[k], df.columns])
        new_df.append(df)

    all_df: pd.DataFrame = (
        pd.concat(new_df, axis=axis).swaplevel(axis=axis).sort_index(axis=axis)
    )
    all_df.columns.names = ["Period:", "factorName:"]

    return all_df


def highlight_by_group(df: pd.DataFrame):
    max_cell_color: pd.DataFrame = (
        df.groupby(level=0, axis=1, group_keys=False)
        .apply(lambda df: df.apply(lambda x: x == np.max(x), axis=1))
        .apply(lambda x: np.where(x, 1, 0))
    )

    min_cell_color: pd.DataFrame = (
        df.groupby(level=0, axis=1, group_keys=False)
        .apply(lambda df: df.apply(lambda x: x == np.min(x), axis=1))
        .apply(lambda x: np.where(x, -1, 0))
    )

    cell_color: pd.DataFrame = (max_cell_color + min_cell_color).apply(
        lambda x: np.where(x == 1, "true", np.where(x == -1, "false", ""))
    )

    s = df.style
    cell_hover = {  # for row hover use <tr> instead of <td>
        "selector": "td:hover",
        "props": [("background-color", "#ffffb3")],
    }
    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;",
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #000066; color: white;",
    }
    s.set_table_styles(
        [
            cell_hover,
            index_names,
            headers,
            {"selector": "th.col_heading.level0", "props": "font-size: 1.5em;"},
            {"selector": ".true", "props": "background-color: #ffe6e6;"},
            {"selector": ".false", "props": "background-color: #e6ffe6;"},
        ]
    )
    s.set_td_classes(cell_color)
    s.format(precision=3)

    return s



def calc_metric(fa: FactorAnalyzer, method: str) -> pd.DataFrame:
    method_dict: Dict = {
        "factor_returns": ("calc_factor_returns", "cumulative_returns"),
        "top_bottom": ("top_down_returns", "top_down_cumulative_returns"),
    }

    if method == "factor_returns":
        factor_returns: pd.DataFrame = getattr(fa, method_dict[method][0])()
    else:
        factor_returns: pd.DataFrame = getattr(fa, method_dict[method][0])

    metric_df: pd.DataFrame = pd.DataFrame(
        columns=[
            "Annualvolatility",
            "CumReturn",
            "AnnualReturn",
            "MaxDrawdown",
            "SharpeRatio",
        ]
    )
    metric_df["Annualvolatility"] = factor_returns.apply(ep.annual_volatility)
    metric_df["CumReturn"] = getattr(fa, method_dict[method][1]).iloc[-1]
    metric_df["AnnualReturn"] = factor_returns.apply(ep.annual_return)
    metric_df["MaxDrawdown"] = factor_returns.apply(ep.max_drawdown)
    metric_df["SharpeRatio"] = factor_returns.apply(ep.sharpe_ratio)

    return metric_df


def get_factor_board(analyze_dict: Dict) -> pd.DataFrame:
    """
    从分析字典中计算和汇总各种指标。

    参数:
    - analyze_dict: 包含 FactorAnalyzer 对象的字典。

    返回:
    - 汇总的 DataFrame。
    """
    dfs:Dict = {}
    for k, fa in analyze_dict.items():
        # 获取信息比率和平均信息系数
        ic_frame:pd.DataFrame = fa.plot_information_table(make_pretty=False)
        ir:pd.DataFrame = ic_frame.loc["IR"].rename("IR")
        ic_avg:pd.DataFrame = ic_frame.loc["IC Mean"].rename("IC Mean")

        # 获取各种累计收益率和信息系数
        metrics:List = [
            fa.mean_return_by_quantile.iloc[0].rename("MinimumQuantileAverageReturn").T,
            fa.mean_return_by_quantile.iloc[-1]
            .rename("MaximumQuantileAverageReturn")
            .T,
            fa.cumulative_returns.apply(lambda x: [x.tolist()]).T.rename(
                columns={0: "CumulativeReturnsVeiws"}
            ),
            fa.top_down_cumulative_returns.apply(lambda x: [x.tolist()]).T.rename(
                columns={0: "TopDownCumulativeReturnsVeiws"}
            ),
            fa.ic.cumsum()
            .apply(lambda x: [x.tolist()])
            .T.rename(columns={0: "CumulativeInformationCoefficient"}),
        ]

        # 计算其他指标
        metric_df:pd.DataFrame = calc_metric(fa, "top_bottom")

        # 合并所有指标
        all_metrics:pd.DataFrame = pd.concat([metric_df, ir, ic_avg] + metrics, axis=1)
        dfs[k] = all_metrics

    # 合并所有因子的数据
    data:pd.DataFrame = pd.concat(dfs, names=["FactorName", "Period"])

    return data
