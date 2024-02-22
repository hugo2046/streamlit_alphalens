"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2024-01-04 10:12:34
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2024-01-04 12:38:31
FilePath: 
Description: 
"""

from __future__ import division, print_function

from typing import Dict, List, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns

# from matplotlib.ticker import ScalarFormatter
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.api import qqplot

# from .compat import rolling_mean
from .performance import cumulative_returns

# TODO:行业分析相关 plot_missing_events_distribution,plot_ic_by_group尚未实现
from .plot_utils import (
    AUTOCORR,
    AVGCUMRET,
    CUMICTS,
    CUMRET,
    CUMRETQ,
    EVENTSDIST,
    ICGROUP,
    ICHEATMAP,
    ICHIST,
    ICQQ,
    ICTS,
    MISSIINGEVENTSDIST,
    QRETURNBAR,
    QRETURNTS,
    QRETURNVIOLIN,
    TBTURNOVER,
    TDCUMRET,
    FACTORHIST,
    get_rgb_color,
)

# from .utils import convert_to_forward_returns_columns, ignore_warning

DECIMAL_TO_BPS = 10000


def pretty_comparison(df: pd.DataFrame):
    cell_hover: Dict = {  # for row hover use <tr> instead of <td>
        "selector": "td:hover",
        "props": [("background-color", "#d8d8d8"), ("color", "black")],
    }

    return (
        df.style.set_table_styles([cell_hover], overwrite=False)
        .highlight_max(axis=1, props="color:white;background-color:#cc0000")
        .highlight_min(axis=1, props="color:white;background-color:#6eb56e")
        .format(precision=3)
    )


def set_table_title(styler, title: str, props: Dict = None):
    if props is None:
        props = {"selector": "caption", "props": "font-size:1.5em;"}

    return styler.set_caption(title).set_table_styles([props], overwrite=False)


def make_background_gradient(df: pd.DataFrame, color: str = "red"):
    cell_hover: Dict = {  # for row hover use <tr> instead of <td>
        "selector": "td:hover",
        "props": [("background-color", "#d8d8d8"), ("color", "black")],
    }

    cm = sns.light_palette(color, as_cmap=True)

    return (
        df.style.background_gradient(cmap=cm)
        .format(precision=3)
        .set_table_styles([cell_hover], overwrite=False)
    )


def plot_ic_ts(ic: Union[pd.Series, pd.DataFrame]) -> List[go.Figure]:
    if isinstance(ic, pd.DataFrame):
        figs: List[go.Figure] = [plot_ic_ts(ser) for _, ser in ic.items()]

        return figs

    titles: str = ICTS.get("TITLE").format(ic.name.replace("_", " "))

    # 创建子图
    fig = go.Figure()

    # 原始 IC 数据
    fig.add_trace(
        go.Scatter(
            x=ic.index,
            y=ic.values,
            mode="lines",
            name=ICTS.get("LEGEND")[0],
            line=dict(color="steelblue", width=0.7),
            opacity=0.7,
        )
    )

    # 移动平均线
    fig.add_trace(
        go.Scatter(
            x=ic.index,
            y=ic.rolling(window=22).mean(),
            mode="lines",
            name=ICTS.get("LEGEND")[1],
            line=dict(color="forestgreen", width=2),
            opacity=0.8,
        ),
    )

    # 水平线
    fig.add_hline(y=0.0, line=dict(color="black", width=1), opacity=0.8)

    # 添加文本

    fig.add_annotation(
        x=0.05,
        y=0.95,
        text=ICTS.get("TEXT").format(ic.mean(), ic.std()),
        showarrow=False,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
    )

    # 更新布局
    fig.update_layout(
        title={
            "text": titles,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title="IC",
            titlefont_size=16,
            tickfont_size=14,
        ),
        showlegend=True,
        hovermode="x unified",
        xaxis_tickformat="%Y-%m-%d",
    )

    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False)

    return fig


def plot_factor_hist(factor: Union[pd.Series, pd.DataFrame]) -> List[go.Figure]:

    if isinstance(factor, pd.DataFrame):
        return [plot_factor_hist(ser) for _, ser in factor.items()]

    # fig = ff.create_distplot([factor.values], [factor.name], show_hist=True,show_curve=show_curve, show_rug=show_rug)

    # fig.add_annotation(
    #     x=0.05,
    #     y=0.95,
    #     text=FACTORHIST.get("TEXT").format(factor.skew(), factor.kurt(),factor.mean()),
    #     showarrow=False,
    #     xref="paper",
    #     yref="paper",
    #     xanchor="left",
    #     yanchor="top",
    # )
    # fig.update_yaxes(showgrid=False)
    # fig.update_xaxes(showgrid=False)
    # return fig
    fig = px.histogram(
    factor.to_frame('factor'), x="factor", color_discrete_sequence=["#838df9"]
    )

    fig.add_annotation(
        x=0.05,
        y=0.95,
        text=FACTORHIST.get("TEXT").format(factor.skew(), factor.kurt(), factor.mean()),
        showarrow=False,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig


def plot_ic_hist(ic: Union[pd.DataFrame, pd.Series]) -> List[go.Figure]:
    if isinstance(ic, pd.DataFrame):
        return [plot_ic_hist(ser) for _, ser in ic.items()]

    ic: pd.DataFrame = ic.fillna(0.0)
    period: str = ic.name
    # 计算合适的 bin 大小
    data_range: float = ic.max() - ic.min()
    bin_count: int = int(
        np.sqrt(len(ic))
    )  # 作为示例，使用数据点数量的平方根作为 bin 数量
    bin_size: float = data_range / bin_count

    # 创建直方图和 KDE
    fig = ff.create_distplot(
        [ic], ["IC"], bin_size=bin_size, show_hist=True, show_rug=False
    )

    # 设置图表布局
    fig.update_layout(
        title={
            "text": ICHIST.get("TITLE") % period,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        xaxis_title="IC",
        xaxis_range=[-1, 1],
        showlegend=False,
    )

    # 添加平均值的垂直线
    fig.add_vline(x=ic.mean(), line=dict(color="red", dash="dash", width=2))

    # 添加文本

    fig.add_annotation(
        x=0.05,
        y=0.95,
        text=ICHIST.get("LEGEND").format(ic.mean(), ic.std()),
        showarrow=False,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        bgcolor="white",
    )

    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def plot_ic_qq(
    ic: Union[pd.Series, pd.DataFrame], theoretical_dist=stats.norm
) -> List[go.Figure]:
    if isinstance(ic, pd.DataFrame):
        return [plot_ic_qq(ser, theoretical_dist) for col, ser in ic.items()]

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name: str = ICQQ.get("NORM")
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name: str = ICQQ.get("T")
    else:
        dist_name: str = ICQQ.get("CUSTOM")

    period_num: str = ic.name
    fig = go.Figure()

    # 计算 Q-Q 数据
    _plt_fig = qqplot(ic.fillna(0.0), theoretical_dist, fit=True, line="45")
    plt.close(_plt_fig)
    qq_data = _plt_fig.gca().lines
    # 提取 Q-Q 数据点
    qq_x = qq_data[0].get_xdata()
    qq_y = qq_data[0].get_ydata()

    # 绘制 Q-Q 图
    fig.add_trace(
        go.Scatter(x=qq_x, y=qq_y, mode="markers", name=f"{period_num} Period")
    )

    # 绘制参考线
    line_x = qq_data[1].get_xdata()
    line_y = qq_data[1].get_ydata()
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line={"color": "red"},
            name="Reference Line",
        )
    )

    # 设置图表布局
    fig.update_layout(
        title={
            "text": ICQQ.get("TITLE").format(period_num, dist_name),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        xaxis_title=ICQQ.get("XLABEL").format(dist_name),
        yaxis_title=ICQQ.get("YLABEL"),
        showlegend=False,
    )

    return fig


def plot_returns_bar(mean_ret_by_q: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for col, ser in mean_ret_by_q.items():
        fig.add_trace(
            go.Bar(
                x=list(map(str, ser.index)),
                y=ser.values * DECIMAL_TO_BPS,
                name=col,
                hovertemplate="<br>".join(
                    [
                        "Mean Return: %{y:.2f}bps",
                    ]
                ),
            )
        )

    fig.update_yaxes(
        zeroline=True, zerolinewidth=1.5, zerolinecolor="black", showgrid=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_layout(
        title={
            "text": QRETURNBAR.get("TITLE"),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title=QRETURNBAR.get("YLABEL"),
            titlefont_size=16,
            tickfont_size=14,
        ),
    )

    return fig


def plot_quantile_returns_bar(
    mean_ret_by_q: pd.DataFrame,
    by_group=False,
    ylim_percentiles=None,
):
    mean_ret_by_q: pd.DataFrame = mean_ret_by_q.copy()
    mean_ret_by_q.columns = mean_ret_by_q.columns.map(
        lambda x: QRETURNBAR.get("COLUMN").format(x.replace("period_", ""))
    )

    # TODO:by_group =True时
    # if ylim_percentiles is not None:
    #     ymin = (
    #         np.nanpercentile(mean_ret_by_q.values, ylim_percentiles[0]) * DECIMAL_TO_BPS
    #     )
    #     ymax = (
    #         np.nanpercentile(mean_ret_by_q.values, ylim_percentiles[1]) * DECIMAL_TO_BPS
    #     )
    # else:
    #     ymin = None
    #     ymax = None

    # if by_group:
    #     num_group = len(mean_ret_by_q.index.get_level_values("group").unique())

    #     if ax is None:
    #         v_spaces = ((num_group - 1) // 2) + 1
    #         f, ax = plt.subplots(
    #             v_spaces, 2, sharex=False, sharey=True, figsize=(18, 6 * v_spaces)
    #         )
    #         ax = ax.flatten()

    #     for a, (sc, cor) in zip(ax, mean_ret_by_q.groupby(level="group")):
    #         (
    #             cor.xs(sc, level="group")
    #             .multiply(DECIMAL_TO_BPS)
    #             .plot(kind="bar", title=sc, ax=a)
    #         )

    #         a.set(xlabel="", ylabel="Mean Return (bps)", ylim=(ymin, ymax))

    #     if num_group < len(ax):
    #         ax[-1].set_visible(False)

    #     return ax

    # else:
    fig: go.Figure = plot_returns_bar(mean_ret_by_q)
    return fig


def plot_quantile_returns_violin(
    return_by_q: pd.DataFrame, ylim_percentiles: Tuple = None
):
    """
    绘制分位数收益小提琴图。

    参数：
    return_by_q (pd.DataFrame): 按分位数分组的收益数据框。
    ylim_percentiles (Tuple, 可选): y轴的百分位数范围。

    返回：
    fig (go.Figure): 绘制的小提琴图。
    """
    return_by_q = return_by_q.copy()

    return_by_q: pd.DataFrame = return_by_q.multiply(DECIMAL_TO_BPS)

    fig = go.Figure()

    for col, df in return_by_q.items():
        fig.add_trace(
            go.Violin(
                x=list(
                    map(
                        lambda x: f"Grop {x:0}",
                        df.index.get_level_values("factor_quantile"),
                    )
                ),
                y=df.values,
                legendgroup=col,
                name=col,
            )
        )

    if ylim_percentiles is not None:
        values: np.ndarray = return_by_q.values.reshape(-1)
        ymin = np.nanpercentile(values, ylim_percentiles[0])
        ymax = np.nanpercentile(values, ylim_percentiles[1])
        fig.update_yaxes(range=[ymin, ymax])

    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        y0=0,
        x1=1,
        y1=0,
        line=dict(
            color="black",
            width=1.5,
            dash="dash",
        ),
    )
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False)
    fig.update_traces(box_visible=True, meanline_visible=True)
    fig.update_layout(violinmode="group")

    fig.update_layout(
        title={
            "text": QRETURNVIOLIN.get("TITLE"),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title=QRETURNVIOLIN.get("YLABEL"),
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend_title_text="forward_periods",
    )

    return fig


def plot_mean_quantile_returns_spread_time_series(
    mean_returns_spread: Union[pd.Series, pd.DataFrame],
    std_err: Union[pd.Series, pd.DataFrame] = None,
    bandwidth: float = 1,
) -> List[go.Figure]:
    """
    绘制均值分位数收益差时间序列的图表。

    参数：
        - mean_returns_spread：均值分位数收益差的数据，可以是 pd.Series 或 pd.DataFrame。
        - std_err：标准误差的数据，可以是 pd.Series 或 pd.DataFrame，默认为 None。
        - bandwidth：标准误差带的宽度，默认为 1。

    返回：
        - figs：包含图表对象的列表。

    """

    if isinstance(mean_returns_spread, pd.DataFrame):
        # UPDATE to mean_returns_spread.iteritems()
        figs: List[go.Figure] = []
        for period, ser in mean_returns_spread.items():
            stdn: pd.Series = None if std_err is None else std_err[period]
            figs.append(
                plot_mean_quantile_returns_spread_time_series(
                    ser,
                    std_err=stdn,
                )
            )

        return figs

    # 创建 Plotly 图表对象
    fig = go.Figure()

    if mean_returns_spread.empty:
        return fig

    periods: str = mean_returns_spread.name
    title: str = QRETURNTS.get("TITLE").format(
        periods.replace("period_", "") if periods is not None else ""
    )

    mean_returns_spread_bps: pd.Series = mean_returns_spread * DECIMAL_TO_BPS
    mean_returns_spread_bps_rolling: pd.Series = mean_returns_spread_bps.rolling(
        window=22
    ).mean()

    # 添加原始数据和移动平均线
    fig.add_trace(
        go.Scatter(
            x=mean_returns_spread.index,
            y=mean_returns_spread_bps,
            mode="lines",
            name=QRETURNTS.get("LEGEND0").format(bandwidth),
            line=dict(color="forestgreen", width=0.7),
            opacity=0.4,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mean_returns_spread.index,
            y=mean_returns_spread_bps_rolling,
            mode="lines",
            name=QRETURNTS.get("LEGEND1"),
            line=dict(color="orangered"),
            opacity=0.7,
        )
    )

    # 添加标准误差带
    if std_err is not None:
        std_err_bps: pd.Series = std_err * DECIMAL_TO_BPS
        upper: pd.Series = mean_returns_spread_bps + (std_err_bps * bandwidth)
        lower: pd.Series = mean_returns_spread_bps - (std_err_bps * bandwidth)
        fig.add_trace(
            go.Scatter(
                x=mean_returns_spread_bps.index,
                y=upper.values + lower.values,
                fill="tonexty",
                fillcolor="rgba(70,130,180,0.3)",
                line_color="rgba(70,130,180,0.3)",
                showlegend=False,
            )
        )

    # 设置图表布局
    ylim: float = np.nanpercentile(abs(mean_returns_spread_bps.values), 95)
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=QRETURNTS.get("YLABEL"),
        yaxis=dict(range=[-ylim, ylim]),
        showlegend=True,
    )
    fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y-%m-%d")
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    # 添加水平线
    fig.add_hline(y=0.0, line=dict(color="black", width=1), opacity=0.8)

    return fig


def plot_factor_rank_auto_correlation(factor_autocorrelation, period=1):
    # 创建 Plotly 图表
    fig = go.Figure()

    # 添加因子自相关的折线
    fig.add_trace(
        go.Scatter(
            x=factor_autocorrelation.index,
            y=factor_autocorrelation,
            line=dict(color="#29698e"),
            mode="lines",
            name="Factor Rank Autocorrelation",
        )
    )

    # 设置图表布局
    fig.update_layout(
        title={
            "text": AUTOCORR.get("TITLE").format(period),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis_title=AUTOCORR.get("YLABEL").format(period),
        hovermode="x unified",
        xaxis_tickformat="%Y-%m-%d",
    )

    # 添加水平线
    fig.add_hline(y=0.0, line=dict(color="black", width=1, dash="dash"))

    # 添加文本注释
    mean_val = factor_autocorrelation.mean()
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=AUTOCORR.get("TEXT").format(mean_val),
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    return fig


def plot_top_bottom_quantile_turnover(
    quantile_turnover: pd.DataFrame, period: int = 1
) -> go.Figure:
    max_quantile: float = quantile_turnover.columns.max()
    min_quantile: float = quantile_turnover.columns.min()
    turnover = pd.DataFrame()
    turnover["top quantile turnover"] = quantile_turnover[max_quantile]
    turnover["bottom quantile turnover"] = quantile_turnover[min_quantile]

    # 创建 Plotly 图表
    fig = go.Figure()

    # 添加顶部分位数的折线
    fig.add_trace(
        go.Scatter(
            x=turnover.index,
            y=turnover["top quantile turnover"],
            mode="lines",
            name=TBTURNOVER.get("TURNOVER").format(max_quantile),
            # opacity=0.8,
            line=dict(width=0.8, color="#6aa8ce"),
        )
    )

    # 添加底部分位数的折线
    fig.add_trace(
        go.Scatter(
            x=turnover.index,
            y=turnover["bottom quantile turnover"],
            mode="lines",
            name=TBTURNOVER.get("TURNOVER").format(min_quantile),
            # opacity=0.8,
            line=dict(width=0.8, color="#e4c188"),
        )
    )

    # 设置图表布局
    fig.update_layout(
        title={
            "text": TBTURNOVER.get("TITLE").format(period),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis_title=TBTURNOVER.get("YLABEL"),
        hovermode="x unified",
        xaxis_tickformat="%Y-%m-%d",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def plot_monthly_ic_heatmap(
    mean_monthly_ic: Union[pd.Series, pd.DataFrame]
) -> List[go.Figure]:
    if isinstance(mean_monthly_ic, pd.DataFrame):
        return [plot_monthly_ic_heatmap(ser) for _, ser in mean_monthly_ic.items()]

    mean_monthly_ic_: pd.Series = mean_monthly_ic.copy()
    periods_num: int = mean_monthly_ic_.name
    new_index_year: pd.Index = mean_monthly_ic_.index.year
    new_index_month: pd.Index = mean_monthly_ic_.index.month

    mean_monthly_ic_.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month], names=["year", "month"]
    )

    mean_monthly_ic_: pd.DataFrame = mean_monthly_ic_.unstack()
    # 自定义颜色映射
    colorscale = [
        [0.0, "rgb(0,128,0)"],  # 低值 (绿色)
        [0.25, "rgb(128,224,128)"],  # 绿色到白色的过渡
        [0.5, "rgb(255,255,255)"],  # 中间值 (白色)
        [0.75, "rgb(255,128,128)"],  # 红色到白色的过渡
        [1.0, "rgb(255,0,0)"],  # 高值 (红色)
    ]

    # 创建热力图
    fig = go.Figure(
        data=go.Heatmap(
            z=mean_monthly_ic_.values,
            x=mean_monthly_ic_.columns,
            y=mean_monthly_ic_.index,
            text=mean_monthly_ic_.applymap(lambda perc: f"{perc:.2%}").values,
            texttemplate="%{text}",
            colorscale=colorscale,
            # colorbar=dict(title="IC"),
            zmid=0,  # 设置中心点为 0
            showscale=False,
        )
    )

    # 设置图表布局
    fig.update_layout(
        title={
            "text": ICHEATMAP.get("TITLE").format(periods_num),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        xaxis_nticks=12,
        yaxis=dict(type="category"),
        xaxis=dict(type="category"),
        yaxis_title="Year",
        xaxis_title="Month",
    )

    return fig


def plot_cumulative_returns(
    factor_returns: Union[pd.DataFrame, pd.Series],
    period: str,
    overlap=True,
    title: str = None,
):
    """
    绘制累计收益图。

    参数：
    factor_returns : Union[pd.DataFrame, pd.Series]
        因子收益数据，可以是DataFrame或Series类型。
    period : str
        前向期间，用于显示在图表标题中。
    title : str, optional
        图表标题，如果未提供，则默认为"Portfolio Cumulative Return (前向期间 Fwd Period)"。

    返回：
    fig : go.Figure
        绘制的累计收益图。
    """

    # if isinstance(factor_returns, pd.Series):
    #     factor_returns: pd.DataFrame = factor_returns.to_frame(name=period)

    overlapping_period = period if overlap else 1
    factor_returns: pd.Series = cumulative_returns(factor_returns, overlapping_period)

    fig = go.Figure()

    # for col, ser in factor_returns.items():
    fig.add_trace(
        go.Scatter(
            x=factor_returns.index,
            y=factor_returns.values,
            # name=col,
            line=dict(
                width=1.5, color="forestgreen"
            ),  # TODO:如果是多个line，需要调整颜色，原始代码中使用的绿色forestgreen #72b175
            hovertemplate="date: %{x:%Y%m%d} <br> CumulativeReturn: %{y:.4f} <extra></extra>",
        )
    )

    fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y-%m-%d")
    fig.update_layout(
        title={
            "text": (CUMRET.get("TITLE").format(period) if title is None else title),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title=CUMRET.get("YLABEL"),
            titlefont_size=16,
            tickfont_size=14,
        ),
    )
    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        y0=1,
        x1=1,
        y1=1,
        line=dict(
            color="black",
            width=1.5,
            dash="dash",
        ),
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)

    return fig


def plot_top_down_cumulative_returns(
    factor_returns: pd.DataFrame, period: int = 1
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=factor_returns.index,
            y=factor_returns.values,
            mode="lines",
            hovertemplate="date: %{x:%Y%m%d} <br> CumulativeReturn: %{y:.4f} <extra></extra>",
            line=dict(color="forestgreen", width=3),
            opacity=0.6,
        )
    )

    fig.update_layout(
        title=TDCUMRET.get("TITLE").format(period),
        xaxis_title="",
        yaxis_title=TDCUMRET.get("YLABEL"),
        autosize=False,
        width=800,
        height=600,
        xaxis_tickformat="%Y-%m-%d",
        hovermode="x unified",
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )

    fig.add_shape(
        type="line",
        x0=factor_returns.index[0],
        y0=1,
        x1=factor_returns.index[-1],
        y1=1,
        line=dict(color="Black", width=1),
    )

    return fig


def plot_cumulative_returns_by_quantile(
    quantile_returns: pd.Series, period: int, overlap: bool = True
):
    """
    绘制按分位数累积收益图表。

    参数：
    quantile_returns (pd.Series): 分位数收益数据，必须是一个 pd.Series 对象。
    period (str): 周期字符串。

    返回：
    fig (go.Figure): 绘制的图表对象。
    """
    if not isinstance(quantile_returns, pd.Series):
        raise ValueError("quantile_returns must be a pd.Series")

    ret_wide: pd.DataFrame = quantile_returns.unstack("factor_quantile")
    overlapping_period = period if overlap else 1
    cum_ret: pd.DataFrame = ret_wide.apply(
        cumulative_returns, args=(overlapping_period,)
    )
    # we want negative quantiles as 'red'
    cum_ret: pd.DataFrame = cum_ret.loc[:, ::-1]

    fig = go.Figure()

    for i, col in enumerate(cum_ret.columns):
        fig.add_trace(
            go.Scatter(
                x=cum_ret.index,
                y=cum_ret[col],
                mode="lines",
                name=col,
                hovertext=[col] * len(cum_ret),
                # log_y=True,
                line=dict(width=2, color=get_rgb_color(i, len(cum_ret.columns))),
                hovertemplate="Group %{hovertext} date:%{x:%Y-%m-%d} Cum:%{y:.4f} <extra></extra>",
            )
        )

    fig.add_shape(
        type="line",
        xref="paper",
        yref="y",
        x0=0,
        y0=1,
        x1=1,
        y1=1,
        line=dict(
            color="black",
            width=1.5,
            dash="dot",
        ),
    )
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y-%m-%d")
    fig.update_layout(
        title={
            "text": CUMRETQ.get("TITLE").format(period),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title=CUMRETQ.get("YLABEL"),
            titlefont_size=16,
            tickfont_size=14,
            type="log",
        ),
    )
    return fig


def plot_average_cumulative_return(
    avg_cumulative_returns: pd.DataFrame,
    std_bar: bool = False,
    periods_before: str = "",
    periods_after: str = "",
) -> go.Figure:
    """
    绘制平均累积收益图。

    参数：
    - avg_cumulative_returns：包含平均累积收益数据的DataFrame。
    - std_bar：是否显示标准差条形图，默认为False。
    - periods_before：在标题中显示的前置期数字符串，默认为空字符串。
    - periods_after：在标题中显示的后置期数字符串，默认为空字符串。

    返回：
    - go.Figure：绘制的平均累积收益图。
    """
    fig = go.Figure()
    quantiles = len(avg_cumulative_returns.index.levels[0].unique())
    palette: List[Tuple] = [cm.RdYlGn_r(i) for i in np.linspace(0, 1, quantiles)]
    palette: List[str] = [
        "rgb" + str((int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))
        for c in palette
    ]

    for i, (quantile, q_ret) in enumerate(
        avg_cumulative_returns.groupby(level="factor_quantile")
    ):
        mean = q_ret.loc[(quantile, "mean")]
        mean.name = AVGCUMRET.get("COLUMN").format(quantile)

        if std_bar:
            std = q_ret.loc[(quantile, "std")]
            fig.add_trace(
                go.Scatter(
                    x=mean.index,
                    y=mean,
                    mode="lines",
                    name=mean.name,
                    line=dict(color=palette[i % len(palette)], width=2),
                    error_y=dict(
                        type="data",
                        symmetric=True,
                        array=std,
                        color=palette[i % len(palette)],
                        thickness=1.5,
                        width=3,
                    ),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=mean.index,
                    y=mean,
                    mode="lines",
                    name=mean.name,
                    line=dict(color=palette[i % len(palette)], width=2),
                )
            )

    fig.update_layout(
        title=AVGCUMRET.get("YLABEL").format(periods_before, periods_after),
        xaxis_title=AVGCUMRET.get("XLABEL"),
        yaxis_title=AVGCUMRET.get("YLABEL"),
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )

    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="black")

    return fig


def plot_average_cumulative_return_by_quantile(
    avg_cumulative_returns: pd.DataFrame, std_bar: bool = False
) -> go.Figure:
    """
    绘制按分位数分组的平均累积收益图。

    参数：
    - avg_cumulative_returns：包含平均累积收益数据的DataFrame。
    - std_bar：是否添加标准差的误差条，默认为False。

    返回：
    - go.Figure：绘制的图形对象。
    """
    # 创建一个包含多个子图的图形

    quantiles = len(avg_cumulative_returns.index.levels[0].unique())
    palette: List[Tuple] = [cm.RdYlGn_r(i) for i in np.linspace(0, 1, quantiles)]
    palette: List[str] = [
        "rgb" + str((int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))
        for c in palette
    ]
    v_spaces = ((quantiles - 1) // 2) + 1

    # 创建一个包含多个子图的图形
    fig = make_subplots(rows=v_spaces, cols=2)

    for i, (quantile, q_ret) in enumerate(
        avg_cumulative_returns.groupby(level="factor_quantile")
    ):
        mean = q_ret.loc[(quantile, "mean")]
        mean.name = AVGCUMRET.get("COLUMN").format(quantile)

        # 计算子图的位置
        row = i // 2 + 1
        col = i % 2 + 1

        # 添加平均值的线到子图
        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=mean,
                mode="lines",
                name=mean.name,
                line=dict(color=palette[i]),
            ),
            row=row,
            col=col,
        )

        # 添加标准差的误差条到子图
        if std_bar:
            std = q_ret.loc[(quantile, "std")]
            fig.add_trace(
                go.Scatter(
                    x=std.index,
                    y=mean,
                    error_y=dict(type="data", array=std, color=palette[i]),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        # 添加垂直线到子图
        fig.add_vline(
            x=0, line_width=1.5, line_dash="dash", line_color="black", row=row, col=col
        )
        # 设置y轴标签
        fig.update_yaxes(title_text=AVGCUMRET.get("YLABEL"), row=row, col=col)
        # 添加图例
        fig.update_layout(showlegend=True)

    fig.update_layout(autosize=False, width=18 * 60, height=6 * 60 * v_spaces)
    # 显示图形
    return fig


def plot_quantile_average_cumulative_return(
    avg_cumulative_returns: pd.DataFrame,
    by_quantile: bool = False,
    std_bar: bool = False,
    periods_before: str = "",
    periods_after: str = "",
):
    """
    绘制分位数平均累积收益图。

    参数：
    - avg_cumulative_returns: 平均累积收益数据框
    - by_quantile: 是否按分位数绘制图形，默认为False
    - std_bar: 是否显示标准差条形图，默认为False
    - periods_before: 绘制图形前的时间段，默认为空
    - periods_after: 绘制图形后的时间段，默认为空

    返回：
    - fig: 绘制的图形对象
    """
    avg_cumulative_returns: pd.DataFrame = avg_cumulative_returns.multiply(
        DECIMAL_TO_BPS
    )

    if by_quantile:
        fig = plot_average_cumulative_return_by_quantile(
            avg_cumulative_returns, std_bar
        )

    else:
        fig = plot_average_cumulative_return(
            avg_cumulative_returns, std_bar, periods_before, periods_after
        )

    return fig


def plot_events_distribution(
    events: pd.DataFrame, num_days: int = 5, full_dates: pd.DatetimeIndex = None
) -> go.Figure:
    """
    绘制事件分布图。

    参数：
    events (pd.DataFrame): 包含事件数据的DataFrame。
    num_days (int): 分组的天数，默认为5。
    full_dates (pd.DatetimeIndex): 完整的日期索引，默认为None。

    返回：
    go.Figure: 绘制的事件分布图。
    """
    if full_dates is None:
        full_dates = events.index.get_level_values("date").unique()

    group = pd.Series(range(len(full_dates)), index=full_dates) // num_days
    grouper_label = group.drop_duplicates()
    grouper = group.reindex(events.index.get_level_values("date"))

    count = events.groupby(grouper.values).count()
    count = count.reindex(grouper_label.values, fill_value=0)
    count.index = grouper_label.index

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=count.index,
            y=count.values,
            marker_color="rgb(55, 83, 109)",
        )
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    fig.update_layout(
        title={
            "text": EVENTSDIST.get("TITLE"),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        xaxis_title=EVENTSDIST.get("XLABEL"),
        yaxis_title=EVENTSDIST.get("YLABEL"),
        xaxis_tickformat="%Y-%m-%d",
    )

    return fig


def plot_cumulative_ic_ts(ic: pd.DataFrame) -> go.Figure:
    if not isinstance(ic, pd.DataFrame):
        raise ValueError("ic must be pd.DataFrame")
    cumic: pd.DataFrame = ic.cumsum()
    sorted_columns = sorted(cumic.columns, key=lambda x: int(x.split("_")[1]))
    cumic: pd.DataFrame = cumic[sorted_columns]
    fig = go.Figure()

    for i, (col, ser) in enumerate(cumic.items()):
        fig.add_trace(
            go.Scatter(
                x=ser.index,
                y=ser.values,
                hovertemplate="<br>Date: %{x|%Y-%m-%d}<br>Cum IC:%{y:.3f}",
                name=col,
                line=dict(width=2, color=get_rgb_color(i, len(cumic.columns))),
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title={
            "text": CUMICTS.get("TITLE"),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title="Cumulative IC",
            titlefont_size=16,
            tickfont_size=14,
        ),
        xaxis_tickformat="%Y-%m-%d",
        hovermode="x unified",
    )
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig
