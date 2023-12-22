
"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-14 08:56:31
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-14 08:58:00
FilePath: 
Description: 
"""

from typing import Dict, List, Tuple, Union

import alphalens.performance as perf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats

from .plotly_utils import get_rgb_color

DECIMAL_TO_BPS = 10000


def plot_quantile_statistics_table(factor_data: pd.DataFrame) -> pd.DataFrame:
    quantile_stats = factor_data.groupby("factor_quantile").agg(
        ["min", "max", "mean", "std", "count"]
    )["factor"]
    quantile_stats["count %"] = (
        quantile_stats["count"] / quantile_stats["count"].sum() * 100.0
    )

    return quantile_stats


def plot_information_table(ic_data: pd.DataFrame) -> pd.DataFrame:
    ic_summary_table: pd.DataFrame = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    return ic_summary_table.T


def plot_returns_table(
    alpha_beta: pd.DataFrame,
    mean_ret_quantile: pd.DataFrame,
    mean_ret_spread_quantile: pd.DataFrame,
) -> pd.DataFrame:
    returns_table = pd.DataFrame()
    # UPDATE to returns_table.append(alpha_beta)
    returns_table: pd.DataFrame = pd.concat((returns_table, alpha_beta))
    returns_table.loc["Mean Period Wise Return Top Quantile (bps)"] = (
        mean_ret_quantile.iloc[-1] * DECIMAL_TO_BPS
    )
    returns_table.loc["Mean Period Wise Return Bottom Quantile (bps)"] = (
        mean_ret_quantile.iloc[0] * DECIMAL_TO_BPS
    )
    returns_table.loc["Mean Period Wise Spread (bps)"] = (
        mean_ret_spread_quantile.mean() * DECIMAL_TO_BPS
    )

    return returns_table


def plot_turnover_table(
    autocorrelation_data: pd.DataFrame, quantile_turnover: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    turnover_table = pd.DataFrame()

    for period in sorted(quantile_turnover.keys()):
        for quantile, p_data in quantile_turnover[period].items():
            turnover_table.loc[
                "Quantile {} Mean Turnover ".format(quantile), "{}D".format(period)
            ] = p_data.mean()
    auto_corr = pd.DataFrame()
    for period, p_data in autocorrelation_data.items():
        auto_corr.loc[
            "Mean Factor Rank Autocorrelation", "{}D".format(period)
        ] = p_data.mean()

    return turnover_table, auto_corr


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
            "text": "Mean Period Wise Return By Factor Quantile",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title="Mean Return (bps)",
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
            "text": "Period Wise Return By Factor Quantile",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title="Return (bps)",
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend_title_text="forward_periods",
    )

    return fig


def plot_cumulative_returns(
    factor_returns: Union[pd.DataFrame, pd.Series], period: str, title: str = None
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

    if isinstance(factor_returns, pd.Series):
        factor_returns: pd.DataFrame = factor_returns.to_frame(name=period)

    factor_returns: pd.DataFrame = perf.cumulative_returns(factor_returns)

    fig = go.Figure()

    for col, ser in factor_returns.items():
        fig.add_trace(
            go.Scatter(
                x=ser.index,
                y=ser.values,
                name=col,
                line=dict(
                    width=1.5, color="forestgreen"
                ),  # TODO:如果是多个line，需要调整颜色，原始代码中使用的绿色forestgreen #72b175
                hovertemplate="date: %{x:%Y%m%d} <br> CumulativeReturn: %{y:.4f} <extra></extra>",
            )
        )

    fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y-%m-%d")
    fig.update_layout(
        title={
            "text": (
                "Portfolio Cumulative Return ({} Fwd Period)".format(period)
                if title is None
                else title
            ),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title="Cumulative Returns",
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


def plot_cumulative_returns_by_quantile(
    quantile_returns: pd.Series,
    period: str,
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

    cum_ret: pd.DataFrame = ret_wide.apply(perf.cumulative_returns)
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
            "text": """Cumulative Return by Quantile({} Period Forward Return)""".format(
                period
            ),
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis=dict(
            title="Log Cumulative Returns",
            titlefont_size=16,
            tickfont_size=14,
            type="log",
        ),
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
    title: str = (
        f"Top Minus Bottom Quantile Mean Return ({periods} Period Forward Return)"
        if periods is not None
        else ""
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
            name="mean returns spread",
            line=dict(color="forestgreen", width=0.7),
            opacity=0.4,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mean_returns_spread.index,
            y=mean_returns_spread_bps_rolling,
            mode="lines",
            name="1 month moving avg",
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
        yaxis_title="Difference In Quantile Mean Return (bps)",
        yaxis=dict(range=[-ylim, ylim]),
        showlegend=True,
    )
    fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y-%m-%d")
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False)
    # 添加水平线
    fig.add_hline(y=0.0, line=dict(color="black", width=1), opacity=0.8)

    return fig


def plot_ic_ts(ic: Union[pd.Series, pd.DataFrame]) -> List[go.Figure]:
    if isinstance(ic, pd.DataFrame):
        figs: List[go.Figure] = [plot_ic_ts(ser) for _, ser in ic.items()]

        return figs

    titles: str = "{} Period Forward Return Information Coefficient (IC)".format(
        ic.name
    )

    # 创建子图
    fig = go.Figure()

    # 原始 IC 数据
    fig.add_trace(
        go.Scatter(
            x=ic.index,
            y=ic.values,
            mode="lines",
            name="IC",
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
            name="1 month moving avg",
            line=dict(color="forestgreen", width=2),
            opacity=0.8,
        ),
    )

    # 水平线
    fig.add_hline(y=0.0, line=dict(color="black", width=1), opacity=0.8)

    # 添加文本
    mean_val = ic.mean()
    std_val = ic.std()
    fig.add_annotation(
        x=0.05,
        y=0.95,
        text=f"Mean {mean_val:.3f} \n Std. {std_val:.3f}",
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


def plot_ic_hist(ic: Union[pd.DataFrame, pd.Series]) -> List[go.Figure]:
    if isinstance(ic, pd.DataFrame):
        return [plot_ic_hist(ser) for _, ser in ic.items()]

    ic: pd.DataFrame = ic.fillna(0.0)
    period: str = ic.name
    # 计算合适的 bin 大小
    data_range: float = ic.max() - ic.min()
    bin_count: int = int(np.sqrt(len(ic)))  # 作为示例，使用数据点数量的平方根作为 bin 数量
    bin_size: float = data_range / bin_count

    # 创建直方图和 KDE
    fig = ff.create_distplot(
        [ic], ["IC"], bin_size=bin_size, show_hist=True, show_rug=False
    )

    # 设置图表布局
    fig.update_layout(
        title={
            "text": f"{period} Period IC",
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
    mean_val: float = ic.mean()
    std_val: float = ic.std()
    fig.add_annotation(
        x=0.05,
        y=0.95,
        text=f"Mean {mean_val:.3f} \n Std. {std_val:.3f}",
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
        dist_name: str = "Normal"
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name: str = "T"
    else:
        dist_name: str = "Theoretical"

    period_num: str = ic.name
    fig = go.Figure()

    # 计算 Q-Q 数据
    _plt_fig = sm.qqplot(ic.fillna(0.0), theoretical_dist, fit=True, line="45")
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
            "text": f"IC {dist_name} Dist. Q-Q",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        xaxis_title=f"{dist_name} Distribution Quantile",
        yaxis_title="Observed Quantile",
        showlegend=False,
    )

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
            "text": "Monthly Mean {} Period IC".format(periods_num),
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
            name="Top Quantile Turnover",
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
            name="Bottom Quantile Turnover",
            # opacity=0.8,
            line=dict(width=0.8, color="#e4c188"),
        )
    )

    # 设置图表布局
    fig.update_layout(
        title={
            "text": f"{period}D Period Top and Bottom Quantile Turnover",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis_title="Proportion Of Names New To Quantile",
        hovermode="x unified",
        xaxis_tickformat="%Y-%m-%d",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False)
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
            "text": f"{period}D Period Factor Rank Autocorrelation",
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        yaxis_title="Autocorrelation Coefficient",
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
        text=f"Mean {mean_val:.3f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    return fig
