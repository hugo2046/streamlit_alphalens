from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from alphalens import performance as perf
from alphalens import utils

from . import plotly_plotting as plotting


def plotting_by_streamlit(figs: go.Figure, use_container_width: bool = True):
    if not isinstance(figs, (list, tuple)):
        figs: List[go.Figure] = [figs]
    for fig in figs:
        st.plotly_chart(fig, use_container_width=use_container_width)


def create_full_tear_sheet(
    factor_data,
    long_short=True,
    group_neutral=False,
):
    tab1, tab2, tab3 = st.tabs(
        ["Returns Analysis", "Information Analysis", "Turnover Analysis"]
    )

    with tab1:
        create_returns_tear_sheet(factor_data, long_short, group_neutral)
    with tab2:
        create_information_tear_sheet(factor_data, group_neutral)
    with tab3:
        create_turnover_tear_sheet(factor_data)


def create_returns_tear_sheet(
    factor_data: pd.DataFrame, long_short: bool = True, group_neutral: bool = False
):
    st.markdown("Quantiles Statistics")
    st.dataframe(plotting.plot_quantile_statistics_table(factor_data))

    factor_returns: pd.DataFrame = perf.factor_returns(
        factor_data, long_short, group_neutral
    )

    mean_quant_ret, _ = perf.mean_return_by_quantile(
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret: pd.DataFrame = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate: pd.DataFrame = mean_quant_ret_bydate.apply(
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily: pd.DataFrame = std_quant_daily.apply(
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta: pd.DataFrame = perf.factor_alpha_beta(
        factor_data, factor_returns, long_short, group_neutral
    )

    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    returns_table: pd.DataFrame = plotting.plot_returns_table(
        alpha_beta, mean_quant_rateret, mean_ret_spread_quant
    )

    st.markdown("Returns Analysis")
    st.dataframe(returns_table)

    st.plotly_chart(
        plotting.plot_quantile_returns_bar(
            mean_quant_rateret,
            by_group=False,
            ylim_percentiles=None,
        ),
        use_container_width=True,
    )

    st.plotly_chart(
        plotting.plot_quantile_returns_violin(
            mean_quant_rateret_bydate, ylim_percentiles=(1, 99)
        ),
        use_container_width=True,
    )

    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        st.warning(
            "'freq' not set in factor_data index: assuming business day", icon="⚠️"
        )

    # Compute cumulative returns from daily simple returns, if '1D'
    # returns are provided.
    if "1D" in factor_returns:
        title = (
            "Factor Weighted "
            + ("Group Neutral " if group_neutral else "")
            + ("Long/Short " if long_short else "")
            + "Portfolio Cumulative Return (1D Period)"
        )

        st.plotly_chart(
            plotting.plot_cumulative_returns(
                factor_returns["1D"], period="1D", title=title
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            plotting.plot_cumulative_returns_by_quantile(
                mean_quant_ret_bydate["1D"],
                period="1D",
            ),
            use_container_width=True,
        )

    figs: List[go.Figure] = plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
    )

    plotting_by_streamlit(figs, use_container_width=True)


def create_information_tear_sheet(
    factor_data: pd.DataFrame, group_neutral: bool = False
):
    ic: pd.DataFrame = perf.factor_information_coefficient(factor_data, group_neutral)

    st.markdown("Information Analysis")

    st.dataframe(plotting.plot_information_table(ic))

    figs = plotting.plot_ic_ts(ic)

    plotting_by_streamlit(plotting.plot_ic_ts(ic), use_container_width=True)

    ic_hist_figs: List[go.Figure] = plotting.plot_ic_hist(ic)
    ic_qq_figs: List[go.Figure] = plotting.plot_ic_qq(ic)

    if len(ic_hist_figs) != len(ic_qq_figs):
        raise ValueError("ic_hist_figs and ic_qq_figs must have the same length")

    fr_cols: int = len(ic.columns)

    for i in range(fr_cols):
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(ic_hist_figs[i], use_container_width=True)

        with col2:
            st.plotly_chart(ic_qq_figs[i], use_container_width=True)

    mean_monthly_ic = perf.mean_information_coefficient(
        factor_data,
        group_adjust=group_neutral,
        by_group=False,
        by_time="M",
    )
    plotting_by_streamlit(
        plotting.plot_monthly_ic_heatmap(mean_monthly_ic),
        use_container_width=True,
    )


def create_turnover_tear_sheet(factor_data: pd.DataFrame, turnover_periods=None):
    if turnover_periods is None:
        input_periods: np.ndarray = utils.get_forward_returns_columns(
            factor_data.columns,
            require_exact_day_multiple=True,
        ).values
        turnover_periods: List[int] = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods: List[int] = utils.timedelta_strings_to_integers(
            turnover_periods,
        )

    quantile_factor: pd.Series = factor_data["factor_quantile"]

    quantile_turnover: Dict = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation: pd.DataFrame = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    st.markdown("Turnover Analysis")

    turnover_table, auto_corr = plotting.plot_turnover_table(
        autocorrelation, quantile_turnover
    )

    st.dataframe(
        turnover_table,
        # use_container_width=True,
    )

    st.dataframe(
        auto_corr,
        # use_container_width=True,
    )

    for period in turnover_periods:
        if quantile_turnover[period].empty:
            continue
        st.plotly_chart(
            plotting.plot_top_bottom_quantile_turnover(
                quantile_turnover[period], period=period
            ),
            use_container_width=True,
        )

    for period in autocorrelation:
        if autocorrelation[period].empty:
            continue
        st.plotly_chart(
            plotting.plot_factor_rank_auto_correlation(
                autocorrelation[period], period=period
            ),
            use_container_width=True,
        )
