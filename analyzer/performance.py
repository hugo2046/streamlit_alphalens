# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.api import OLS, add_constant

from .compat import rolling_apply
from .prepare import common_start_returns, demean_forward_returns
from .utils import get_forward_returns_columns


def get_factor_ttest(factor_ser: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """
    计算因子的t检验值。

    参数：
    factor_ser: pd.Series
        因子数据序列。
    returns: pd.DataFrame
        收益数据框。

    返回：
    pd.Series
        包含因子超额收益和t值的序列。
    """
    result = OLS(returns, add_constant(factor_ser)).fit()
    return pd.Series({"beat_factor": result.params[1], "tvalue": result.pvalues[1]})


def calculate_factor_ttest(factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算因子的t检验值。

    参数：
    factor_data (pd.DataFrame): 包含因子数据的DataFrame。

    返回：
    pd.DataFrame: 包含因子的t检验值的DataFrame。
    """

    def src_ttest(group: pd.DataFrame, forward_returns_columns: List[str]) -> pd.Series:
        factor_series: pd.Series = group["factor"]
        return group[forward_returns_columns].apply(
            lambda x: get_factor_ttest(factor_series, x)
        )

    forward_returns_columns: List[str] = get_forward_returns_columns(
        factor_data.columns
    )
    ttest_result: pd.DataFrame = factor_data.groupby(level="date").apply(
        lambda group: src_ttest(group, forward_returns_columns)
    )

    return ttest_result.unstack(level=1)


def factor_information_coefficient(
    factor_data, group_adjust=False, by_group=False, method=stats.spearmanr
):
    """
    通过因子值与因子远期收益计算信息系数(IC).

    参数
    ----------
    factor_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
    group_adjust : bool
        是否使用分组去均值后的因子远期收益计算 IC.
    by_group : bool
        是否分组计算 IC.
    Returns
    -------
    ic : pd.DataFrame
        因子信息系数(IC).
    """

    def src_ic(group):
        f = group["factor"]
        _ic = group[get_forward_returns_columns(factor_data.columns)].apply(
            lambda x: method(x, f)[0]
        )
        return _ic

    factor_data = factor_data.copy()

    grouper = [factor_data.index.get_level_values("date")]

    if group_adjust:
        factor_data = demean_forward_returns(factor_data, grouper + ["group"])
    if by_group:
        grouper.append("group")

    with np.errstate(divide="ignore", invalid="ignore"):
        ic = factor_data.groupby(grouper, group_keys=False).apply(src_ic)

    return ic


def mean_information_coefficient(
    factor_data,
    group_adjust=False,
    by_group=False,
    by_time=None,
    method=stats.spearmanr,
):
    """
    根据不同分组求因子 IC 均值.

    参数
    ----------
    factor_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
    group_adjust : bool
        是否使用分组去均值后的因子远期收益计算 IC.
    by_group : bool
        是否分组计算 IC.
    by_time : str (pd time_rule), optional
        根据相应的时间频率计算 IC 均值
        时间频率参见 http://pandas.pydata.org/pandas-docs/stable/timeseries.html

    返回值
    -------
    ic : pd.DataFrame
        根据不同分组求出的因子 IC 均值序列
    """

    ic = factor_information_coefficient(
        factor_data, group_adjust, by_group, method=method
    )

    grouper = []
    if by_time is not None:
        grouper.append(pd.Grouper(freq=by_time))
    if by_group:
        grouper.append("group")

    if len(grouper) == 0:
        ic = ic.mean()

    else:
        ic = (
            ic.reset_index().set_index("date").groupby(grouper, group_keys=False).mean()
        )

    return ic


def factor_returns(factor_data, demeaned=True, group_adjust=False):
    """
    计算按因子值加权的投资组合的收益
    权重为去均值的因子除以其绝对值之和 (实现总杠杆率为1).

    参数
    ----------
    factor_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
    demeaned : bool
        因子分析是否基于一个多空组合? 如果是 True, 则计算权重时因子值需要去均值
    group_adjust : bool
        因子分析是否基于一个分组(行业)中性的组合?
        如果是 True, 则计算权重时因子值需要根据分组和日期去均值

    返回值
    -------
    returns : pd.DataFrame
        每期零风险暴露的多空组合收益
    """

    def to_weights(group, is_long_short):
        if is_long_short:
            demeaned_vals = group - group.mean()
            return demeaned_vals / demeaned_vals.abs().sum()
        else:
            return group / group.abs().sum()

    grouper = [factor_data.index.get_level_values("date")]
    if group_adjust:
        grouper.append("group")

    weights = factor_data.groupby(grouper, group_keys=False)["factor"].apply(
        to_weights, demeaned
    )

    if group_adjust:
        weights = weights.groupby(level="date", group_keys=False).apply(
            to_weights, False
        )

    weighted_returns = factor_data[
        get_forward_returns_columns(factor_data.columns)
    ].multiply(weights, axis=0)

    returns = weighted_returns.groupby(level="date", group_keys=False).sum()

    return returns


def factor_alpha_beta(factor_data, demeaned=True, group_adjust=False):
    """
    计算因子的alpha(超额收益),
    alpha t-统计量 (alpha 显著性）和 beta(市场暴露).
    使用每期平均远期收益作为自变量(视为市场组合收益)
    因子值加权平均的远期收益作为因变量(视为因子收益), 进行回归.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
    demeaned : bool
        因子分析是否基于一个多空组合? 如果是 True, 则计算权重时因子值需要去均值
    group_adjust : bool
        因子分析是否基于一个分组(行业)中性的组合?
        如果是 True, 则计算权重时因子值需要根据分组和日期去均值
    Returns
    -------
    alpha_beta : pd.Series
        一个包含 alpha, beta, a t-统计量(alpha) 的序列
    """

    returns = factor_returns(factor_data, demeaned, group_adjust)

    universe_ret = (
        factor_data.groupby(level="date", group_keys=False)[
            get_forward_returns_columns(factor_data.columns)
        ]
        .mean()
        .loc[returns.index]
    )

    if isinstance(returns, pd.Series):
        returns.name = universe_ret.columns.values[0]
        returns = pd.DataFrame(returns)

    alpha_beta = pd.DataFrame()
    for period in returns.columns.values:
        x = universe_ret[period].values
        y = returns[period].values
        x = add_constant(x)
        period_int = int(period.replace("period_", ""))

        reg_fit = OLS(y, x).fit()
        alpha, beta = reg_fit.params

        alpha_beta.loc["Ann. alpha", period] = (1 + alpha) ** (250.0 / period_int) - 1
        alpha_beta.loc["beta", period] = beta

    return alpha_beta


def cumulative_returns(returns, period):
    """
    从'N 期'因子远期收益率构建累积收益
    当 'period' N 大于 1 时, 建立平均 N 个交错的投资组合 (在随后的时段 1,2,3，...，N 开始),
    每个 N 个周期重新调仓, 最后计算 N 个投资组合累积收益的均值。

    参数
    ----------
    returns: pd.Series
        N 期因子远期收益序列
    period: integer
        对应的因子远期收益时间跨度

    返回值
    -------
    pd.Series
        累积收益序列
    """

    returns = returns.fillna(0)

    if period == 1:
        return returns.add(1).cumprod()
    #
    # 构建 N 个交错的投资组合
    #

    def split_portfolio(ret, period):
        return pd.DataFrame(np.diag(ret))

    sub_portfolios = returns.groupby(
        np.arange(len(returns.index)) // period, axis=0
    ).apply(split_portfolio, period)
    sub_portfolios.index = returns.index

    #
    # 将 N 期收益转换为 1 期收益, 方便计算累积收益
    #

    def rate_of_returns(ret, period):
        return ((np.nansum(ret) + 1) ** (1.0 / period)) - 1

    sub_portfolios = rolling_apply(
        sub_portfolios,
        window=period,
        func=rate_of_returns,
        min_periods=1,
        args=(period,),
    )
    sub_portfolios = sub_portfolios.add(1).cumprod()

    #
    # 求 N 个投资组合累积收益均值
    #
    return sub_portfolios.mean(axis=1)


def weighted_mean_return(factor_data, grouper):
    """计算(年化)加权平均/标准差"""
    forward_returns_columns = get_forward_returns_columns(factor_data.columns)

    def agg(values, weights):
        count = len(values)
        average = np.average(values, weights=weights, axis=0)
        # Fast and numerically precise
        variance = (
            np.average((values - average) ** 2, weights=weights, axis=0)
            * count
            / max((count - 1), 1)
        )
        return pd.Series(
            [average, np.sqrt(variance), count], index=["mean", "std", "count"]
        )

    group_stats = factor_data.groupby(grouper)[
        forward_returns_columns.append(pd.Index(["weights"]))
    ].apply(
        lambda x: x[forward_returns_columns].apply(
            agg, weights=x["weights"].fillna(0.0).values
        )
    )

    mean_ret = group_stats.xs("mean", level=-1)

    std_error_ret = group_stats.xs("std", level=-1) / np.sqrt(
        group_stats.xs("count", level=-1)
    )

    return mean_ret, std_error_ret


def mean_return_by_quantile(
    factor_data, by_date=False, by_group=False, demeaned=True, group_adjust=False
):
    """
    计算各分位数的因子远期收益均值和标准差

    参数
    ----------
    factor_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
    by_date : bool
        如果为 True, 则按日期计算各分位数的因子远期收益均值
    by_group : bool
        如果为 True, 则分组计算各分位数的因子远期收益均值
    demeaned : bool
        是否按日期对因子远期收益去均值
    group_adjust : bool
        是否按日期和分组对因子远期收益去均值
    Returns
    -------
    mean_ret : pd.DataFrame
        各分位数因子远期收益均值
    std_error_ret : pd.DataFrame
        各分位数因子远期收益标准差
    """

    if group_adjust:
        grouper = [factor_data.index.get_level_values("date")] + ["group"]
        factor_data = demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    grouper = ["factor_quantile"]
    if by_date:
        grouper.append(factor_data.index.get_level_values("date"))

    if by_group:
        grouper.append("group")

    mean_ret, std_error_ret = weighted_mean_return(factor_data, grouper=grouper)

    return mean_ret, std_error_ret


def compute_mean_returns_spread(mean_returns, upper_quant, lower_quant, std_err=None):
    """
    计算两个分位数的平均收益之差, 和(可选)计算此差异的标准差

    参数
    ----------
    mean_returns : pd.DataFrame
        各分位数因子远期收益均值
    upper_quant : int
        作为被减数的因子分位数
    lower_quant : int
        作为减数的因子分位数
    std_err : pd.DataFrame
        各分位数因子远期收益标准差

    返回值
    -------
    mean_return_difference : pd.Series
        每期两个分位数的平均收益之差
    joint_std_err : pd.Series
        每期两个分位数的平均收益标准差之差
    """
    if isinstance(mean_returns.index, pd.MultiIndex):
        mean_return_difference = mean_returns.xs(
            upper_quant, level="factor_quantile"
        ) - mean_returns.xs(lower_quant, level="factor_quantile")
    else:
        mean_return_difference = (
            mean_returns.loc[upper_quant] - mean_returns.loc[lower_quant]
        )

    if isinstance(std_err.index, pd.MultiIndex):
        std1 = std_err.xs(upper_quant, level="factor_quantile")
        std2 = std_err.xs(lower_quant, level="factor_quantile")
    else:
        std1 = std_err.loc[upper_quant]
        std2 = std_err.loc[lower_quant]
    joint_std_err = np.sqrt(std1**2 + std2**2)

    return mean_return_difference, joint_std_err


def quantile_turnover(quantile_factor, quantile, period=1):
    """
    计算当期在分位数中的因子不在上一期分位数中的比例

    Parameters
    ----------
    quantile_factor : pd.Series
        包含日期, 资产, 和因子分位数的 DataFrame.
    quantile : int
        对应的分位数
    period: int, optional
        对应的因子远期收益时间跨度
    Returns
    -------
    quant_turnover : pd.Series
        每期对饮分位数因子的换手率
    """

    quant_names = quantile_factor[quantile_factor == quantile]
    quant_name_sets = quant_names.groupby(level=["date"], group_keys=False).apply(
        lambda x: set(x.index.get_level_values("asset"))
    )
    new_names = (quant_name_sets - quant_name_sets.shift(period)).dropna()
    quant_turnover = new_names.apply(lambda x: len(x)) / quant_name_sets.apply(
        lambda x: len(x)
    )
    quant_turnover.name = quantile
    return quant_turnover


def factor_autocorrelation(factor_data, period=1, rank=True):
    """
    计算指定时间跨度内平均因子排名/因子值的自相关性.
    该指标对于衡量因子的换手率非常有用.
    如果每个因子值在一个周期内随机变化，我们预计自相关为 0.

    参数
    ----------
    factor_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
    period: int, optional
        对应的因子远期收益时间跨度
    Returns
    -------
    autocorr : pd.Series
        滞后一期的因子自相关性
    """

    grouper = [factor_data.index.get_level_values("date")]

    if rank:
        ranks = factor_data.groupby(grouper, group_keys=False)[["factor"]].rank()
    else:
        ranks = factor_data[["factor"]]
    asset_factor_rank = ranks.reset_index().pivot(
        index="date", columns="asset", values="factor"
    )

    autocorr = asset_factor_rank.corrwith(asset_factor_rank.shift(period), axis=1)
    autocorr.name = period
    return autocorr


def average_cumulative_return_by_quantile(
    factor_data,
    prices,
    periods_before=10,
    periods_after=15,
    demeaned=True,
    group_adjust=False,
    by_group=False,
):
    """
    计算由 periods_before 到 periods_after 定义的周期范围内的因子分位数的平均累积收益率

    参数
    ----------
    factor_data : pd.DataFrame - MultiIndex
        一个 DataFrame, index 为日期 (level 0) 和资产(level 1) 的 MultiIndex,
        values 包括因子的值, 各期因子远期收益, 因子分位数,
        因子分组(可选), 因子权重(可选)
    prices : pd.DataFrame
        用于计算因子远期收益的价格数据
        columns 为资产, index 为 日期.
        价格数据必须覆盖因子分析时间段以及额外远期收益计算中的最大预期期数.
    periods_before : int, optional
        之前多少期
    periods_after  : int, optional
        之后多少期
    demeaned : bool, optional
        是否按日期对因子远期收益去均值
    group_adjust : bool
        是否按日期和分组对因子远期收益去均值
    by_group : bool
        如果为 True, 则分组计算各分位数的因子远期累积收益
    Returns
    -------
    cumulative returns and std deviation : pd.DataFrame
        一个 DataFrame, index 为分位数 (level 0) 和 'mean'/'std' (level 1) 的 MultiIndex
        columns 为取值范围从 -periods_before 到 periods_after 的整数
        如果 by_group=True, 则 index 会多出一个 'group' level
    """

    def cumulative_return(q_fact, demean_by):
        return common_start_returns(
            q_fact, prices, periods_before, periods_after, True, True, demean_by
        )

    def average_cumulative_return(q_fact, demean_by):
        q_returns = cumulative_return(q_fact, demean_by)
        return pd.DataFrame(
            {"mean": q_returns.mean(axis=1), "std": q_returns.std(axis=1)}
        ).T

    if by_group:
        returns_bygroup = []

        for group, g_data in factor_data.groupby("group"):
            g_fq = g_data["factor_quantile"]
            if group_adjust:
                demean_by = g_fq  # demeans at group level
            elif demeaned:
                demean_by = factor_data["factor_quantile"]  # demean by all
            else:
                demean_by = None
            #
            # Align cumulative return from different dates to the same index
            # then compute mean and std
            #
            avgcumret = g_fq.groupby(g_fq, group_keys=False).apply(
                average_cumulative_return, demean_by
            )
            avgcumret["group"] = group
            avgcumret.set_index("group", append=True, inplace=True)
            returns_bygroup.append(avgcumret)

        return pd.concat(returns_bygroup, axis=0)

    else:
        if group_adjust:
            all_returns = []
            for group, g_data in factor_data.groupby("group"):
                g_fq = g_data["factor_quantile"]
                avgcumret = g_fq.groupby(g_fq, group_keys=False).apply(
                    cumulative_return, g_fq
                )
                all_returns.append(avgcumret)
            q_returns = pd.concat(all_returns, axis=1)
            q_returns = pd.DataFrame(
                {"mean": q_returns.mean(axis=1), "std": q_returns.std(axis=1)}
            )
            return q_returns.unstack(level=1).stack(level=0)
        elif demeaned:
            fq = factor_data["factor_quantile"]
            return fq.groupby(fq).apply(average_cumulative_return, fq)
        else:
            fq = factor_data["factor_quantile"]
            return fq.groupby(fq).apply(average_cumulative_return, None)


DECIMAL_TO_BPS = 10000


def get_returns_table(
    alpha_beta: pd.DataFrame,
    mean_ret_quantile: pd.DataFrame,
    mean_ret_spread_quantile: pd.DataFrame,
) -> pd.DataFrame:
    returns_table: pd.DataFrame = alpha_beta.copy()
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


def get_quantile_turnover_mean(quantile_turnover: pd.DataFrame) -> pd.DataFrame:
    """
    计算分位数换手率的平均值。

    参数：
    quantile_turnover (pd.DataFrame): 包含分位数换手率的数据框。

    返回：
    pd.DataFrame: 包含每个时间段分位数换手率平均值的数据框。
    """

    def _get_quantile_avg(df: pd.DataFrame) -> pd.Series:
        """
        计算给定数据框的平均值。

        参数：
        df (pd.DataFrame): 需要计算平均值的数据框。

        返回：
        pd.Series: 包含平均值的序列。
        """
        avg_ser: pd.Series = df.mean()
        avg_ser.index = avg_ser.index.map(lambda x: f"Quantile {x} Mean Turnover")
        return avg_ser

    return pd.DataFrame(
        {period: _get_quantile_avg(df) for period, df in quantile_turnover.items()}
    )


def get_autocorrelation_mean(autocorrelation_data: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    计算自相关系数的均值。

    参数：
    autocorrelation_data (pd.DataFrame): 包含自相关系数的数据框。

    返回：
    pd.DataFrame: 包含均值的数据框。
    """
    return (
        autocorrelation_data.mean().to_frame(name="Mean Factor Rank Autocorrelation").T
    )


def get_turnover_table(
    autocorrelation_data: pd.DataFrame,
    quantile_turnover: pd.DataFrame,
) -> pd.DataFrame:
    return get_quantile_turnover_mean(quantile_turnover), get_autocorrelation_mean(
        autocorrelation_data
    )


def get_tstats_table(ttest_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算t统计量表格。

    参数：
    ttest_data (pd.DataFrame)：包含t统计数据的DataFrame。

    返回：
    pd.DataFrame：包含t统计量表格的DataFrame。
    """
    ttest_frame: pd.DataFrame = ttest_data.swaplevel(axis=1)
    tvalue: pd.DataFrame = ttest_frame.loc[:, "tvalue"]
    beat: pd.DataFrame = ttest_frame.loc[:, "beat_factor"]
    ttest_summary_table = pd.DataFrame()
    ttest_summary_table["|T| Mean"] = tvalue.abs().mean()
    ttest_summary_table["|T|>2 Rate"] = (tvalue.abs() > 2).sum() / tvalue.count()
    ttest_summary_table["T Mean/T Std."] = tvalue.mean() / tvalue.std()
    ttest_summary_table["Beat Mean"] = beat.mean()
    t_stat, p_value = stats.ttest_1samp(beat, 0)
    ttest_summary_table["t-stat(Beat factor)"] = t_stat
    ttest_summary_table["p-value(Beat factor)"] = p_value

    return ttest_summary_table.T


def get_information_table(ic_data:pd.DataFrame)->pd.DataFrame:
    """
    计算信息系数（IC）数据的统计指标。

    参数：
    ic_data：pd.DataFrame，包含IC数据的DataFrame。

    返回：
    ic_summary_table：pd.DataFrame，包含IC数据的统计指标的DataFrame。
    """
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["IR"] = ic_data.mean() / ic_data.std()
    ic_summary_table["IC>0 Rate"] =  (ic_data>0).sum() / ic_data.count()
    ic_summary_table["|IC|>0.02 Rate"] = (ic_data.abs()>0.02).sum() / ic_data.count()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    return ic_summary_table.T


def get_quantile_statistics_table(factor_data):
    """
    获取分位数统计表。

    参数：
    factor_data (pandas.DataFrame): 包含因子数据的数据帧。

    返回：
    pandas.DataFrame: 分位数统计表，包含最小值、最大值、均值、标准差、计数和计数百分比。

    """
    quantile_stats = factor_data.groupby('factor_quantile') \
        .agg(['min', 'max', 'mean', 'std', 'count'])['factor']
    quantile_stats['count %'] = quantile_stats['count'] \
        / quantile_stats['count'].sum() * 100.

    return quantile_stats