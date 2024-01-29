"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2024-01-16 14:29:46
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2024-01-29 10:21:44
FilePath: 
Description: 
"""

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, JsCode
import streamlit_antd_components as sac
from ..utils import local_json_lottie

# Sparkline的tooltip渲染器
lineTooltipRenderer = JsCode(
    """
    function toolipRenderer(params) {
    const {xValue,yValue,context} = params;
    return `<div class='sparkline-tooltip' style='background-color: rgb(78,78,255); color: white; opacity: 0.7;'>
            <div class='tooltip-title'>${context.data.FactorName}</div>
            <div class='tooltip-content'>
            <div>value:${yValue.toFixed(2)}</div>
            <div>date:${xValue}</div>
            </div>`;}
    """
)

columnsTooltipRenderer = JsCode(
    """
    function toolipRenderer(params) {
    const {xValue,yValue,context} = params;
    return `<div class='sparkline-tooltip' style='background-color: rgb(78,78,255); color: white; opacity: 0.7;'>
            <div class='tooltip-title'>${context.data.FactorName}</div>
            <div class='tooltip-content'>
            <div>value:${(yValue*100).toFixed(2)}%</div>
            <div>group:${xValue}</div>
            </div>`;}
    """
)

# Sparkline的column Point of interest渲染器
columnFormatter = JsCode(
    """
    function columnFormatter(params){
        const {yValue} = params;

        return {
            // if yValue is negative, the column should be dark red, otherwise it should be purple
            fill: yValue < 0 ? 'green' : '#a90001',
            stroke: yValue < 0 ? 'green' : '#a90001'
        }
    }
    """
)

# Sparkline的line Point of interest渲染器
lineMarkerFormatter = JsCode(
    """
    function lineMarkerFormatter(params){
    const { last } = params;

    return {
        size: last ? 3 : 0,
        fill: last ? '#c24340' : '#98c36d',
        stroke: last ? '#c24340' : '#98c36d'
        }        
    }
    """
)

# 单元格格式化器
numberFormatter = JsCode(
    """
    function numberFormatter(params) {
    return params.value ? params.value.toFixed(3) : ''; 
    }
    """
)

percentageFormatter = JsCode(
    """
    function percentageFormatter(params) {
    if (params.value !== null && params.value !== undefined) {
        // 将值转换为百分数并保留两位小数
        return (params.value * 100).toFixed(2) + '%';
    } else {
        return '';
    }
    }"""
)

cellStyle = JsCode(
    """function cellStyle(params) {
    if (params.value > 0) {
        // 如果值大于0，字体颜色设为红色
        return { color: 'red' };
    } else if (params.value < 0) {
        // 如果值小于0，字体颜色设为绿色
        return { color: 'green' };
    } else {
        // 如果值等于0，使用默认样式
        return null;
    }
    }"""
)


def show_factor_board_table(table: pd.DataFrame) -> None:
    """
    小数:numberFormatter-三位小数
    百分数:percentageFormatter
    cellStyle:大于0为红色 小于0为绿色

    """

    gridOptions = {
        # 'domLayout': 'normal',
        "columnDefs": [
            {"field": "FactorName", "minWidth": 150, "suppressMovable": "true"},
            {
                "headerName": "Factor Return",
                "children": [
                    {
                        "field": "AnnualVolatility",
                        "minWidth": 50,
                        "cellDataType": "number",
                        "suppressMovable": "true",
                        "valueFormatter": percentageFormatter,
                        "cellStyle": cellStyle,
                        "sortable": "true",
                    },
                    {
                        "field": "CumReturn",
                        "minWidth": 80,
                        "cellDataType": "number",
                        "suppressMovable": "true",
                        "sortable": "true",
                        "valueFormatter": percentageFormatter,
                        "cellStyle": cellStyle,
                    },
                    {
                        "field": "AnnualReturn",
                        "minWidth": 80,
                        "cellDataType": "number",
                        "suppressMovable": "true",
                        "sortable": "true",
                        "valueFormatter": percentageFormatter,
                        "cellStyle": cellStyle,
                    },
                    {
                        "field": "MaxDrawdown",
                        "minWidth": 80,
                        "cellDataType": "number",
                        "suppressMovable": "true",
                        "sortable": "true",
                        "valueFormatter": percentageFormatter,
                        "cellStyle": cellStyle,
                    },
                    {
                        "field": "SharpeRatio",
                        "minWidth": 50,
                        "cellDataType": "number",
                        "suppressMovable": "true",
                        "sortable": "true",
                        "valueFormatter": numberFormatter,
                        "cellStyle": cellStyle,
                    },
                ],
            },
            {
                "headerName": "Factor Metrics",
                "children": [
                    {
                        "field": "IR",
                        "minWidth": 40,
                        "cellDataType": "number",
                        "suppressMovable": "true",
                        "sortable": "true",
                        "valueFormatter": numberFormatter,
                        "cellStyle": cellStyle,
                    },
                    {
                        "field": "IC Mean",
                        "minWidth": 40,
                        "cellDataType": "number",
                        "suppressMovable": "true",
                        "sortable": "true",
                        "valueFormatter": numberFormatter,
                        "cellStyle": cellStyle,
                    },
                ],
            },
            {
                "headerName": "Trend",
                "children": [
                    {
                        "field": "MeanReturnByQuantile",
                        "minWidth": 150,
                        "cellRenderer": "agSparklineCellRenderer",
                        "suppressMovable": "true",
                        "cellRendererParams": {
                            "sparklineOptions": {
                                "type": "column",
                                "column": {"width": 2, "stroke": "rgb(79, 129, 189)"},
                                "axis": {"type": "category"},
                                "formatter": columnFormatter,
                                "tooltip": {
                                    "renderer": columnsTooltipRenderer,
                                },
                            },
                        },
                    },
                    {
                        "field": "CumulativeReturnsVeiws",
                        "minWidth": 250,
                        "cellRenderer": "agSparklineCellRenderer",
                        "suppressMovable": "true",
                        "cellRendererParams": {
                            "sparklineOptions": {
                                "type": "area",
                                "line": {"stroke": "#ff5357", "strokeWidth": 1},
                                "fill": "rgba(231,0,0, 0.3)",
                                "axis": {
                                    "type": "category",
                                    "stroke": "rgb(204, 204, 235)",
                                },
                                "tooltip": {
                                    "renderer": lineTooltipRenderer,
                                },
                                "padding": {
                                    "top": 5,
                                    "right": 5,
                                    "bottom": 5,
                                    "left": 5,
                                },
                                "marker": {
                                    "formatter": lineMarkerFormatter,
                                },
                            },
                        },
                    },
                    {
                        "field": "TopDownCumulativeReturnsVeiws",
                        "minWidth": 250,
                        "cellRenderer": "agSparklineCellRenderer",
                        "suppressMovable": "true",
                        "cellRendererParams": {
                            "sparklineOptions": {
                                "type": "area",
                                "tooltip": {
                                    "renderer": lineTooltipRenderer,
                                },
                                "line": {"stroke": "rgb(77,89,185)", "strokeWidth": 1},
                                "fill": "rgba(77,89,185, 0.3)",
                                "axis": {
                                    "type": "category",
                                    "stroke": "rgb(204, 204, 235)",
                                },
                            }
                        },
                    },
                    {
                        "field": "CumulativeInformationCoefficient",
                        "minWidth": 250,
                        "cellRenderer": "agSparklineCellRenderer",
                        "suppressMovable": "true",
                        "cellRendererParams": {
                            "sparklineOptions": {
                                "type": "area",
                                "tooltip": {
                                    "renderer": lineTooltipRenderer,
                                },
                                "line": {
                                    "stroke": "rgb(107, 174, 213)",
                                    "strokeWidth": 1,
                                },
                                "fill": "rgba(107, 174, 213, 0.3)",
                                "axis": {
                                    "type": "category",
                                    "stroke": "rgb(204, 204, 235)",
                                },
                            }
                        },
                    },
                ],
            },
        ]
    }

    AgGrid(
        table,
        gridOptions=gridOptions,
        width="100%",
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        # enable_enterprise_modules=True,
    )


def icon(x, class_=None):
    bi = f"bi bi-{x}"
    bi = bi + " " + class_ if class_ is not None else bi
    return f'<i class="{bi}"></i>'


def board_table():
    
    local_json_lottie("page/img/moon.json", height=200)
    st.header("因子看板")
    sac.alert(
        label="**Highlight**",
        description=f"""{icon('1-circle')} `选择周期`后,因子看板会自动更新.
    {icon('2-circle')} 因子看板数据在每日23:00更新.
    {icon('3-circle')} 主要用于监控因子的表现, 以及因子的趋势.
    {icon('4-circle')} 因子看板数据来源于`因子分析`模块.
    """,
        color="pink",
        icon="emoji-smile",
    )
    test_df: pd.DataFrame = pd.read_pickle("data/board_table.pkl").reset_index()
    period = st.selectbox("选择周期", ("period_1", "period_5", "period_10"), index=0)
    slice_df = test_df.query("Period == @period")
    show_factor_board_table(slice_df)
