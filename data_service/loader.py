"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-14 22:39:48
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-14 22:56:04
FilePath: 
Description: 因子读取
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union

import dolphindb as ddb
import pandas as pd
from streamlit_utils.utils import datetime2str
from .table_setting import CREAT_EODPRICES, CREATE_FACTORDEV
from . import config


def check_path_exist(path: str) -> bool:
    if isinstance(path, str):
        path: Path = Path(path)
    return path.exists()


class LoaderBase(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_factor_data(
        self, factor_name: Union[List, str], start_dt: str, end_dt: str
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_stock_price(
        self,
        codes: Union[str, List],
        start_dt: str,
        end_dt: str,
        fields: Union[str, List],
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_factor_name_list(self) -> List[str]:
        pass

    @abstractmethod
    def get_factor_begin_and_end_period(self, factor_name: str) -> List[str]:
        pass


class DolinphdbLoader(LoaderBase):
    def __init__(
        self,
        host: str = config.DB_CONN["host"],
        port: int = config.DB_CONN["port"],
        username: str = config.DB_CONN["username"],
        password: str = config.DB_CONN["password"],
    ) -> None:
        # 连接数据库
        self.session: ddb.session = ddb.session()
        self.session.connect(host, port, username, password)
        self.factor_data: pd.DataFrame = None
        self.stock_price: pd.DataFrame = None

        if not self.session.existsDatabase(config.FACTPR_DB_PATH):
            self.session.run(CREATE_FACTORDEV)

        if not self.session.existsDatabase(config.PRICE_DB_PATH):
            self.session.run(CREAT_EODPRICES)

    def get_factor_data(
        self, factor_name: Union[List, str], start_dt: str = None, end_dt: str = None
    ) -> pd.DataFrame:
        if not isinstance(factor_name, (str, list)):
            raise ValueError("factor_name must be str or list")

        sel_factor: str = (
            f"factor_name=='{factor_name}'"
            if isinstance(factor_name, str)
            else f"factor_name in {factor_name}"
        )

        time_between: List[str] = []
        if start_dt is not None:
            start_dt_str: str = datetime2str(start_dt, "%Y.%m.%d")
            time_between.append(f"trade_date >= {start_dt_str}")
        if end_dt is not None:
            end_dt_str: str = datetime2str(end_dt, "%Y.%m.%d")
            time_between.append(f"trade_date <= {end_dt_str}")
        time_between_str: str = " and ".join(time_between)

        expr: str = (
            f"{sel_factor} and ({time_between_str})" if time_between_str else sel_factor
        )

        query_expr: str = f"""
        factor_table = loadTable('{config.FACTPR_DB_PATH}', '{config.FACTOR_TABLE_NAME}')
        select * from factor_table where {expr} and (code like '6%SH' or code like '3%SZ' or code like '0%SZ')
        """

        self.factor_data = self.session.run(query_expr, clearMemory=True)
        return self.factor_data

    def get_stock_price(
        self,
        codes: Union[str, List],
        start_dt: str,
        end_dt: str,
        fields: Union[str, List],
    ) -> pd.DataFrame:
        fields: List[str] = [fields] if isinstance(fields, str) else fields

        default_fields: List[str] = ["code", "trade_date"] + [
            field for field in fields if field not in ["trade_date", "code"]
        ]
        default_fields: List[str] = [
            {"vwap": "avg_price as vwap"}.get(i, i) for i in default_fields
        ]
        default_fields_str: str = ",".join(default_fields)

        if not isinstance(codes, (str, list)):
            raise ValueError("codes must be str or list")

        sel_codes: str = {str: f"code=='{codes}'", list: f"code in {codes}"}.get(
            type(codes), ""
        )

        time_between: List[str] = []
        if start_dt is not None:
            start_dt_str: str = datetime2str(start_dt, "%Y.%m.%d")
            time_between.append(f"trade_date >= {start_dt_str}")
        if end_dt is not None:
            end_dt_str: str = datetime2str(end_dt, "%Y.%m.%d")
            time_between.append(f"trade_date <= {end_dt_str}")
        time_between_str: str = " and ".join(time_between)

        expr: str = (
            f"{sel_codes} and ({time_between_str})" if time_between else sel_codes
        )

        query_expr: str = f"""
        price_table = loadTable('{config.PRICE_DB_PATH}', '{config.PRICE_TABLE_NAME}')
        select {default_fields_str} from price_table where {expr} and (code like '6%SH' or code like '3%SZ' or code like '0%SZ')
        """

        self.stock_price = self.session.run(query_expr, clearMemory=True)
        return self.stock_price

    def get_factor_name_list(self) -> List[str]:
        expr = f"""
        table = loadTable('{config.FACTPR_DB_PATH}', '{config.FACTOR_TABLE_NAME}')
        schema(table).partitionSchema[1]
        """

        factor_name: List[str] = self.session.run(expr, clearMemory=True).tolist()
        return [factor for factor in factor_name if factor not in ["f1", "f2"]]

    def get_factor_begin_and_end_period(self, factor_name: str) -> List[str]:
        if not isinstance(factor_name, str):
            raise ValueError("factor_name must be str")

        query_expr: str = f"""
        factor_table = loadTable('{config.FACTPR_DB_PATH}', '{config.FACTOR_TABLE_NAME}')
        select min(trade_date),max(trade_date) from factor_table where factor_name == "{factor_name}"
        """
        return (
            self.session.run(query_expr, clearMemory=True)
            .iloc[0]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

    def get_factor_board(self, watch_dt: str = None, count: int = None) -> pd.DataFrame:

        def _query_max_time(table_name: str) -> pd.DatetimeIndex:

            current_dt: pd.DatetimeIndex = self.session.run(
                f"""select max(trade_date) as current_dt from loadTable("{config.FACTOR_PATH}","{table_name}")"""
            )["current_dt"].iloc[0]

            return datetime2str(current_dt, "%Y.%m.%d")

        def _get_offsetday_from_trend_table(watch_dt: str, count: int) -> str:

            all_idx: pd.Series = self.session.run(
                f"""select distinct trade_date from loadTable("{config.FACTOR_PATH}","{config.BOARD_TREND}") where trade_date <= {watch_dt}"""
            )["trade_date"]
            return datetime2str(all_idx.iloc[-count], "%Y.%m.%d")

        if watch_dt is None:
            watch_dt: str = _query_max_time(config.BOARD_TABLE)

        summary_table: pd.DataFrame = self.session.run(
            f"""select * from loadTable("{config.FACTOR_PATH}","{config.BOARD_TABLE}") where trade_date = {watch_dt}"""
        )

        max_dt: str = _query_max_time(config.BOARD_TREND)
        begin_dt: str = (
            _get_offsetday_from_trend_table(watch_dt, count)
            if count is not None
            else None
        )
        if begin_dt is None:
            trend_df: pd.DataFrame = self.session.run(
                f"""select * from loadTable("{config.FACTOR_PATH}","{config.BOARD_TREND}") where trade_date <= {watch_dt}"""
            )

        else:
            trend_df: pd.DataFrame = self.session.run(
                f"""select * from loadTable({config.FACTOR_PATH},{config.BOARD_TREND}") where trade_date >= {begin_dt} and trade_date <= {watch_dt}"""
            )


class ParquetLoder(LoaderBase):
    def __init__(
        self,
        price_path: str = config.PARQUET_PATH["price_path"],
        factor_path: str = config.PARQUET_PATH["factor_path"],
    ) -> None:

        if not check_path_exist(price_path) or not check_path_exist(factor_path):
            raise ValueError("price_path or factor_path not exist")

        self.price_path = price_path
        self.factor_path = factor_path

    def get_factor_data(
        self, factor_name: Union[List, str], start_dt: str = None, end_dt: str = None
    ) -> pd.DataFrame:

        if not isinstance(factor_name, (str, list)):
            raise ValueError("factor_name must be str or list")

        sel: List[Tuple] = []
        if isinstance(factor_name, str):
            sel.append(("factor_name", "=", factor_name))

        elif isinstance(factor_name, list):
            sel.append(("factor_name", "in", factor_name))

        else:
            raise ValueError("factor_name must be str or list")

        if start_dt is not None:
            sel.append(("trade_date", ">=", pd.to_datetime(start_dt)))
        if end_dt is not None:

            sel.append(("trade_date", "<=", pd.to_datetime(end_dt)))

        return pd.read_parquet(self.factor_path, filters=sel)

    def get_stock_price(
        self,
        codes: Union[str, List],
        start_dt: str,
        end_dt: str,
        fields: Union[str, List],
    ) -> pd.DataFrame:

        fields: List[str] = [fields] if isinstance(fields, str) else fields

        default_fields: List[str] = ["code", "trade_date"] + [
            field for field in fields if field not in ["trade_date", "code"]
        ]

        sel: List[Tuple] = []
        if isinstance(codes, str):
            sel.append(("code", "=", codes))

        elif isinstance(codes, list):
            sel.append(("code", "in", codes))

        else:
            raise ValueError("code must be str or list")

        if start_dt is not None:
            sel.append(("trade_date", ">=", pd.to_datetime(start_dt)))
        if end_dt is not None:

            sel.append(("trade_date", "<=", pd.to_datetime(end_dt)))

        return pd.read_parquet(self.price_path, filters=sel, columns=default_fields)

    def get_factor_name_list(self) -> List[str]:

        return (
            pd.read_parquet(self.factor_path, columns=["factor_name"])["factor_name"]
            .unique()
            .tolist()
        )

    def get_factor_begin_and_end_period(self, factor_name: str) -> List:

        df: pd.DataFrame = pd.read_parquet(
            self.factor_path,
            columns=["trade_date", "factor_name"],
            fileds=[("factor_name", "=", factor_name)],
        )
        return [df["trade_date"].min(), df["trade_date"].max()]


class Loader:

    def __init__(self, is_ddb: bool) -> None:
        if is_ddb:
            self.loader = DolinphdbLoader()
        else:
            self.loader = ParquetLoder()

    def get_factor_data(
        self, factor_name: Union[List, str], start_dt: str = None, end_dt: str = None
    ) -> pd.DataFrame:
        return self.loader.get_factor_data(factor_name, start_dt, end_dt)

    def get_stock_price(
        self,
        codes: Union[str, List],
        start_dt: str,
        end_dt: str,
        fields: Union[str, List],
    ) -> pd.DataFrame:
        return self.loader.get_stock_price(codes, start_dt, end_dt, fields)

    def get_factor_name_list(self) -> List[str]:
        return self.loader.get_factor_name_list()

    def get_factor_begin_and_end_period(self, factor_name: str) -> List[str]:
        return self.loader.get_factor_begin_and_end_period(factor_name)
