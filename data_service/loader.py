"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-14 22:39:48
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-14 22:56:04
FilePath: 
Description: 因子读取
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import dolphindb as ddb
import pandas as pd
from streamlit_utils.utils import datetime2str

from . import config


class Loader(ABC):
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


class DolinphdbLoader(Loader):
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

    def get_factor_data(
        self, factor_name: Union[List, str], start_dt: str, end_dt: str
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

    @property
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


class CSVLoder:
    def __init__(self, price_path: str, factor_path: str) -> None:
        self.price_path = price_path
        self.factor_path = factor_path

    def get_factor_data(
        self,
        codes: Union[List, str] = None,
        start_dt: str = None,
        end_dt: str = None,
        fields: Union[str, List] = None,
    ) -> pd.DataFrame:
        if isinstance(fields, str):
            fields: List[str] = [fields]
        fields: List[str] = list({"trade_date", "code"}.union(fields))
        df: pd.DataFrame = pd.read_csv(self.factor_path, parse_dates=True)
        df.sort_values("trade_date", inplace=True)

        df: pd.DataFrame = df.query(
            "trade_date >= @start_dt and trade_date <= @end_dt"
        )[fields]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if codes:
            df: pd.DataFrame = df[df["code"].isin(codes)]

        return df

    def get_stock_price(
        self,
        codes: Union[str, List],
        start_dt: str,
        end_dt: str,
        fields: Union[str, List],
    ) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(self.price_path, parse_dates=True)
        df.sort_values("trade_date", inplace=True)
        fields: List[str] = list({"trade_date", "code"}.union(fields))
        df: pd.DataFrame = df.query(
            "trade_date >= @start_dt and trade_date <= @end_dt"
        )[fields]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        if codes:
            df: pd.DataFrame = df[df["code"].isin(codes)]

        return df

    def get_factor_name(self) -> List[str]:
        # 获取csv文件的列名
        df: pd.DataFrame = pd.read_csv(
            self.factor_path, index_col=None, parse_dates=True, nrows=1
        )
        return [col for col in df.columns if col not in ["code", "trade_date"]]


# class DolinphdbLoader:
#     def __init__(self, host: str, port: int, username: str, password: str) -> None:
#         self.host: str = host
#         self.port: int = int(port)
#         self.username: str = username
#         self.password: str = password

#         # 连接数据库
#         self.session: ddb.session = ddb.session()
#         self.session.connect(self.host, self.port, self.username, self.password)

#     def __del__(self) -> None:
#         if not self.session.isClosed:
#             self.session.close()

#     def get_factor_data(
#         self,
#         codes: Union[List, str] = None,
#         start_dt: str = None,
#         end_dt: str = None,
#         fields: Union[str, List] = None,
#     ) -> pd.DataFrame:
#         befault_col: set = {"code", "trade_date"}
#         fields: set = set(fields) if fields else set()
#         fields: List[str] = list(befault_col.union(fields))

#         if start_dt is None or end_dt is None:
#             raise ValueError("start_dt 和 end_dt 不能为空")

#         start_dt: str = datetime2str(start_dt, "%Y.%m.%d")
#         end_dt: str = datetime2str(end_dt, "%Y.%m.%d")

#         sel_time_expr: str = f"trade_date >= {start_dt} and trade_date <= {end_dt}"

#         code_expr_map: Dict = {str: f"code == '{codes}'", list: f"code in {codes}"}
#         sel_code_expr: str = code_expr_map.get(type(codes), "")
#         expr: str = (
#             sel_time_expr
#             + (f" and {sel_code_expr}" if sel_code_expr else "")
#             + " and (code like '%SZ' or code like '%SH')"
#         )

#         table = self.session.loadTable(
#             tableName=config.FACTOR_TABLE_NAME, dbPath=config.FACTPR_DB_PATH
#         )
#         df: pd.DataFrame = table.select(fields).where(expr).toDF()
#         self.session.undef(table.tableName, "VAR")
#         return df.sort_values("trade_date")

#     def get_factor_name(self) -> List[str]:
#         table = self.session.loadTable(
#             tableName=config.FACTOR_TABLE_NAME, dbPath=config.FACTPR_DB_PATH
#         )
#         return [
#             col for col in table.schema["name"] if col not in ["code", "trade_date"]
#         ]

#     def get_stock_price(
#         self,
#         codes: Union[str, List],
#         start_dt: str,
#         end_dt: str,
#         fields: Union[str, List],
#     ) -> pd.DataFrame:
#         befault_col: set = {"code", "trade_date"}
#         fields: set = set(fields) if fields else set()
#         fields: List[str] = list(befault_col.union(fields))

#         if start_dt is None or end_dt is None:
#             raise ValueError("start_dt 和 end_dt 不能为空")

#         start_dt: str = datetime2str(start_dt, "%Y.%m.%d")
#         end_dt: str = datetime2str(end_dt, "%Y.%m.%d")

#         sel_time_expr: str = f"trade_date >= {start_dt} and trade_date <= {end_dt}"

#         code_expr_map: Dict = {str: f"code == '{codes}'", list: f"code in {codes}"}
#         sel_code_expr: str = code_expr_map.get(type(codes), "")
#         expr: str = (
#             sel_time_expr
#             + (f" and {sel_code_expr}" if sel_code_expr else "")
#             + " and (code like '%SZ' or code like '%SH')"
#         )

#         table = self.session.loadTable(
#             tableName=config.PRICE_TABLE_NAME, dbPath=config.PRICE_DB_PATH
#         )
#         df: pd.DataFrame = table.select(fields).where(expr).toDF()
#         self.session.undef(table.tableName, "VAR")
#         return df.sort_values("trade_date")


# class DataLoader:
#     def __init__(self, method: str) -> None:
#         params: Dict = {"csv": config.CSV_PATH, "db": config.DB_CONN}[method.lower()]
#         self.loader: Union[DolinphdbLoader, CSVLoder] = {
#             "csv": CSVLoder,
#             "dolphindb": DolinphdbLoader,
#         }[method](**params)

#     def get_factor_data(
#         self,
#         start_dt: str,
#         end_dt: str,
#         factor_name: List[str],
#     ) -> pd.DataFrame:
#         return self.loader.get_factor_data(
#             start_dt=start_dt, end_dt=end_dt, fields=factor_name
#         )

#     def get_stock_price(
#         self, codes: Union[str, List], start_dt: str, end_dt: str
#     ) -> pd.DataFrame:
#         return self.loader.get_stock_price(
#             codes=codes,
#             start_dt=start_dt,
#             end_dt=end_dt,
#             fields=["code", "trade_date", "vwap"],
#         )

#     def get_factor_name(self) -> List[str]:
#         return self.loader.get_factor_name()
