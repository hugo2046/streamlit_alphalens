"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2023-12-14 22:39:48
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2023-12-14 22:56:04
FilePath: 
Description: 因子读取
"""
import pandas as pd
import dolphindb as ddb
from streamlit_utils.utils import datetime2str
from typing import Union, List, Dict
from . import config


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


class DolinphdbLoader:
    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self.host: str = host
        self.port: int = int(port)
        self.username: str = username
        self.password: str = password

        # 连接数据库
        self.session: ddb.session = ddb.session()
        self.session.connect(self.host, self.port, self.username, self.password)

    def __del__(self) -> None:
        if not self.session.isClosed:
            self.session.close()

    def get_factor_data(
        self,
        codes: Union[List, str] = None,
        start_dt: str = None,
        end_dt: str = None,
        fields: Union[str, List] = None,
    ) -> pd.DataFrame:
        befault_col: set = {"code", "trade_date"}
        fields: set = set(fields) if fields else set()
        fields: List[str] = list(befault_col.union(fields))

        if start_dt is None or end_dt is None:
            raise ValueError("start_dt 和 end_dt 不能为空")

        start_dt: str = datetime2str(start_dt, "%Y.%m.%d")
        end_dt: str = datetime2str(end_dt, "%Y.%m.%d")

        sel_time_expr: str = f"trade_date >= {start_dt} and trade_date <= {end_dt}"

        code_expr_map: Dict = {str: f"code == '{codes}'", list: f"code in {codes}"}
        sel_code_expr: str = code_expr_map.get(type(codes), "")
        expr: str = (
            sel_time_expr
            + (f" and {sel_code_expr}" if sel_code_expr else "")
            + " and (code like '%SZ' or code like '%SH')"
        )

        table = self.session.loadTable(
            tableName=config.FACTOR_TABLE_NAME, dbPath=config.FACTPR_DB_PATH
        )
        df: pd.DataFrame = table.select(fields).where(expr).toDF()
        self.session.undef(table.tableName, "VAR")
        return df.sort_values("trade_date")

    def get_factor_name(self) -> List[str]:
        table = self.session.loadTable(
            tableName=config.FACTOR_TABLE_NAME, dbPath=config.FACTPR_DB_PATH
        )
        return [
            col for col in table.schema["name"] if col not in ["code", "trade_date"]
        ]

    def get_stock_price(
        self,
        codes: Union[str, List],
        start_dt: str,
        end_dt: str,
        fields: Union[str, List],
    ) -> pd.DataFrame:
        befault_col: set = {"code", "trade_date"}
        fields: set = set(fields) if fields else set()
        fields: List[str] = list(befault_col.union(fields))

        if start_dt is None or end_dt is None:
            raise ValueError("start_dt 和 end_dt 不能为空")

        start_dt: str = datetime2str(start_dt, "%Y.%m.%d")
        end_dt: str = datetime2str(end_dt, "%Y.%m.%d")

        sel_time_expr: str = f"trade_date >= {start_dt} and trade_date <= {end_dt}"

        code_expr_map: Dict = {str: f"code == '{codes}'", list: f"code in {codes}"}
        sel_code_expr: str = code_expr_map.get(type(codes), "")
        expr: str = (
            sel_time_expr
            + (f" and {sel_code_expr}" if sel_code_expr else "")
            + " and (code like '%SZ' or code like '%SH')"
        )

        table = self.session.loadTable(
            tableName=config.PRICE_TABLE_NAME, dbPath=config.PRICE_DB_PATH
        )
        df: pd.DataFrame = table.select(fields).where(expr).toDF()
        self.session.undef(table.tableName, "VAR")
        return df.sort_values("trade_date")


class DataLoader:
    def __init__(self, method: str) -> None:
        params: Dict = {"csv": config.CSV_PATH, "db": config.DB_CONN}[method.lower()]
        self.loader: Union[DolinphdbLoader, CSVLoder] = {
            "csv": CSVLoder,
            "dolphindb": DolinphdbLoader,
        }[method](**params)

    def get_factor_data(
        self,
        start_dt: str,
        end_dt: str,
        factor_name: List[str],
    ) -> pd.DataFrame:
        return self.loader.get_factor_data(
            start_dt=start_dt, end_dt=end_dt, fields=factor_name
        )

    def get_stock_price(
        self, codes: Union[str, List], start_dt: str, end_dt: str
    ) -> pd.DataFrame:
        return self.loader.get_stock_price(
            codes=codes,
            start_dt=start_dt,
            end_dt=end_dt,
            fields=["code", "trade_date", "vwap"],
        )

    def get_factor_name(self) -> List[str]:
        return self.loader.get_factor_name()
