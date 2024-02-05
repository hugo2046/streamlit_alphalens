'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2024-02-05 13:00:29
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2024-02-05 13:04:53
FilePath: 
Description: ddb表格模型
'''
from .config import FACTOR_TABLE_NAME, FACTPR_DB_PATH, PRICE_TABLE_NAME, PRICE_DB_PATH


CREATE_FACTORDEV: str = f"""
create database "{FACTPR_DB_PATH}" 
partitioned by RANGE(date(datetimeAdd(2000.12M,0..80*12,'M'))), VALUE(`f1`f2), 
engine='TSDB'

create table "{FACTPR_DB_PATH}"."{FACTOR_TABLE_NAME}"(
    trade_date DATE, 
    code SYMBOL, 
    factor_name SYMBOL,
    value DOUBLE
)
partitioned by trade_date, factor_name,
sortColumns=[`code, `trade_date], 
keepDuplicates=LAST, 
sortKeyMappingFunction=[hashBucket{{, 500}}]
"""

# 日线数据
CREAT_EODPRICES: str = f"""
db_path = '{PRICE_DB_PATH}'
db = database(db_path, RANGE, 1990.12M + (0..40)*12,engine="TSDB")


cols = ['code','trade_date','open','high','low','close','pre_close','change','pct_chg','volume','amount','adj_factor','limit','stopping','avg_price','adjopen','adjhigh','adjlow','adjclose']
types = [SYMBOL,DATE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,INT,INT,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE,DOUBLE]
t = table(1000:0, cols, types)
db.createPartitionedTable(table=t, tableName="{PRICE_TABLE_NAME}", partitionColumns=["trade_date"],sortColumns=['code','trade_date'],keepDuplicates=LAST)
"""
