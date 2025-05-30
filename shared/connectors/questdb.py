import socket
import time
from typing import Dict, List, Any, Optional, Tuple
import requests
from urllib.parse import quote
from ..utils.config import Config
from ..utils.logging import setup_logger
from ..utils.metrics import MetricsCollector
import pandas as pd
from datetime import datetime
import struct


class QuestDBConnector:
    def __init__(self):
        self.config = Config()
        self.logger = setup_logger(__name__)
        self.metrics = MetricsCollector()

        self.host = self.config.database.host
        self.port = self.config.database.port
        self.http_port = self.config.database.http_port

        self.http_base_url = f"http://{self.host}:{self.http_port}"
        self._socket = None
        self._connect_with_retry()

    def _connect_with_retry(self, max_retries: int = 5, retry_delay: float = 2.0):
        for attempt in range(max_retries):
            try:
                self._connect()
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"Failed to connect after {max_retries} attempts")
                    raise

    def _connect(self):
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self.host, self.port))
            self.logger.info(f"Connected to QuestDB at {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to QuestDB: {e}")
            raise

    def _reconnect(self):
        if self._socket:
            try:
                self._socket.close()
            except:
                pass
        self._connect_with_retry()

    def write_line(self, line: str):
        try:
            self._socket.sendall((line + '\n').encode())
        except Exception as e:
            self.logger.error(f"Failed to write line: {e}")
            self._reconnect()
            self._socket.sendall((line + '\n').encode())

    def write_kline(self, symbol: str, interval: str, data: Dict[str, Any]):
        table_name = f"klines_{interval}"

        timestamp = int(data['timestamp'] * 1_000_000)

        line = (
            f"{table_name},"
            f"symbol={symbol} "
            f"open={data['open']},"
            f"high={data['high']},"
            f"low={data['low']},"
            f"close={data['close']},"
            f"volume={data['volume']},"
            f"quote_volume={data.get('quote_volume', 0)},"
            f"trades={data.get('trades', 0)},"
            f"taker_buy_volume={data.get('taker_buy_volume', 0)},"
            f"taker_buy_quote_volume={data.get('taker_buy_quote_volume', 0)} "
            f"{timestamp}"
        )

        self.write_line(line)
        self.metrics.record_db_write(table_name, 'success')

    def batch_write_klines(self, symbol: str, interval: str, data_list: List[Dict[str, Any]]):
        table_name = f"klines_{interval}"

        for data in data_list:
            timestamp = int(data['timestamp'] * 1_000_000)

            line = (
                f"{table_name},"
                f"symbol={symbol} "
                f"open={data['open']},"
                f"high={data['high']},"
                f"low={data['low']},"
                f"close={data['close']},"
                f"volume={data['volume']},"
                f"quote_volume={data.get('quote_volume', 0)},"
                f"trades={data.get('trades', 0)},"
                f"taker_buy_volume={data.get('taker_buy_volume', 0)},"
                f"taker_buy_quote_volume={data.get('taker_buy_quote_volume', 0)} "
                f"{timestamp}"
            )

            self.write_line(line)

        self.metrics.record_db_write(table_name, 'success')

    def execute_query(self, query: str, retry_count: int = 3) -> List[Dict[str, Any]]:
        for attempt in range(retry_count):
            try:
                response = requests.get(
                    f"{self.http_base_url}/exec",
                    params={'query': query},
                    timeout=30
                )
                response.raise_for_status()

                data = response.json()

                if 'dataset' not in data:
                    return []

                columns = [col['name'] for col in data['columns']]
                rows = data['dataset']

                return [dict(zip(columns, row)) for row in rows]

            except requests.exceptions.RequestException as e:
                if attempt < retry_count - 1:
                    self.logger.warning(f"Query failed (attempt {attempt + 1}/{retry_count}): {e}")
                    time.sleep(1)
                else:
                    self.logger.error(f"Query execution failed after {retry_count} attempts: {e}")
                    raise

    def get_latest_timestamp(self, symbol: str, interval: str) -> Optional[int]:
        table_name = f"klines_{interval}"
        query = f"""
            SELECT timestamp 
            FROM {table_name} 
            WHERE symbol = '{symbol}' 
            ORDER BY timestamp DESC 
            LIMIT 1
        """

        try:
            result = self.execute_query(query)
            if result:
                return int(result[0]['timestamp'])
        except:
            pass
        return None

    def create_tables(self):
        intervals = self.config.intervals

        for interval in intervals:
            table_name = f"klines_{interval}"

            create_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    symbol SYMBOL capacity 256 CACHE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    quote_volume DOUBLE,
                    trades LONG,
                    taker_buy_volume DOUBLE,
                    taker_buy_quote_volume DOUBLE,
                    timestamp TIMESTAMP
                ) timestamp(timestamp) PARTITION BY DAY WAL;
            """

            try:
                self.execute_query(create_query)
                self.logger.info(f"Created table {table_name}")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    self.logger.error(f"Failed to create table {table_name}: {e}")
                    raise

    def get_klines_df(self, symbol: str, interval: str,
                      start_time: Optional[int] = None,
                      end_time: Optional[int] = None,
                      limit: int = 1000) -> pd.DataFrame:
        table_name = f"klines_{interval}"

        where_conditions = [f"symbol = '{symbol}'"]

        if start_time:
            where_conditions.append(f"timestamp >= {start_time}")
        if end_time:
            where_conditions.append(f"timestamp <= {end_time}")

        where_clause = " AND ".join(where_conditions)

        query = f"""
            SELECT * FROM {table_name}
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """

        try:
            result = self.execute_query(query)

            if not result:
                return pd.DataFrame()

            df = pd.DataFrame(result)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            return df
        except Exception as e:
            self.logger.error(f"Failed to get klines dataframe: {e}")
            return pd.DataFrame()

    def close(self):
        if self._socket:
            try:
                self._socket.close()
                self.logger.info("Closed QuestDB connection")
            except:
                pass