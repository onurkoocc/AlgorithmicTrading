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
        self._connect()

    def _connect(self):
        try:
            if self._socket:
                try:
                    self._socket.close()
                except:
                    pass

            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self._socket.settimeout(30.0)
            self._socket.connect((self.host, self.port))
            self.logger.info(f"Connected to QuestDB at {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to QuestDB: {e}")
            raise

    def _send_line(self, line: str, retry_count: int = 3):
        for attempt in range(retry_count):
            try:
                if not line.endswith('\n'):
                    line += '\n'

                self._socket.sendall(line.encode('utf-8'))
                return True

            except (socket.error, BrokenPipeError, ConnectionResetError) as e:
                if attempt < retry_count - 1:
                    self.logger.warning(f"Send failed (attempt {attempt + 1}/{retry_count}): {e}")
                    time.sleep(0.1)
                    self._connect()
                else:
                    self.logger.error(f"Failed to send data after {retry_count} attempts")
                    raise
        return False

    def write_kline(self, symbol: str, interval: str, data: Dict[str, Any]):
        table_name = f"klines_{interval}"
        timestamp_ns = int(data['timestamp'] * 1_000_000_000)

        line = (
            f"{table_name},"
            f"symbol={symbol} "
            f"open={data['open']},"
            f"high={data['high']},"
            f"low={data['low']},"
            f"close={data['close']},"
            f"volume={data['volume']},"
            f"quote_volume={data.get('quote_volume', 0)},"
            f"trades={data.get('trades', 0)}i,"
            f"taker_buy_volume={data.get('taker_buy_volume', 0)},"
            f"taker_buy_quote_volume={data.get('taker_buy_quote_volume', 0)} "
            f"{timestamp_ns}"
        )

        self._send_line(line)
        self.metrics.record_db_write(table_name, 'success')

    def batch_write_klines(self, symbol: str, interval: str, data_list: List[Dict[str, Any]]):
        if not data_list:
            return

        table_name = f"klines_{interval}"
        batch_size = 100
        total_written = 0
        unique_timestamps = set()

        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]

            try:
                for data in batch:
                    timestamp_ns = int(data['timestamp'] * 1_000_000_000)

                    if timestamp_ns in unique_timestamps:
                        continue

                    unique_timestamps.add(timestamp_ns)

                    line = (
                        f"{table_name},"
                        f"symbol={symbol} "
                        f"open={data['open']},"
                        f"high={data['high']},"
                        f"low={data['low']},"
                        f"close={data['close']},"
                        f"volume={data['volume']},"
                        f"quote_volume={data.get('quote_volume', 0)},"
                        f"trades={data.get('trades', 0)}i,"
                        f"taker_buy_volume={data.get('taker_buy_volume', 0)},"
                        f"taker_buy_quote_volume={data.get('taker_buy_quote_volume', 0)} "
                        f"{timestamp_ns}"
                    )

                    self._send_line(line)
                    total_written += 1

                if i + batch_size < len(data_list):
                    time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Failed to write batch for {symbol} {interval}: {e}")
                self.metrics.record_db_write(table_name, 'error')
                try:
                    self._connect()
                except:
                    pass

        if total_written > 0:
            self.logger.info(f"Successfully written {total_written}/{len(data_list)} records for {symbol} {interval}")
            self.metrics.record_db_write(table_name, 'success')

    def execute_query(self, query: str, retry_count: int = 3) -> List[Dict[str, Any]]:
        for attempt in range(retry_count):
            try:
                response = requests.get(
                    f"{self.http_base_url}/exec",
                    params={'query': query},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    if 'dataset' not in data:
                        return []

                    columns = [col['name'] for col in data['columns']]
                    rows = data['dataset']

                    return [dict(zip(columns, row)) for row in rows]
                else:
                    raise Exception(f"Query failed with status {response.status_code}: {response.text}")

            except Exception as e:
                if attempt < retry_count - 1:
                    self.logger.warning(f"Query failed (attempt {attempt + 1}/{retry_count}): {e}")
                    time.sleep(1)
                else:
                    self.logger.error(f"Query execution failed after {retry_count} attempts: {e}")
                    raise

    def get_latest_timestamp(self, symbol: str, interval: str) -> Optional[int]:
        table_name = f"klines_{interval}"
        query = f"""
            SELECT max(timestamp) as timestamp 
            FROM {table_name} 
            WHERE symbol = '{symbol}'
        """

        try:
            result = self.execute_query(query)
            if result and result[0]['timestamp'] is not None:
                return int(result[0]['timestamp'] / 1000)
        except Exception as e:
            self.logger.warning(f"Failed to get latest timestamp for {symbol} {interval}: {e}")
        return None

    def create_tables(self):
        intervals = self.config.intervals

        for interval in intervals:
            table_name = f"klines_{interval}"

            drop_query = f"DROP TABLE IF EXISTS {table_name}"

            try:
                self.execute_query(drop_query)
                self.logger.info(f"Dropped existing table {table_name}")
            except:
                pass

            create_query = f"""
                CREATE TABLE {table_name} (
                    symbol SYMBOL capacity 256 CACHE INDEX,
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
                ) timestamp(timestamp) PARTITION BY DAY WAL DEDUP UPSERT KEYS(symbol, timestamp);
            """

            try:
                self.execute_query(create_query)
                self.logger.info(f"Created table {table_name}")
            except Exception as e:
                self.logger.error(f"Failed to create table {table_name}: {e}")
                raise

    def get_klines_df(self, symbol: str, interval: str,
                      start_time: Optional[int] = None,
                      end_time: Optional[int] = None,
                      limit: int = 1000) -> pd.DataFrame:
        table_name = f"klines_{interval}"

        where_conditions = [f"symbol = '{symbol}'"]

        if start_time:
            where_conditions.append(f"timestamp >= {start_time}000000")
        if end_time:
            where_conditions.append(f"timestamp <= {end_time}000000")

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
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
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

    def verify_data(self, symbol: str, interval: str) -> int:
        table_name = f"klines_{interval}"
        query = f"SELECT count() as cnt FROM {table_name} WHERE symbol = '{symbol}'"

        try:
            result = self.execute_query(query)
            if result:
                return result[0]['cnt']
        except:
            pass
        return 0

    def check_duplicates(self, symbol: str, interval: str) -> int:
        table_name = f"klines_{interval}"
        query = f"""
            SELECT timestamp, count(*) as cnt 
            FROM {table_name} 
            WHERE symbol = '{symbol}' 
            GROUP BY timestamp 
            HAVING count(*) > 1
        """

        try:
            result = self.execute_query(query)
            if result:
                return len(result)
        except:
            pass
        return 0

    def wait_for_commit(self, timeout: int = 5):
        time.sleep(timeout)

    def get_features_df(self, symbol: str, interval: str,
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None,
                        limit: int = 1000) -> pd.DataFrame:
        table_name = f"features_{interval}"

        where_conditions = [f"symbol = '{symbol}'"]

        if start_time:
            where_conditions.append(f"timestamp >= {start_time}000000")
        if end_time:
            where_conditions.append(f"timestamp <= {end_time}000000")

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
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            return df
        except Exception as e:
            self.logger.error(f"Failed to get features dataframe: {e}")
            return pd.DataFrame()