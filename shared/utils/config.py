import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass
import threading


@dataclass
class DatabaseConfig:
    host: str
    port: int
    http_port: int


@dataclass
class RedisConfig:
    host: str
    port: int
    db: int


@dataclass
class BinanceConfig:
    api_key: str
    api_secret: str
    testnet: bool
    futures_base_url: str
    rest_base_url: str


class Config:
    _instance: Optional['Config'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config: Dict[str, Any] = {}
        self._config_path = Path("/app/config/config.yaml")
        self._last_modified = 0
        self._load_config()
        self._initialized = True

    def _load_config(self):
        if not self._config_path.exists():
            self._config_path = Path("config/config.yaml")

        with open(self._config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self._last_modified = os.path.getmtime(self._config_path)

    def reload_if_changed(self):
        current_modified = os.path.getmtime(self._config_path)
        if current_modified > self._last_modified:
            self._load_config()

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def database(self) -> DatabaseConfig:
        db_config = self.get('database.questdb', {})
        return DatabaseConfig(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 9009),
            http_port=db_config.get('http_port', 9000)
        )

    @property
    def redis(self) -> RedisConfig:
        redis_config = self.get('redis', {})
        return RedisConfig(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0)
        )

    @property
    def binance(self) -> BinanceConfig:
        binance_config = self.get('binance', {})
        return BinanceConfig(
            api_key=binance_config.get('api_key', ''),
            api_secret=binance_config.get('api_secret', ''),
            testnet=binance_config.get('testnet', True),
            futures_base_url=binance_config.get('futures_base_url', 'wss://fstream.binance.com'),
            rest_base_url=binance_config.get('rest_base_url', 'https://fapi.binance.com')
        )

    @property
    def symbols(self) -> list:
        return self.get('data_collection.symbols', ['BTCUSDT', 'ETHUSDT'])

    @property
    def intervals(self) -> list:
        return self.get('data_collection.intervals', ['1m', '5m', '15m', '1h', '4h', '1d'])

    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()