from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class Kline:
    symbol: str
    interval: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    taker_buy_volume: float
    taker_buy_quote_volume: float
    is_closed: bool = True

    @classmethod
    def from_binance_ws(cls, symbol: str, interval: str, data: list) -> 'Kline':
        return cls(
            symbol=symbol,
            interval=interval,
            timestamp=data[0] / 1000.0,
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            quote_volume=float(data[7]),
            trades=int(data[8]),
            taker_buy_volume=float(data[9]),
            taker_buy_quote_volume=float(data[10]),
            is_closed=data[11]
        )

    @classmethod
    def from_binance_rest(cls, symbol: str, interval: str, data: list) -> 'Kline':
        return cls(
            symbol=symbol,
            interval=interval,
            timestamp=data[0] / 1000.0,
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            quote_volume=float(data[7]),
            trades=int(data[8]),
            taker_buy_volume=float(data[9]),
            taker_buy_quote_volume=float(data[10]),
            is_closed=True
        )

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'interval': self.interval,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'quote_volume': self.quote_volume,
            'trades': self.trades,
            'taker_buy_volume': self.taker_buy_volume,
            'taker_buy_quote_volume': self.taker_buy_quote_volume,
            'is_closed': self.is_closed
        }


@dataclass
class OrderBook:
    symbol: str
    timestamp: float
    bids: list
    asks: list

    @property
    def best_bid(self) -> Optional[float]:
        return float(self.bids[0][0]) if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return float(self.asks[0][0]) if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class Trade:
    symbol: str
    timestamp: float
    price: float
    quantity: float
    is_buyer_maker: bool

    @classmethod
    def from_binance_ws(cls, symbol: str, data: dict) -> 'Trade':
        return cls(
            symbol=symbol,
            timestamp=data['T'] / 1000.0,
            price=float(data['p']),
            quantity=float(data['q']),
            is_buyer_maker=data['m']
        )