import pandas as pd
import numpy as np
from typing import Optional


class MicrostructureCalculator:
    def __init__(self):
        self.lookback_windows = [5, 10, 20, 50]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        features = self._calculate_spread_features(df, features)
        features = self._calculate_order_flow_features(df, features)
        features = self._calculate_volume_profile_features(df, features)
        features = self._calculate_price_impact_features(df, features)
        features = self._calculate_liquidity_features(df, features)

        return features

    def _calculate_spread_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        features['avg_spread'] = (df['high'] - df['low']).rolling(window=20).mean()
        features['spread_volatility'] = (df['high'] - df['low']).rolling(window=20).std()
        features['relative_spread'] = (df['high'] - df['low']) / df['close']

        features['spread_momentum'] = features['avg_spread'] / features['avg_spread'].shift(10)

        return features

    def _calculate_order_flow_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'taker_buy_volume' in df.columns:
            features['buy_pressure'] = df['taker_buy_volume'] / df['volume']
            features['sell_pressure'] = 1 - features['buy_pressure']

            features['order_flow_imbalance'] = features['buy_pressure'].rolling(window=20).mean() - 0.5

            features['volume_weighted_buy_pressure'] = (
                (df['taker_buy_volume'] * df['close']).rolling(window=20).sum() /
                (df['volume'] * df['close']).rolling(window=20).sum()
            )

        return features

    def _calculate_volume_profile_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for window in self.lookback_windows:
            vwap = (df['close'] * df['volume']).rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
            features[f'vwap_deviation_{window}'] = (df['close'] - vwap) / vwap

        features['volume_concentration'] = df['volume'].rolling(window=5).sum() / df['volume'].rolling(window=20).sum()

        if 'trades' in df.columns:
            features['avg_trade_size'] = df['volume'] / (df['trades'] + 1)
            features['trade_size_momentum'] = features['avg_trade_size'] / features['avg_trade_size'].shift(20)

        return features

    def _calculate_price_impact_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        volume_bars = df['volume'].rolling(window=20).mean()
        price_change = np.abs(df['close'].pct_change())

        features['price_impact'] = price_change / (df['volume'] / volume_bars + 1e-10)

        features['kyle_lambda'] = np.abs(df['close'].diff()) / np.sqrt(df['volume'] + 1e-10)

        features['amihud_illiquidity'] = (np.abs(df['close'].pct_change()) / (df['volume'] * df['close'] + 1e-10)).rolling(window=20).mean()

        return features

    def _calculate_liquidity_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        features['volume_turnover'] = df['volume'].rolling(window=20).sum() / df['volume'].rolling(window=100).sum()

        high_low_spread = df['high'] - df['low']
        features['effective_spread'] = 2 * np.abs(df['close'] - (df['high'] + df['low']) / 2)

        features['realized_volatility'] = np.sqrt(
            ((df['high'] / df['low']).apply(np.log) ** 2 / (4 * np.log(2))).rolling(window=20).mean()
        )

        if 'quote_volume' in df.columns:
            features['depth_imbalance'] = df['quote_volume'].rolling(window=5).mean() / df['quote_volume'].rolling(window=20).mean()

        return features