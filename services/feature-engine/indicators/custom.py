import pandas as pd
import numpy as np
from typing import Optional


class CustomIndicatorCalculator:
    def __init__(self):
        self.lookback_periods = [1, 5, 10, 20]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        features = self._calculate_price_changes(df, features)
        features = self._calculate_returns(df, features)
        features = self._calculate_volatility(df, features)
        features = self._calculate_candle_patterns(df, features)
        features = self._calculate_trend_features(df, features)
        features = self._calculate_momentum_features(df, features)
        features = self._calculate_market_regime(df, features)

        return features

    def _calculate_price_changes(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for period in self.lookback_periods:
            features[f'price_change_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)

        return features

    def _calculate_returns(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for period in self.lookback_periods:
            features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))

        return features

    def _calculate_volatility(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        returns = np.log(df['close'] / df['close'].shift(1))

        features['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        features['volatility_50'] = returns.rolling(window=50).std() * np.sqrt(252)

        features['sharpe_ratio_20'] = (returns.rolling(window=20).mean() * 252) / (returns.rolling(window=20).std() * np.sqrt(252))

        return features

    def _calculate_candle_patterns(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']

        features['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'] + 1e-10)
        features['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'] + 1e-10)
        features['body_size'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)

        features['is_bullish_candle'] = df['close'] > df['open']

        return features

    def _calculate_trend_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        sma_200 = df['close'].rolling(window=200).mean()

        features['trend_strength'] = (df['close'] - sma_200) / sma_200

        ema_short = df['close'].ewm(span=12, adjust=False).mean()
        ema_long = df['close'].ewm(span=26, adjust=False).mean()
        features['momentum_score'] = (ema_short - ema_long) / ema_long

        return features

    def _calculate_momentum_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        features['volume_momentum'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()

        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        features['price_position'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-10)

        return features

    def _calculate_market_regime(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        returns = np.log(df['close'] / df['close'].shift(1))
        volatility = returns.rolling(window=20).std()
        trend = df['close'].rolling(window=50).mean().diff(10)

        conditions = [
            (trend > 0) & (volatility < volatility.rolling(window=100).median()),
            (trend > 0) & (volatility >= volatility.rolling(window=100).median()),
            (trend <= 0) & (volatility < volatility.rolling(window=100).median()),
            (trend <= 0) & (volatility >= volatility.rolling(window=100).median())
        ]

        values = [1, 2, 3, 4]

        features['market_regime'] = np.select(conditions, values, default=0)

        return features