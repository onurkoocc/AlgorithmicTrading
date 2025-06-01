import pandas as pd
import numpy as np
from typing import Optional


class CustomIndicatorCalculator:
    def __init__(self):
        self.lookback_periods = [1, 5, 10, 20]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)

        features = self._calculate_volatility(df, features)
        features = self._calculate_price_momentum(df, features)
        features = self._calculate_volume_features(df, features)
        features = self._calculate_trend_features(df, features)

        return features

    def _calculate_volatility(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return features

        returns = df['close'].pct_change()

        if len(df) >= 20:
            features['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252)

        if len(df) >= 50:
            features['volatility_50'] = returns.rolling(window=50).std() * np.sqrt(252)

        if len(df) >= 20:
            parkinson = np.sqrt(
                np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
            ).rolling(window=20).mean()
            features['realized_volatility'] = parkinson * np.sqrt(252)

        return features

    def _calculate_price_momentum(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if len(df) >= 20:
            highest_20 = df['high'].rolling(window=20).max()
            lowest_20 = df['low'].rolling(window=20).min()
            price_range = highest_20 - lowest_20

            features['price_position'] = np.where(
                price_range > 0,
                (df['close'] - lowest_20) / price_range,
                0.5
            )

        if len(df) >= 50:
            sma_50 = df['close'].rolling(window=50).mean()
            features['trend_strength'] = (df['close'] - sma_50) / sma_50.replace(0, np.nan)

        if len(df) >= 26:
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            features['momentum_score'] = (ema_12 - ema_26) / ema_26.replace(0, np.nan)

        return features

    def _calculate_volume_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            return features

        if len(df) >= 20:
            volume_sma_5 = df['volume'].rolling(window=5).mean()
            volume_sma_20 = df['volume'].rolling(window=20).mean()

            features['volume_momentum'] = volume_sma_5 / volume_sma_20.replace(0, np.nan)

        if 'taker_buy_volume' in df.columns:
            features['buy_pressure'] = df['taker_buy_volume'] / df['volume'].replace(0, np.nan)

            if len(df) >= 20:
                buy_pressure_ma = features['buy_pressure'].rolling(window=20).mean()
                features['order_flow_imbalance'] = buy_pressure_ma - 0.5

        return features

    def _calculate_trend_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return features

        if len(df) >= 20:
            returns = df['close'].pct_change()
            rolling_mean = returns.rolling(window=20).mean()
            rolling_std = returns.rolling(window=20).std()

            features['sharpe_ratio_20'] = np.where(
                rolling_std > 0,
                rolling_mean * np.sqrt(252) / (rolling_std * np.sqrt(252)),
                0
            )

        return features