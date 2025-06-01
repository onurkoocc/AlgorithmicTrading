import pandas as pd
import numpy as np
from typing import Optional


class MicrostructureCalculator:
    def __init__(self):
        self.lookback_windows = [5, 10, 20, 50]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=df.index)

        features = pd.DataFrame(index=df.index)

        features = self._calculate_spread_features(df, features)
        features = self._calculate_price_impact_features(df, features)
        features = self._calculate_liquidity_features(df, features)

        return features

    def _calculate_spread_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        spread = df['high'] - df['low']

        features['spread_ratio'] = spread / df['close'].replace(0, np.nan)

        if len(df) >= 20:
            features['avg_spread'] = spread.rolling(window=20).mean()
            features['spread_volatility'] = spread.rolling(window=20).std()

        return features

    def _calculate_price_impact_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            return features

        if len(df) >= 2:
            price_change = df['close'].diff().abs()
            sqrt_volume = np.sqrt(df['volume'].replace(0, np.nan))

            features['kyle_lambda'] = price_change / sqrt_volume

            if len(df) >= 20:
                features['kyle_lambda'] = features['kyle_lambda'].rolling(window=20).mean()

        if len(df) >= 20:
            returns = df['close'].pct_change().abs()
            dollar_volume = df['close'] * df['volume']

            amihud = returns / dollar_volume.replace(0, np.nan)
            features['amihud_illiquidity'] = amihud.rolling(window=20).mean()

            volume_bar = df['volume'].rolling(window=20).mean()
            volume_ratio = df['volume'] / volume_bar.replace(0, np.nan)

            features['price_impact'] = returns / volume_ratio.replace(0, np.nan)

        return features

    def _calculate_liquidity_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            return features

        if len(df) >= 100:
            volume_20 = df['volume'].rolling(window=20).sum()
            volume_100 = df['volume'].rolling(window=100).sum()

            features['volume_turnover'] = volume_20 / volume_100.replace(0, np.nan)

        if len(df) >= 2:
            mid_price = (df['high'] + df['low']) / 2
            features['effective_spread'] = 2 * (df['close'] - mid_price).abs()

        return features