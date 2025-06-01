# services/feature-engine/indicators/microstructure.py
import pandas as pd
import numpy as np
import talib
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
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values

        spread = high_prices - low_prices

        features['spread_ratio'] = np.where(
            close_prices > 0,
            spread / close_prices,
            np.nan
        )

        if len(df) >= 20:
            features['avg_spread'] = talib.SMA(spread, timeperiod=20)
            features['spread_volatility'] = talib.STDDEV(spread, timeperiod=20)

        return features

    def _calculate_price_impact_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            return features

        close_prices = df['close'].values
        volume = df['volume'].values

        if len(df) >= 2:
            price_change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
            sqrt_volume = np.sqrt(np.where(volume > 0, volume, np.nan))

            kyle_lambda = np.where(
                sqrt_volume > 0,
                price_change / sqrt_volume,
                np.nan
            )

            if len(df) >= 20:
                features['kyle_lambda'] = talib.SMA(kyle_lambda, timeperiod=20)
            else:
                features['kyle_lambda'] = kyle_lambda

        if len(df) >= 20:
            returns = np.abs(talib.ROC(close_prices, timeperiod=1) / 100.0)
            dollar_volume = close_prices * volume

            with np.errstate(divide='ignore', invalid='ignore'):
                amihud = np.where(
                    dollar_volume > 0,
                    returns / dollar_volume,
                    np.nan
                )
            features['amihud_illiquidity'] = talib.SMA(amihud, timeperiod=20)

            volume_bar = talib.SMA(volume, timeperiod=20)
            volume_ratio = np.where(
                volume_bar > 0,
                volume / volume_bar,
                np.nan
            )

            with np.errstate(divide='ignore', invalid='ignore'):
                features['price_impact'] = np.where(
                    volume_ratio > 0,
                    returns / volume_ratio,
                    np.nan
                )

        return features

    def _calculate_liquidity_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            return features

        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volume = df['volume'].values

        if len(df) >= 100:
            volume_20 = talib.SUM(volume, timeperiod=20)
            volume_100 = talib.SUM(volume, timeperiod=100)

            features['volume_turnover'] = np.where(
                volume_100 > 0,
                volume_20 / volume_100,
                np.nan
            )

        if len(df) >= 2:
            mid_price = (high_prices + low_prices) / 2
            features['effective_spread'] = 2 * np.abs(close_prices - mid_price)

        return features