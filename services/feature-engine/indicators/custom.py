# services/feature-engine/indicators/custom.py
import pandas as pd
import numpy as np
import talib
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

        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values

        returns = talib.ROC(close_prices, timeperiod=1) / 100.0

        if len(df) >= 20:
            volatility_20 = talib.STDDEV(returns, timeperiod=20) * np.sqrt(252)
            features['volatility_20'] = volatility_20

        if len(df) >= 50:
            volatility_50 = talib.STDDEV(returns, timeperiod=50) * np.sqrt(252)
            features['volatility_50'] = volatility_50

        if len(df) >= 20:
            parkinson = np.sqrt(
                np.log(high_prices / low_prices) ** 2 / (4 * np.log(2))
            )
            parkinson_ma = talib.SMA(parkinson, timeperiod=20)
            features['realized_volatility'] = parkinson_ma * np.sqrt(252)

        return features

    def _calculate_price_momentum(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values

        if len(df) >= 20:
            highest_20 = talib.MAX(high_prices, timeperiod=20)
            lowest_20 = talib.MIN(low_prices, timeperiod=20)
            price_range = highest_20 - lowest_20

            features['price_position'] = np.where(
                price_range > 0,
                (close_prices - lowest_20) / price_range,
                0.5
            )

        if len(df) >= 50:
            sma_50 = talib.SMA(close_prices, timeperiod=50)
            features['trend_strength'] = np.where(
                sma_50 > 0,
                (close_prices - sma_50) / sma_50,
                np.nan
            )

        if len(df) >= 26:
            ema_12 = talib.EMA(close_prices, timeperiod=12)
            ema_26 = talib.EMA(close_prices, timeperiod=26)
            features['momentum_score'] = np.where(
                ema_26 > 0,
                (ema_12 - ema_26) / ema_26,
                np.nan
            )

        return features

    def _calculate_volume_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'volume' not in df.columns:
            return features

        volume = df['volume'].values

        if len(df) >= 20:
            volume_sma_5 = talib.SMA(volume, timeperiod=5)
            volume_sma_20 = talib.SMA(volume, timeperiod=20)

            features['volume_momentum'] = np.where(
                volume_sma_20 > 0,
                volume_sma_5 / volume_sma_20,
                np.nan
            )

        if 'taker_buy_volume' in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                features['buy_pressure'] = np.where(
                    volume > 0,
                    df['taker_buy_volume'].values / volume,
                    np.nan
                )

            if len(df) >= 20:
                buy_pressure_ma = talib.SMA(features['buy_pressure'].values, timeperiod=20)
                features['order_flow_imbalance'] = buy_pressure_ma - 0.5

        return features

    def _calculate_trend_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2:
            return features

        close_prices = df['close'].values

        if len(df) >= 20:
            returns = talib.ROC(close_prices, timeperiod=1) / 100.0
            rolling_mean = talib.SMA(returns, timeperiod=20)
            rolling_std = talib.STDDEV(returns, timeperiod=20)

            features['sharpe_ratio_20'] = np.where(
                rolling_std > 0,
                rolling_mean * np.sqrt(252) / (rolling_std * np.sqrt(252)),
                0
            )

        return features