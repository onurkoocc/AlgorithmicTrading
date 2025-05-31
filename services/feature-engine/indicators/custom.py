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
            shifted_close = df['close'].shift(period)
            mask = shifted_close > 0
            features[f'price_change_{period}'] = pd.Series(0.0, index=df.index)
            features.loc[mask, f'price_change_{period}'] = (df['close'][mask] - shifted_close[mask]) / shifted_close[
                mask]

        return features

    def _calculate_returns(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        for period in self.lookback_periods:
            shifted_close = df['close'].shift(period)
            mask = shifted_close > 0
            features[f'log_return_{period}'] = pd.Series(0.0, index=df.index)
            features.loc[mask, f'log_return_{period}'] = np.log(df['close'][mask] / shifted_close[mask])

        return features

    def _calculate_volatility(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        shifted_close = df['close'].shift(1)
        mask = shifted_close > 0
        returns = pd.Series(0.0, index=df.index)
        returns[mask] = np.log(df['close'][mask] / shifted_close[mask])

        features['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        features['volatility_50'] = returns.rolling(window=50).std() * np.sqrt(252)

        returns_mean = returns.rolling(window=20).mean()
        returns_std = returns.rolling(window=20).std()

        features['sharpe_ratio_20'] = pd.Series(0.0, index=df.index)
        std_mask = returns_std > 0
        features.loc[std_mask, 'sharpe_ratio_20'] = (returns_mean[std_mask] * 252) / (
                    returns_std[std_mask] * np.sqrt(252))

        return features

    def _calculate_candle_patterns(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        low_mask = df['low'] > 0
        features['high_low_ratio'] = pd.Series(1.0, index=df.index)
        features.loc[low_mask, 'high_low_ratio'] = df['high'][low_mask] / df['low'][low_mask]

        open_mask = df['open'] > 0
        features['close_open_ratio'] = pd.Series(1.0, index=df.index)
        features.loc[open_mask, 'close_open_ratio'] = df['close'][open_mask] / df['open'][open_mask]

        hl_range = df['high'] - df['low']
        range_mask = hl_range > 0

        features['upper_shadow'] = pd.Series(0.0, index=df.index)
        features['lower_shadow'] = pd.Series(0.0, index=df.index)
        features['body_size'] = pd.Series(0.0, index=df.index)

        if range_mask.any():
            max_oc = np.maximum(df['open'][range_mask], df['close'][range_mask])
            min_oc = np.minimum(df['open'][range_mask], df['close'][range_mask])

            features.loc[range_mask, 'upper_shadow'] = (df['high'][range_mask] - max_oc) / hl_range[range_mask]
            features.loc[range_mask, 'lower_shadow'] = (min_oc - df['low'][range_mask]) / hl_range[range_mask]
            features.loc[range_mask, 'body_size'] = np.abs(df['close'][range_mask] - df['open'][range_mask]) / hl_range[
                range_mask]

        features['is_bullish_candle'] = df['close'] > df['open']

        return features

    def _calculate_trend_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        sma_200 = df['close'].rolling(window=200).mean()

        features['trend_strength'] = pd.Series(0.0, index=df.index)
        sma_mask = sma_200 > 0
        features.loc[sma_mask, 'trend_strength'] = (df['close'][sma_mask] - sma_200[sma_mask]) / sma_200[sma_mask]

        ema_short = df['close'].ewm(span=12, adjust=False).mean()
        ema_long = df['close'].ewm(span=26, adjust=False).mean()

        features['momentum_score'] = pd.Series(0.0, index=df.index)
        ema_mask = ema_long > 0
        features.loc[ema_mask, 'momentum_score'] = (ema_short[ema_mask] - ema_long[ema_mask]) / ema_long[ema_mask]

        return features

    def _calculate_momentum_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        volume_sma_5 = df['volume'].rolling(window=5).mean()
        volume_sma_20 = df['volume'].rolling(window=20).mean()

        features['volume_momentum'] = pd.Series(1.0, index=df.index)
        vol_mask = volume_sma_20 > 0
        features.loc[vol_mask, 'volume_momentum'] = volume_sma_5[vol_mask] / volume_sma_20[vol_mask]

        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        price_range = high_20 - low_20

        features['price_position'] = pd.Series(0.5, index=df.index)
        price_mask = price_range > 0
        features.loc[price_mask, 'price_position'] = (df['close'][price_mask] - low_20[price_mask]) / price_range[
            price_mask]

        return features

    def _calculate_market_regime(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        shifted_close = df['close'].shift(1)
        mask = shifted_close > 0
        returns = pd.Series(0.0, index=df.index)
        returns[mask] = np.log(df['close'][mask] / shifted_close[mask])

        volatility = returns.rolling(window=20).std()
        volatility_median = volatility.rolling(window=100).median()
        trend = df['close'].rolling(window=50).mean().diff(10)

        conditions = [
            (trend > 0) & (volatility < volatility_median),
            (trend > 0) & (volatility >= volatility_median),
            (trend <= 0) & (volatility < volatility_median),
            (trend <= 0) & (volatility >= volatility_median)
        ]

        values = [1, 2, 3, 4]

        features['market_regime'] = np.select(conditions, values, default=0)

        return features