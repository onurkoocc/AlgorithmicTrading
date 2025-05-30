import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class MultiTimeframeAggregator:
    def __init__(self):
        self.timeframe_mappings = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '8h': 480,
            '12h': 720,
            '1d': 1440
        }

        self.aggregation_functions = {
            'close': 'last',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'volume': 'sum',
            'sma': 'mean',
            'ema': 'last',
            'rsi': 'mean',
            'atr': 'mean',
            'volatility': 'mean'
        }

    def aggregate_features(self, features_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        base_timeframe = self._get_base_timeframe(features_dict.keys())
        base_features = features_dict[base_timeframe].copy()

        for timeframe, features_df in features_dict.items():
            if timeframe == base_timeframe:
                continue

            aligned_features = self._align_timeframes(base_features, features_df, base_timeframe, timeframe)

            for col in aligned_features.columns:
                if col not in ['symbol', 'timestamp']:
                    base_features[f'{col}_{timeframe}'] = aligned_features[col]

        return base_features

    def _get_base_timeframe(self, timeframes: List[str]) -> str:
        sorted_timeframes = sorted(timeframes, key=lambda x: self.timeframe_mappings.get(x, 1440))
        return sorted_timeframes[0]

    def _align_timeframes(self, base_df: pd.DataFrame, target_df: pd.DataFrame,
                          base_timeframe: str, target_timeframe: str) -> pd.DataFrame:
        base_minutes = self.timeframe_mappings[base_timeframe]
        target_minutes = self.timeframe_mappings[target_timeframe]

        if target_minutes <= base_minutes:
            return self._downsample_features(target_df, base_minutes / target_minutes)
        else:
            return self._upsample_features(target_df, base_df.index)

    def _downsample_features(self, df: pd.DataFrame, factor: float) -> pd.DataFrame:
        resampled = pd.DataFrame(index=df.index[::int(factor)])

        for col in df.columns:
            base_col = col.split('_')[0]
            agg_func = self.aggregation_functions.get(base_col, 'mean')

            if agg_func == 'first':
                resampled[col] = df[col].iloc[::int(factor)]
            elif agg_func == 'last':
                resampled[col] = df[col].iloc[int(factor)-1::int(factor)]
            elif agg_func == 'max':
                resampled[col] = df[col].rolling(window=int(factor)).max().iloc[int(factor)-1::int(factor)]
            elif agg_func == 'min':
                resampled[col] = df[col].rolling(window=int(factor)).min().iloc[int(factor)-1::int(factor)]
            elif agg_func == 'sum':
                resampled[col] = df[col].rolling(window=int(factor)).sum().iloc[int(factor)-1::int(factor)]
            else:
                resampled[col] = df[col].rolling(window=int(factor)).mean().iloc[int(factor)-1::int(factor)]

        return resampled

    def _upsample_features(self, df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        return df.reindex(target_index, method='ffill')

    def create_lag_features(self, df: pd.DataFrame, feature_columns: List[str],
                            lag_periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        lag_features = pd.DataFrame(index=df.index)

        for col in feature_columns:
            if col in df.columns:
                for lag in lag_periods:
                    lag_features[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return pd.concat([df, lag_features], axis=1)

    def create_rolling_features(self, df: pd.DataFrame, feature_columns: List[str],
                                windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        rolling_features = pd.DataFrame(index=df.index)

        for col in feature_columns:
            if col in df.columns:
                for window in windows:
                    rolling_features[f'{col}_mean_{window}'] = df[col].rolling(window=window).mean()
                    rolling_features[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                    rolling_features[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                    rolling_features[f'{col}_max_{window}'] = df[col].rolling(window=window).max()

        return pd.concat([df, rolling_features], axis=1)

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        interaction_features = pd.DataFrame(index=df.index)

        if 'rsi_14' in df.columns and 'volume_ratio' in df.columns:
            interaction_features['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio']

        if 'macd' in df.columns and 'atr_14' in df.columns:
            interaction_features['macd_volatility_ratio'] = df['macd'] / (df['atr_14'] + 1e-10)

        if 'bb_percent' in df.columns and 'volume_momentum' in df.columns:
            interaction_features['bb_volume_signal'] = df['bb_percent'] * df['volume_momentum']

        if 'adx_14' in df.columns and 'trend_strength' in df.columns:
            interaction_features['trend_confirmation'] = df['adx_14'] * df['trend_strength']

        return pd.concat([df, interaction_features], axis=1)