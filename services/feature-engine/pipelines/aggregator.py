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
        if not features_dict:
            return pd.DataFrame()

        base_timeframe = self._get_base_timeframe(list(features_dict.keys()))
        base_features = features_dict.get(base_timeframe)

        if base_features is None or base_features.empty:
            return pd.DataFrame()

        base_features = base_features.copy()

        for timeframe, features_df in features_dict.items():
            if timeframe == base_timeframe or features_df.empty:
                continue

            aligned_features = self._align_timeframes(
                base_features, features_df, base_timeframe, timeframe
            )

            if not aligned_features.empty:
                for col in aligned_features.columns:
                    if col not in ['symbol', 'timestamp']:
                        new_col_name = f'{col}_{timeframe}'
                        base_features[new_col_name] = aligned_features[col]

        return base_features

    def _get_base_timeframe(self, timeframes: List[str]) -> str:
        sorted_timeframes = sorted(
            timeframes,
            key=lambda x: self.timeframe_mappings.get(x, 1440)
        )
        return sorted_timeframes[0] if sorted_timeframes else '1h'

    def _align_timeframes(self, base_df: pd.DataFrame, target_df: pd.DataFrame,
                          base_timeframe: str, target_timeframe: str) -> pd.DataFrame:
        base_minutes = self.timeframe_mappings.get(base_timeframe, 60)
        target_minutes = self.timeframe_mappings.get(target_timeframe, 60)

        if target_minutes == base_minutes:
            return target_df

        if target_minutes < base_minutes:
            return self._downsample_features(target_df, base_minutes / target_minutes)
        else:
            return self._upsample_features(target_df, base_df.index)

    def _downsample_features(self, df: pd.DataFrame, factor: float) -> pd.DataFrame:
        if df.empty or factor <= 1:
            return df

        factor = int(factor)
        resampled_index = df.index[::factor]
        resampled = pd.DataFrame(index=resampled_index)

        for col in df.columns:
            if col in ['symbol', 'timestamp']:
                continue

            base_col = col.split('_')[0]
            agg_func = self.aggregation_functions.get(base_col, 'mean')

            if agg_func == 'first':
                resampled[col] = df[col].iloc[::factor].values
            elif agg_func == 'last':
                resampled[col] = df[col].iloc[factor - 1::factor].values
            elif agg_func == 'max':
                values = []
                for i in range(0, len(df) - factor + 1, factor):
                    values.append(df[col].iloc[i:i + factor].max())
                resampled[col] = values[:len(resampled_index)]
            elif agg_func == 'min':
                values = []
                for i in range(0, len(df) - factor + 1, factor):
                    values.append(df[col].iloc[i:i + factor].min())
                resampled[col] = values[:len(resampled_index)]
            elif agg_func == 'sum':
                values = []
                for i in range(0, len(df) - factor + 1, factor):
                    values.append(df[col].iloc[i:i + factor].sum())
                resampled[col] = values[:len(resampled_index)]
            else:
                values = []
                for i in range(0, len(df) - factor + 1, factor):
                    values.append(df[col].iloc[i:i + factor].mean())
                resampled[col] = values[:len(resampled_index)]

        return resampled

    def _upsample_features(self, df: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(index=target_index)

        return df.reindex(target_index, method='ffill')

    def create_lag_features(self, df: pd.DataFrame, feature_columns: List[str],
                            lag_periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        if df.empty:
            return df

        lag_features = pd.DataFrame(index=df.index)

        for col in feature_columns:
            if col in df.columns:
                for lag in lag_periods:
                    lag_features[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return pd.concat([df, lag_features], axis=1)

    def create_rolling_features(self, df: pd.DataFrame, feature_columns: List[str],
                                windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        if df.empty:
            return df

        rolling_features = pd.DataFrame(index=df.index)

        for col in feature_columns:
            if col in df.columns:
                for window in windows:
                    if len(df) >= window:
                        rolling_features[f'{col}_mean_{window}'] = df[col].rolling(window=window).mean()
                        rolling_features[f'{col}_std_{window}'] = df[col].rolling(window=window).std()
                        rolling_features[f'{col}_min_{window}'] = df[col].rolling(window=window).min()
                        rolling_features[f'{col}_max_{window}'] = df[col].rolling(window=window).max()

        return pd.concat([df, rolling_features], axis=1)