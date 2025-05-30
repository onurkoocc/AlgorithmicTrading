import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy import stats


class DataQualityMonitor:
    def __init__(self):
        self.quality_thresholds = {
            'missing_percentage': 5.0,
            'outlier_percentage': 1.0,
            'correlation_threshold': 0.95,
            'zero_variance_threshold': 1e-10
        }

        self.feature_stats = defaultdict(dict)

    def check_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        quality_report = {
            'total_rows': len(df),
            'total_features': len(df.columns),
            'missing_analysis': self._analyze_missing_data(df),
            'outlier_analysis': self._analyze_outliers(df),
            'correlation_analysis': self._analyze_correlations(df),
            'variance_analysis': self._analyze_variance(df),
            'data_integrity': self._check_data_integrity(df),
            'feature_quality_scores': self._calculate_feature_quality_scores(df)
        }

        quality_report['overall_quality_score'] = self._calculate_overall_quality_score(quality_report)
        quality_report['recommendations'] = self._generate_recommendations(quality_report)

        return quality_report

    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, any]:
        missing_counts = df.isnull().sum()
        missing_percentage = (missing_counts / len(df)) * 100

        return {
            'total_missing': int(missing_counts.sum()),
            'missing_percentage': float(missing_percentage.mean()),
            'features_with_missing': {
                col: float(pct) for col, pct in missing_percentage[missing_percentage > 0].items()
            },
            'critical_missing_features': [
                col for col, pct in missing_percentage.items()
                if pct > self.quality_thresholds['missing_percentage']
            ]
        }

    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        outlier_details = {}

        for col in numeric_cols:
            if col in ['symbol', 'timestamp', 'feature_version', 'market_regime']:
                continue

            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = z_scores > 3
            outlier_count = outliers.sum()

            if outlier_count > 0:
                outlier_counts[col] = int(outlier_count)
                outlier_percentage = (outlier_count / len(df[col].dropna())) * 100

                if outlier_percentage > self.quality_thresholds['outlier_percentage']:
                    outlier_details[col] = {
                        'count': int(outlier_count),
                        'percentage': float(outlier_percentage),
                        'max_zscore': float(z_scores.max())
                    }

        return {
            'total_outliers': sum(outlier_counts.values()),
            'outlier_features': outlier_counts,
            'critical_outlier_features': outlier_details
        }

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['symbol', 'timestamp', 'feature_version']]

        if len(numeric_cols) < 2:
            return {'highly_correlated_pairs': []}

        corr_matrix = df[numeric_cols].corr()
        highly_correlated = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > self.quality_thresholds['correlation_threshold']:
                    highly_correlated.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })

        return {
            'highly_correlated_pairs': highly_correlated,
            'max_correlation': float(corr_matrix.abs().values[~np.eye(len(corr_matrix), dtype=bool)].max())
            if len(corr_matrix) > 1 else 0.0
        }

    def _analyze_variance(self, df: pd.DataFrame) -> Dict[str, any]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_variance_features = []

        for col in numeric_cols:
            if col in ['symbol', 'timestamp', 'feature_version', 'is_bullish_candle']:
                continue

            variance = df[col].var()
            if variance < self.quality_thresholds['zero_variance_threshold']:
                low_variance_features.append({
                    'feature': col,
                    'variance': float(variance)
                })

        return {
            'low_variance_features': low_variance_features,
            'zero_variance_count': len([f for f in low_variance_features if f['variance'] == 0])
        }

    def _check_data_integrity(self, df: pd.DataFrame) -> Dict[str, any]:
        integrity_issues = []

        if 'high' in df.columns and 'low' in df.columns:
            invalid_high_low = df['high'] < df['low']
            if invalid_high_low.any():
                integrity_issues.append({
                    'type': 'invalid_high_low',
                    'count': int(invalid_high_low.sum()),
                    'percentage': float((invalid_high_low.sum() / len(df)) * 100)
                })

        for price_col in ['open', 'high', 'low', 'close']:
            if price_col in df.columns:
                negative_prices = df[price_col] < 0
                if negative_prices.any():
                    integrity_issues.append({
                        'type': f'negative_{price_col}',
                        'count': int(negative_prices.sum()),
                        'percentage': float((negative_prices.sum() / len(df)) * 100)
                    })

        if 'volume' in df.columns:
            negative_volume = df['volume'] < 0
            if negative_volume.any():
                integrity_issues.append({
                    'type': 'negative_volume',
                    'count': int(negative_volume.sum()),
                    'percentage': float((negative_volume.sum() / len(df)) * 100)
                })

        return {
            'integrity_issues': integrity_issues,
            'is_valid': len(integrity_issues) == 0
        }

    def _calculate_feature_quality_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        feature_scores = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['symbol', 'timestamp', 'feature_version']:
                continue

            score = 100.0

            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            score -= min(missing_pct * 2, 50)

            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_pct = ((z_scores > 3).sum() / len(df[col].dropna())) * 100
            score -= min(outlier_pct * 5, 30)

            variance = df[col].var()
            if variance < self.quality_thresholds['zero_variance_threshold']:
                score -= 20

            feature_scores[col] = max(score, 0.0)

        return feature_scores

    def _calculate_overall_quality_score(self, quality_report: Dict[str, any]) -> float:
        score = 100.0

        missing_penalty = min(quality_report['missing_analysis']['missing_percentage'] * 2, 30)
        score -= missing_penalty

        outlier_ratio = (quality_report['outlier_analysis']['total_outliers'] /
                         (quality_report['total_rows'] * quality_report['total_features'])) * 100
        outlier_penalty = min(outlier_ratio * 5, 20)
        score -= outlier_penalty

        correlation_penalty = len(quality_report['correlation_analysis']['highly_correlated_pairs']) * 2
        score -= min(correlation_penalty, 20)

        variance_penalty = len(quality_report['variance_analysis']['low_variance_features']) * 2
        score -= min(variance_penalty, 20)

        if not quality_report['data_integrity']['is_valid']:
            score -= 10

        return max(score, 0.0)

    def _generate_recommendations(self, quality_report: Dict[str, any]) -> List[str]:
        recommendations = []

        if quality_report['missing_analysis']['critical_missing_features']:
            recommendations.append(
                f"High missing data in features: {', '.join(quality_report['missing_analysis']['critical_missing_features'])[:100]}..."
            )

        if quality_report['outlier_analysis']['critical_outlier_features']:
            recommendations.append(
                f"Significant outliers in: {', '.join(quality_report['outlier_analysis']['critical_outlier_features'].keys())[:100]}..."
            )

        if quality_report['correlation_analysis']['highly_correlated_pairs']:
            recommendations.append(
                f"Remove highly correlated features to reduce redundancy ({len(quality_report['correlation_analysis']['highly_correlated_pairs'])} pairs found)"
            )

        if quality_report['variance_analysis']['zero_variance_count'] > 0:
            recommendations.append(
                f"Remove {quality_report['variance_analysis']['zero_variance_count']} zero-variance features"
            )

        if not quality_report['data_integrity']['is_valid']:
            recommendations.append("Fix data integrity issues before using features")

        if quality_report['overall_quality_score'] < 70:
            recommendations.append("Overall quality score is low - review data pipeline")

        return recommendations