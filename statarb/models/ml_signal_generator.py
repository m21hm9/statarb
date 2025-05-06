import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    Generate features for machine learning models from price data.
    """

    def __init__(
            self,
            lookback_periods: List[int] = [5, 10, 20, 60, 120],
            use_ta: bool = True
    ):
        """
        Initialize the feature generator.

        Args:
            lookback_periods: List of lookback periods for features
            use_ta: Whether to use technical analysis features
        """
        self.lookback_periods = lookback_periods
        self.use_ta = use_ta

    def create_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from price data.

        Args:
            price_data: DataFrame of price data

        Returns:
            DataFrame with features
        """
        features = pd.DataFrame(index=price_data.index)

        # Loop through each asset
        for col in price_data.columns:
            # Calculate returns
            for period in self.lookback_periods:
                # Absolute return
                features[f'{col}_return_{period}d'] = price_data[col].pct_change(period)

                # Log return
                features[f'{col}_log_return_{period}d'] = np.log(price_data[col] / price_data[col].shift(period))

            # Simple 1-day return (target variable for prediction)
            features[f'{col}_return_1d'] = price_data[col].pct_change(1)

            # Moving averages
            for period in self.lookback_periods:
                features[f'{col}_ma_{period}d'] = price_data[col].rolling(window=period).mean()

                # Distance from moving average
                features[f'{col}_dist_ma_{period}d'] = price_data[col] / features[f'{col}_ma_{period}d'] - 1

            # Volatility
            for period in self.lookback_periods:
                features[f'{col}_vol_{period}d'] = price_data[col].pct_change().rolling(window=period).std()

            # Technical indicators (if requested)
            if self.use_ta:
                try:
                    import talib

                    # RSI
                    features[f'{col}_rsi_14'] = talib.RSI(price_data[col].values, timeperiod=14)

                    # MACD
                    macd, macdsignal, macdhist = talib.MACD(
                        price_data[col].values,
                        fastperiod=12,
                        slowperiod=26,
                        signalperiod=9
                    )
                    features[f'{col}_macd'] = macd
                    features[f'{col}_macdsignal'] = macdsignal
                    features[f'{col}_macdhist'] = macdhist

                    # Bollinger Bands
                    upperband, middleband, lowerband = talib.BBANDS(
                        price_data[col].values,
                        timeperiod=20
                    )
                    features[f'{col}_bb_upper'] = upperband
                    features[f'{col}_bb_middle'] = middleband
                    features[f'{col}_bb_lower'] = lowerband

                    # Percentage distance from upper and lower bands
                    features[f'{col}_bb_upper_dist'] = price_data[col] / upperband - 1
                    features[f'{col}_bb_lower_dist'] = price_data[col] / lowerband - 1

                except ImportError:
                    logger.warning("TA-Lib not available. Skipping technical indicators.")
                    # Simple alternatives to technical indicators

                    # Approximate RSI using simple python
                    delta = price_data[col].diff()
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    features[f'{col}_rsi_14'] = 100 - (100 / (1 + rs))

        # Drop rows with NaN values
        features = features.dropna()

        return features

    def select_features(
            self,
            features: pd.DataFrame,
            target_col: str,
            method: str = 'correlation',
            n_features: int = 20
    ) -> pd.DataFrame:
        """
        Select relevant features.

        Args:
            features: DataFrame with features
            target_col: Target column name
            method: Feature selection method ('correlation', 'f_regression', or 'mutual_info')
            n_features: Number of features to select

        Returns:
            DataFrame with selected features
        """
        # Skip if the dataset is too small
        if len(features) < 30:
            return features

        if method == 'correlation':
            # Calculate correlation with target
            correlations = features.corrwith(features[target_col]).abs().sort_values(ascending=False)

            # Select top N features
            selected_cols = [target_col] + correlations[1:n_features + 1].index.tolist()

        elif method == 'f_regression':
            # Separate features from target
            X = features.drop(columns=[target_col])
            y = features[target_col]

            # Apply f_regression
            selector = SelectKBest(f_regression, k=n_features)
            selector.fit(X, y)

            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)

            # Get selected feature names
            selected_cols = [target_col] + [X.columns[i] for i in selected_indices]

        else:
            # Default to using all features
            selected_cols = features.columns

        return features[selected_cols]


class XGBoostModel:
    """
    XGBoost model for return prediction and signal generation.
    """

    def __init__(
            self,
            prediction_horizon: int = 5,
            train_test_split_ratio: float = 0.8,
            feature_selection_method: str = 'correlation',
            num_features: int = 20,
            xgb_params: Optional[Dict] = None
    ):
        """
        Initialize the XGBoost model.

        Args:
            prediction_horizon: Prediction horizon in days
            train_test_split_ratio: Train-test split ratio
            feature_selection_method: Method for feature selection
            num_features: Number of features to use
            xgb_params: XGBoost parameters
        """
        self.prediction_horizon = prediction_horizon
        self.train_test_split_ratio = train_test_split_ratio
        self.feature_selection_method = feature_selection_method
        self.num_features = num_features

        # Default XGBoost parameters if not provided
        if xgb_params is None:
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 100,
                'random_state': 42
            }
        else:
            self.xgb_params = xgb_params

        # Initialize model
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.selected_features = None

    def train(
            self,
            features: pd.DataFrame,
            target_col: str,
            cutoff_date: Optional[str] = None
    ) -> Dict:
        """
        Train the XGBoost model.

        Args:
            features: DataFrame with features
            target_col: Target column name
            cutoff_date: Optional cutoff date for train-test split

        Returns:
            Dictionary with training results
        """
        # Select features
        feature_generator = FeatureGenerator()
        selected_features = feature_generator.select_features(
            features,
            target_col,
            method=self.feature_selection_method,
            n_features=self.num_features
        )

        # Store selected feature names
        self.selected_features = selected_features.columns.tolist()
        self.feature_names = [f for f in self.selected_features if f != target_col]

        # Split data
        if cutoff_date is not None:
            train_data = selected_features[selected_features.index <= cutoff_date]
            test_data = selected_features[selected_features.index > cutoff_date]
        else:
            # Split data by ratio
            train_size = int(len(selected_features) * self.train_test_split_ratio)
            train_data = selected_features.iloc[:train_size]
            test_data = selected_features.iloc[train_size:]

        # Separate features and target
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]

        X_test = test_data.drop(columns=[target_col])
        y_test = test_data[target_col]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize and train XGBoost model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_train_scaled, y_train)

        # Make predictions on test set
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Create results dictionary
        results = {
            'model': self.model,
            'train_data': train_data,
            'test_data': test_data,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': test_rmse,  # For backward compatibility
                'r2': test_r2  # For backward compatibility
            },
            'feature_importance': {
                name: score for name, score in zip(
                    self.feature_names,
                    self.model.feature_importances_
                )
            }
        }

        return results

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Make predictions on new data.

        Args:
            features: DataFrame with features

        Returns:
            Series with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Check if all selected features are available
        missing_features = [f for f in self.feature_names if f not in features.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Extract selected features
        X = features[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        # Return predictions as a Series
        return pd.Series(predictions, index=features.index)

    def generate_signals(
            self,
            predictions: pd.Series,
            threshold: float = 0.0,
            long_percentile: float = 0.8,
            short_percentile: float = 0.2
    ) -> pd.DataFrame:
        """
        Generate trading signals from predictions.

        Args:
            predictions: Series with predictions
            threshold: Minimum absolute prediction for a signal
            long_percentile: Percentile for long signals
            short_percentile: Percentile for short signals

        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=predictions.index)
        signals['prediction'] = predictions

        # Absolute threshold approach
        signals['signal'] = 0
        signals.loc[predictions > threshold, 'signal'] = 1
        signals.loc[predictions < -threshold, 'signal'] = -1

        # Percentile approach if requested
        if 0 < long_percentile < 1 and 0 < short_percentile < 1:
            # Calculate percentiles
            upper_threshold = predictions.quantile(long_percentile)
            lower_threshold = predictions.quantile(short_percentile)

            # Generate signals
            signals['signal'] = 0
            signals.loc[predictions > upper_threshold, 'signal'] = 1
            signals.loc[predictions < lower_threshold, 'signal'] = -1

        return signals

    def save_model(self, model_dir: str, model_name: str):
        """
        Save the model to disk.

        Args:
            model_dir: Directory to save model
            model_name: Name of model file
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Save XGBoost model
        model_path = os.path.join(model_dir, f"{model_name}.json")
        self.model.save_model(model_path)

        # Save scaler
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)

        # Save feature names
        features_path = os.path.join(model_dir, f"{model_name}_features.pkl")
        joblib.dump({
            'feature_names': self.feature_names,
            'selected_features': self.selected_features
        }, features_path)

        logger.info(f"Model saved to {model_dir}")

    def load_model(self, model_dir: str, model_name: str):
        """
        Load a model from disk.

        Args:
            model_dir: Directory with model
            model_name: Name of model file
        """
        # Load XGBoost model
        model_path = os.path.join(model_dir, f"{model_name}.json")
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        # Load scaler
        scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
        self.scaler = joblib.load(scaler_path)

        # Load feature names
        features_path = os.path.join(model_dir, f"{model_name}_features.pkl")
        features_data = joblib.load(features_path)
        self.feature_names = features_data['feature_names']
        self.selected_features = features_data['selected_features']

        logger.info(f"Model loaded from {model_dir}")

    def plot_feature_importance(self) -> plt.Figure:
        """
        Plot feature importances.

        Returns:
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Get feature importances
        importances = self.model.feature_importances_

        # Sort features by importance
        indices = np.argsort(importances)[::-1]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot feature importances
        ax.bar(range(len(importances)), importances[indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([self.feature_names[i] for i in indices], rotation=90)
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title('XGBoost Feature Importance')

        plt.tight_layout()

        return fig

    def plot_predictions(self, results: Dict) -> plt.Figure:
        """
        Plot predictions against actual values.

        Args:
            results: Dictionary with training results

        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot training set predictions
        train_data = results['train_data']
        y_train = train_data.iloc[:, 0]  # Assuming target is the first column
        y_pred_train = results['y_pred_train']

        ax1.scatter(y_train, y_pred_train, alpha=0.3)
        ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Training Set: Actual vs Predicted')

        # Plot test set predictions
        test_data = results['test_data']
        y_test = test_data.iloc[:, 0]  # Assuming target is the first column
        y_pred_test = results['y_pred_test']

        ax2.scatter(y_test, y_pred_test, alpha=0.3)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Test Set: Actual vs Predicted')

        plt.tight_layout()

        return fig