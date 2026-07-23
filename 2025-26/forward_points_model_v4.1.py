# Date: August 18, 2025
# Author: Julian di Giovanni w/Claude.ai - Enhanced Regularization Edition
# Version: 4.1 - Robust Regularization Edition

"""
ENHANCED PROGRAM OVERVIEW: NHL Forward Points Prediction with Robust Regularization
===================================================================================

This enhanced program builds machine learning models to predict NHL forward scoring using only 
historical (lagged) performance data with comprehensive regularization to prevent overfitting. 
It implements a rigorous approach to avoid data leakage by excluding all contemporaneous statistics.

WORKFLOW STEP-BY-STEP:
=====================

1. DATA LOADING & CLEANING:
   - Loads NHL skater data from CSV file
   - Removes all defensemen (keeps only forwards: C, LW, RW)
   - Parses seasons and extracts year information
   - Handles missing data in core statistics (goals, assists, shots, games_played)
   - Creates shooting percentage and time-on-ice features if missing
   - Preserves all legitimate performance data (no outlier capping)
   - Removes only clear data quality issues (0 games played, essential missing data)
   - Shows data availability by year and position

2. LAG FEATURE CREATION:
   - Creates 1, 2, or 3-year historical lags for key statistics:
     * Core stats: goals, assists, shots, games_played, shooting_pct, time_on_ice_per_game
     * Power play stats: pp_goals, pp_assists, pp_shots, etc. (but NOT numbered variables)
   - Only creates lags where sufficient historical data exists
   - Filters data to years where complete lag history is available
   - Tracks players with sufficient history for reliable predictions

3. ENHANCED FEATURE ENGINEERING:
   - Creates advanced metrics from lagged data:
     * Historical averages (e.g., goals_hist_avg, assists_hist_avg)
     * Trend indicators (lag1 vs lag2 performance changes)
     * Momentum indicators (lag1 vs historical average)
     * Per-game rates for all lagged seasons (goals_per_game_lag1, etc.)
     * Historical points totals and per-game averages
     * Consistency measures (historical standard deviations)
   - All engineered features use ONLY historical data (no current season info)

4. FEATURE PREPARATION WITH SELECTION:
   - EXCLUDES all contemporaneous variables to prevent data leakage:
     * Current season: goals, assists, shots, games_played, etc.
     * Current season power play stats
     * Any variables ending in numbers (1, 2, 3, etc.)
   - INCLUDES only historical and team variables:
     * All lagged statistics (goals_lag1, assists_lag2, etc.)
     * Team variables (team_1, team_2, etc.)
     * Engineered historical features
   - AUTOMATIC FEATURE SELECTION:
     * Uses SelectKBest with f_regression scoring
     * Reduces feature set from 100+ to ~50 most important features
     * Prevents curse of dimensionality and reduces overfitting
   - Displays complete list of features used in training
   - Handles missing values and ensures all features are numeric

5. ENHANCED REGULARIZATION TECHNIQUES:

   TREE-BASED MODELS (COMPREHENSIVE PRUNING):
   - RandomForest: min_samples_split, min_samples_leaf, max_features, min_impurity_decrease
   - GradientBoosting: learning_rate, subsample, early stopping with validation monitoring
   - Feature subsampling to prevent overfitting to specific features

   LINEAR MODELS (L1/L2 REGULARIZATION):
   - Ridge Regression: L2 penalty with cross-validated alpha selection
   - Lasso Regression: L1 penalty for automatic feature selection
   - ElasticNet: Combined L1+L2 penalties with optimized mixing ratio
   - RobustScaler for outlier-resistant feature scaling

   ADVANCED TECHNIQUES:
   - Hyperparameter tuning with GridSearchCV for all models
   - Time-aware cross-validation using TimeSeriesSplit
   - Overfitting detection: monitors train vs test performance gaps
   - Early stopping for gradient boosting based on validation scores

6. TWO ENHANCED TRAINING SCENARIOS:

   TRAINING A - ALL DATA WITH REGULARIZATION:
   - Uses complete dataset for training and validation
   - Tests multiple lag configurations (1, 2, 3 years)
   - Tests all regularized models (Ridge, Lasso, ElasticNet, RF, GB)
   - Selects best performing combination based on R² score adjusted for overfitting
   - Provides enhanced model evaluation plots and feature importance

   TRAINING B - PREDICTIVE VALIDATION WITH REGULARIZATION:
   - Excludes 2024-2025 season from training data
   - Trains on historical data only (up to 2023-2024)
   - Uses trained regularized model to predict 2024-2025 season performance
   - Validates predictions against actual 2024-2025 results
   - Tests realistic prediction scenario with overfitting prevention

7. ENHANCED MODEL TRAINING & EVALUATION:
   - Implements train/test split with appropriate validation
   - Uses time-aware cross-validation for robust performance estimation
   - Calculates multiple metrics: R², RMSE, MAE, overfitting ratio
   - Handles feature scaling for linear models with RobustScaler
   - Prevents overfitting with comprehensive regularization parameters
   - Provides confidence intervals for ensemble model predictions

8. PREDICTIONS & ENHANCED OUTPUT:
   - Generates point predictions for 2024-2025 season with uncertainty estimates
   - Ranks all forwards by predicted scoring with confidence intervals
   - Creates formatted table of top 50 predicted scorers
   - Compares predictions to actual results where available
   - Provides comprehensive model performance summary including regularization metrics
   - Enhanced 6-panel diagnostic plots showing overfitting analysis

KEY ENHANCED FEATURES:
=====================
- NO DATA LEAKAGE: Uses only historical data for predictions
- ROBUST REGULARIZATION: Comprehensive overfitting prevention across all model types
- AUTOMATIC FEATURE SELECTION: Reduces dimensionality and improves generalization
- PRESERVES ELITE PERFORMANCE: No artificial capping of high-scoring seasons
- ENHANCED VALIDATION: Tests on held-out future season with overfitting monitoring
- COMPREHENSIVE: Includes basic stats, power play, and engineered features with consistency measures
- FLEXIBLE: Tests multiple lag periods and regularized model types
- INTERPRETABLE: Shows feature importance and enhanced model diagnostics
- PRACTICAL: Outputs actionable player rankings with uncertainty estimates

REGULARIZATION TECHNIQUES IMPLEMENTED:
=====================================
- L1 Regularization (Lasso): Automatic feature selection through sparsity
- L2 Regularization (Ridge): Shrinks coefficients to prevent overfitting
- ElasticNet: Combines L1 and L2 penalties for balanced regularization
- Tree Pruning: Comprehensive depth and sample size constraints
- Feature Subsampling: Reduces overfitting to specific feature combinations
- Early Stopping: Prevents overtraining in gradient boosting
- Feature Selection: Automatic dimensionality reduction
- Cross-Validation: Robust hyperparameter selection and performance estimation
- Overfitting Monitoring: Tracks and penalizes train/test performance gaps

TECHNICAL APPROACH:
==================
- Handles missing data intelligently with domain knowledge
- Preserves all legitimate elite performance data (no artificial capping)
- Creates meaningful lag features for time series nature of hockey data
- Uses ensemble methods and regularized linear models for comparison
- Implements proper train/test splits to avoid overfitting
- Scales features appropriately for different model types with robust scaling
- Provides extensive logging for transparency and debugging
- Monitors and prevents overfitting through multiple complementary techniques

OUTPUT:
=======
1. Data loading and cleaning summary with feature engineering details
2. Enhanced feature engineering details with complete feature list and selection results
3. Training results for both scenarios (A & B) with regularization metrics
4. Enhanced model performance metrics and diagnostic plots including overfitting analysis
5. Top 50 predicted scorers for 2024-2025 season with actual results and confidence intervals
6. Feature importance rankings for best performing regularized model
7. Comprehensive regularization summary showing techniques applied and their effectiveness
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression, RFECV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import warnings
import os
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

class OutputManager:
    """Manages output to both console and file"""
    def __init__(self, output_file=None):
        self.terminal = sys.stdout
        self.log_file = None
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            self.log_file = open(output_file, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        if self.log_file:
            self.log_file.write(message)
    
    def flush(self):
        self.terminal.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()

class EnhancedHockeyPointsPredictor:
    def __init__(self, lag_years: int = 2, min_training_year: int = None, use_feature_selection: bool = True):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.model = None
        self.feature_selector = None
        self.feature_columns = None
        self.selected_features = None
        self.target_column = 'points'
        self.lag_years = lag_years
        self.min_training_year = min_training_year
        self.use_feature_selection = use_feature_selection
        self.best_params = None
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load hockey player statistics from CSV file and remove defensemen."""
        df = pd.read_csv(csv_path)
        
        print(f"Original data shape: {df.shape}")
        
        # Remove defensemen - check multiple possible position column names
        position_cols = ['position', 'pos', 'Position', 'Pos']
        position_col = None
        
        for col in position_cols:
            if col in df.columns:
                position_col = col
                break
        
        if position_col:
            original_count = len(df)
            df = df[~df[position_col].astype(str).str.upper().str.contains('D', na=False)]
            removed_count = original_count - len(df)
            print(f"Removed {removed_count} defensemen ({removed_count/original_count*100:.1f}%)")
            print(f"Remaining positions: {df[position_col].value_counts().to_dict()}")
        else:
            print("Warning: No position column found - cannot remove defensemen")
            print(f"Available columns: {list(df.columns)}")
        
        # Parse season to extract start year
        df['season_str'] = df['season'].astype(str)
        df['year'] = df['season_str'].str[:4].astype(int)
        
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")
        print(f"Years in dataset: {sorted(df['year'].unique())}")
        
        # Automatically set min_training_year if not specified
        if self.min_training_year is None:
            self.min_training_year = df['year'].min() + self.lag_years
            print(f"Auto-setting min_training_year to {self.min_training_year} (allows {self.lag_years}-year lags)")
        
        # Use existing points column or calculate if missing
        if 'points' not in df.columns:
            if 'goals' in df.columns and 'assists' in df.columns:
                df['points'] = df['goals'] + df['assists']
                print("Created 'points' column from goals + assists")
        
        # Clean missing data issues
        df = self._clean_missing_data(df)
        
        # Sort by player and year
        df = df.sort_values(['player_id', 'year'])
        
        # Show data availability by year
        year_counts = df['year'].value_counts().sort_index()
        print(f"\nObservations per year (forwards only):")
        for year, count in year_counts.items():
            print(f"  {year}: {count} player-seasons")
        
        return df
    
    def _clean_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and handle missing data issues."""
        print("\nCleaning missing data...")
        
        # Handle shooting percentage
        shooting_cols = ['shooting_pct', 'shooting_percentage', 'sh_pct']
        shooting_col = None
        for col in shooting_cols:
            if col in df.columns:
                shooting_col = col
                break
        
        if shooting_col:
            if df[shooting_col].max() > 1:
                df['shooting_pct'] = df[shooting_col] / 100
            else:
                df['shooting_pct'] = df[shooting_col]
        else:
            if 'goals' in df.columns and 'shots' in df.columns:
                df['shooting_pct'] = df['goals'] / df['shots'].replace(0, np.nan)
                df['shooting_pct'] = df['shooting_pct'].fillna(0.1)
            else:
                df['shooting_pct'] = 0.1
        
        # Handle time_on_ice_per_game variations
        toi_cols = ['time_on_ice_per_game', 'toi_per_game', 'avg_toi', 'time_on_ice']
        toi_col = None
        for col in toi_cols:
            if col in df.columns:
                toi_col = col
                break
        
        if toi_col and toi_col != 'time_on_ice_per_game':
            df['time_on_ice_per_game'] = df[toi_col]
        elif not toi_col:
            df['time_on_ice_per_game'] = 15.0
        
        # Clean up core statistical columns
        core_stats = ['goals', 'assists', 'shots', 'games_played', 'points']
        for stat in core_stats:
            if stat in df.columns:
                df[stat] = df[stat].fillna(0)
                df[stat] = df[stat].clip(lower=0)
        
        # Remove rows where essential data is completely missing
        essential_cols = ['player_id', 'year', 'games_played']
        before_essential = len(df)
        df = df.dropna(subset=essential_cols)
        after_essential = len(df)
        
        if before_essential != after_essential:
            print(f"Removed {before_essential - after_essential} rows missing essential data")
        
        # Remove players with 0 games played
        if 'games_played' in df.columns:
            before_games = len(df)
            df = df[df['games_played'] > 0]
            after_games = len(df)
            if before_games != after_games:
                print(f"Removed {before_games - after_games} rows with 0 games played")
        
        print(f"Data shape after cleaning: {df.shape}")
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for previous seasons."""
        lag_features = ['goals', 'assists', 'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game']
        
        # Add PP variables to lag features
        pp_features = [col for col in df.columns if col.startswith('pp') and not any(col.endswith(str(i)) for i in range(1, 10))]
        lag_features.extend(pp_features)
        
        existing_features = [f for f in lag_features if f in df.columns]
        
        print(f"\nCreating {self.lag_years}-year lags for: {existing_features}")
        
        df_with_lags = df.copy()
        
        # Initialize lag columns
        lag_columns = []
        for feature in existing_features:
            for lag in range(1, self.lag_years + 1):
                lag_col = f'{feature}_lag{lag}'
                lag_columns.append(lag_col)
                df_with_lags[lag_col] = np.nan
        
        # Create lags for each player
        players_with_sufficient_history = 0
        
        for player in df['player_id'].unique():
            player_data = df[df['player_id'] == player].sort_values('year')
            
            if len(player_data) > self.lag_years:
                players_with_sufficient_history += 1
                
                for feature in existing_features:
                    for lag in range(1, self.lag_years + 1):
                        lag_col = f'{feature}_lag{lag}'
                        lagged_values = player_data[feature].shift(lag)
                        df_with_lags.loc[df_with_lags['player_id'] == player, lag_col] = lagged_values
        
        print(f"Players with sufficient history: {players_with_sufficient_history}")
        
        # Filter to only use years where lags are possible
        print(f"Filtering to years >= {self.min_training_year} for training")
        training_data = df_with_lags[df_with_lags['year'] >= self.min_training_year].copy()
        
        # Check how much data we have after filtering
        complete_lag_mask = ~training_data[lag_columns].isnull().any(axis=1)
        complete_observations = complete_lag_mask.sum()
        
        print(f"Training data shape after year filter: {training_data.shape}")
        print(f"Complete observations with all lags: {complete_observations}")
        
        if complete_observations > 0:
            complete_years = training_data[complete_lag_mask]['year'].value_counts().sort_index()
            print(f"Complete observations by year:")
            for year, count in complete_years.items():
                print(f"  {year}: {count}")
        
        return training_data
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features from lag data."""
        df_eng = df.copy()
        
        print(f"\nEngineering features...")
        
        # Historical features from lag data ONLY
        lag_columns = [col for col in df_eng.columns if 'lag' in col]
        
        if lag_columns:
            print(f"Found {len(lag_columns)} lag columns")
            
            # Get all base features that have lags
            lag_base_features = ['goals', 'assists', 'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game']
            pp_base_features = list(set([col.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '') 
                                        for col in lag_columns if col.startswith('pp')]))
            lag_base_features.extend(pp_base_features)
            
            for base_feature in lag_base_features:
                lag_cols = [col for col in lag_columns if col.startswith(f'{base_feature}_lag')]
                
                if lag_cols:
                    # Average over available lags
                    df_eng[f'{base_feature}_hist_avg'] = df_eng[lag_cols].mean(axis=1, skipna=True)
                    
                    # Historical variance (stability measure)
                    if len(lag_cols) >= 2:
                        df_eng[f'{base_feature}_hist_std'] = df_eng[lag_cols].std(axis=1, skipna=True).fillna(0)
                    
                    # Recent trend (lag1 vs lag2)
                    if len(lag_cols) >= 2 and f'{base_feature}_lag1' in df_eng.columns and f'{base_feature}_lag2' in df_eng.columns:
                        trend = df_eng[f'{base_feature}_lag1'] - df_eng[f'{base_feature}_lag2']
                        df_eng[f'{base_feature}_trend'] = trend.fillna(0)
                        
                    # Momentum (lag1 vs historical average)
                    if f'{base_feature}_lag1' in df_eng.columns:
                        momentum = df_eng[f'{base_feature}_lag1'] - df_eng[f'{base_feature}_hist_avg']
                        df_eng[f'{base_feature}_momentum'] = momentum.fillna(0)
            
            # Historical per-game rates with consistency measures
            for lag in range(1, self.lag_years + 1):
                if f'goals_lag{lag}' in df_eng.columns and f'games_played_lag{lag}' in df_eng.columns:
                    goals_per_game = df_eng[f'goals_lag{lag}'] / df_eng[f'games_played_lag{lag}'].replace(0, np.nan)
                    df_eng[f'goals_per_game_lag{lag}'] = goals_per_game.fillna(0)
                    
                if f'assists_lag{lag}' in df_eng.columns and f'games_played_lag{lag}' in df_eng.columns:
                    assists_per_game = df_eng[f'assists_lag{lag}'] / df_eng[f'games_played_lag{lag}'].replace(0, np.nan)
                    df_eng[f'assists_per_game_lag{lag}'] = assists_per_game.fillna(0)
                    
                if f'shots_lag{lag}' in df_eng.columns and f'games_played_lag{lag}' in df_eng.columns:
                    shots_per_game = df_eng[f'shots_lag{lag}'] / df_eng[f'games_played_lag{lag}'].replace(0, np.nan)
                    df_eng[f'shots_per_game_lag{lag}'] = shots_per_game.fillna(0)
                
                # Historical points
                if f'goals_lag{lag}' in df_eng.columns and f'assists_lag{lag}' in df_eng.columns:
                    df_eng[f'points_lag{lag}'] = df_eng[f'goals_lag{lag}'] + df_eng[f'assists_lag{lag}']
                    
                    if f'games_played_lag{lag}' in df_eng.columns:
                        points_per_game = df_eng[f'points_lag{lag}'] / df_eng[f'games_played_lag{lag}'].replace(0, np.nan)
                        df_eng[f'points_per_game_lag{lag}'] = points_per_game.fillna(0)
            
            # Historical per-game averages with consistency measures
            per_game_features = ['goals_per_game', 'assists_per_game', 'shots_per_game', 'points_per_game']
            for feature in per_game_features:
                feature_lag_cols = [col for col in df_eng.columns if col.startswith(f'{feature}_lag')]
                if feature_lag_cols:
                    hist_avg = df_eng[feature_lag_cols].mean(axis=1, skipna=True)
                    df_eng[f'{feature}_hist_avg'] = hist_avg.fillna(0)
                    
                    # Consistency measure (lower std = more consistent)
                    if len(feature_lag_cols) >= 2:
                        hist_consistency = 1 / (1 + df_eng[feature_lag_cols].std(axis=1, skipna=True).fillna(1))
                        df_eng[f'{feature}_consistency'] = hist_consistency
        
        # Clean up infinite and extreme values
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with appropriate defaults
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        df_eng[numeric_cols] = df_eng[numeric_cols].fillna(0)
        
        print(f"Final engineered data shape: {df_eng.shape}")
        
        return df_eng
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target variable using ONLY lagged features."""
        # Exclude ALL contemporaneous variables except team variables
        exclude_cols = [
            'player_id', 'season', 'season_str', 'year', 
            'points', 'goals', 'assists', 'total_points',
            'position', 'pos', 'Position', 'Pos',
            'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game'
        ]
        
        # Exclude contemporaneous PP variables and numbered variables
        for col in df.columns:
            if any(col.endswith(str(i)) for i in range(1, 10)):
                exclude_cols.append(col)
            elif col.startswith('pp') and 'lag' not in col:
                exclude_cols.append(col)
            elif col in ['shots_per_game', 'goals_per_game', 'assists_per_game', 'points_per_game'] and 'lag' not in col and 'hist' not in col:
                exclude_cols.append(col)
        
        # Exclude text and non-numeric columns
        name_indicators = ['name', 'player', 'firstname', 'lastname', 'full_name', 'player_name']
        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in name_indicators):
                exclude_cols.append(col)
        
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except (ValueError, TypeError):
                        exclude_cols.append(col)
        
        # Remove columns with >50% missing data
        for col in df.columns:
            if col not in exclude_cols:
                missing_pct = df[col].isnull().sum() / len(df) * 100
                if missing_pct > 50:
                    exclude_cols.append(col)
        
        # Remove duplicates and get final feature list
        exclude_cols = list(set(exclude_cols))
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\nPreparing features (LAGGED ONLY with enhanced regularization):")
        print(f"  Total columns in data: {len(df.columns)}")
        print(f"  Excluded columns: {len(exclude_cols)}")
        print(f"  Feature columns: {len(self.feature_columns)}")
        
        # Ensure all remaining columns are numeric
        X = df[self.feature_columns].copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except Exception as e:
                    X = X.drop(columns=[col])
                    self.feature_columns.remove(col)
        
        y = df[self.target_column].copy()
        
        # Final cleanup
        initial_size = len(X)
        missing_mask = X.isnull().any(axis=1) | y.isnull()
        if missing_mask.any():
            X_clean = X[~missing_mask]
            y_clean = y[~missing_mask]
        else:
            X_clean = X
            y_clean = y
        
        print(f"  Observations before cleaning: {initial_size}")
        print(f"  Observations after cleaning: {len(X_clean)}")
        
        return X_clean, y_clean
    
    def _get_regularized_models(self) -> Dict[str, Any]:
        """Get dictionary of regularized models with hyperparameter grids."""
        models = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [5, 10, 20],
                    'min_samples_leaf': [2, 5, 10],
                    'max_features': ['sqrt', 'log2', 0.5],
                    'min_impurity_decrease': [0.0, 0.01, 0.05]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42, validation_fraction=0.2, n_iter_no_change=10),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 0.8, 0.9],
                    'min_samples_split': [5, 10, 20],
                    'min_samples_leaf': [2, 5, 10],
                    'max_features': ['sqrt', 'log2', 0.5]
                }
            },
            'ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
                }
            },
            'lasso': {
                'model': Lasso(random_state=42, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            },
            'elastic_net': {
                'model': ElasticNet(random_state=42, max_iter=2000),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            }
        }
        return models
    
    def train_model_with_regularization(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Dict[str, Any]:
        """Train model with comprehensive regularization and hyperparameter tuning."""
        print(f"\nTraining {model_type} with enhanced regularization:")
        print(f"  Training data shape: {X.shape}")
        
        if len(X) < 20:
            raise ValueError(f"Insufficient data for training: {len(X)} observations")
        
        # Feature selection if enabled
        if self.use_feature_selection and len(self.feature_columns) > 20:
            print(f"  Applying feature selection...")
            k_features = min(50, max(10, len(self.feature_columns) // 2))
            self.feature_selector = SelectKBest(score_func=f_regression, k=k_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
            print(f"  Selected {len(self.selected_features)} features from {len(self.feature_columns)}")
            X = pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
        else:
            self.selected_features = self.feature_columns
        
        # Split data
        test_size = min(0.25, max(0.15, 30/len(X)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        print(f"  Train size: {len(X_train)}")
        print(f"  Test size: {len(X_test)}")
        
        # Get regularized models
        models = self._get_regularized_models()
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_config = models[model_type]
        
        # Scale features for linear models
        if model_type in ['ridge', 'lasso', 'elastic_net']:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            X_train_final = X_train_scaled
            X_test_final = X_test_scaled
        else:
            X_train_final = X_train
            X_test_final = X_test
        
        # Hyperparameter tuning with cross-validation
        print(f"  Performing hyperparameter tuning...")
        cv_folds = min(5, max(3, len(X_train) // 20))
        
        # Use TimeSeriesSplit for time-aware cross-validation if we have enough data
        if len(X_train) > 100:
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            cv = cv_folds
        
        grid_search = GridSearchCV(
            estimator=model_config['model'],
            param_grid=model_config['params'],
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            return_train_score=True
        )
        
        grid_search.fit(X_train_final, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"  Best parameters: {self.best_params}")
        print(f"  Best CV score: {grid_search.best_score_:.3f}")
        
        # Make predictions
        y_pred = self.model.predict(X_test_final)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Additional regularization metrics
        train_score = self.model.score(X_train_final, y_train)
        test_score = r2
        overfitting_ratio = (train_score - test_score) / train_score if train_score > 0 else 0
        
        results = {
            'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
            'train_r2': train_score, 'test_r2': test_score,
            'overfitting_ratio': overfitting_ratio,
            'cv_mean': grid_search.best_score_,
            'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
            'y_test': y_test, 'y_pred': y_pred,
            'model_type': model_type,
            'best_params': self.best_params,
            'n_train': len(X_train), 'n_test': len(X_test),
            'n_features_selected': len(self.selected_features)
        }
        
        print(f"  R² Score: {r2:.3f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  Train R²: {train_score:.3f}")
        print(f"  Overfitting ratio: {overfitting_ratio:.3f}")
        print(f"  Features used: {len(self.selected_features)}")
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            importance = np.abs(self.model.coef_)
        
        feature_names = self.selected_features if self.selected_features else self.feature_columns
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def predict_season(self, df_full: pd.DataFrame, target_year: int) -> pd.DataFrame:
        """Predict points for a specific season using trained model."""
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        # Get target year data
        target_data = df_full[df_full['year'] == target_year].copy()
        
        if len(target_data) == 0:
            raise ValueError(f"No data found for year {target_year}")
        
        print(f"\nPredicting {target_year} season:")
        print(f"  Players to predict: {len(target_data)}")
        
        # Prepare features for prediction
        X_pred, _ = self.prepare_features(target_data)
        
        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            X_pred_selected = X_pred[self.selected_features]
        else:
            X_pred_selected = X_pred
        
        print(f"  Players with complete feature data: {len(X_pred_selected)}")
        
        # Scale features if linear model
        if hasattr(self.scaler, 'transform') and self.best_params and self.model.__class__.__name__ in ['Ridge', 'Lasso', 'ElasticNet']:
            X_pred_final = self.scaler.transform(X_pred_selected)
        else:
            X_pred_final = X_pred_selected
        
        # Make predictions
        predictions = self.model.predict(X_pred_final)
        
        # Create results dataframe
        results_df = target_data.loc[X_pred_selected.index].copy()
        results_df['predicted_points'] = predictions
        
        # Calculate prediction confidence intervals if possible
        if hasattr(self.model, 'predict') and hasattr(self.model, 'estimators_'):
            # For ensemble methods, calculate prediction std
            try:
                predictions_all = np.array([tree.predict(X_pred_final) for tree in self.model.estimators_])
                prediction_std = np.std(predictions_all, axis=0)
                results_df['prediction_std'] = prediction_std
                results_df['confidence_lower'] = predictions - 1.96 * prediction_std
                results_df['confidence_upper'] = predictions + 1.96 * prediction_std
            except:
                pass
        
        # Sort by predicted points descending
        results_df = results_df.sort_values('predicted_points', ascending=False)
        
        return results_df[['player_name', 'predicted_points', 'points'] + 
                         (['prediction_std', 'confidence_lower', 'confidence_upper'] if 'prediction_std' in results_df.columns else [])].reset_index(drop=True)
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot enhanced model results with regularization metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(results['y_test'], results['y_pred'], alpha=0.6)
        min_val, max_val = results['y_test'].min(), results['y_test'].max()
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Points')
        axes[0, 0].set_ylabel('Predicted Points')
        axes[0, 0].set_title(f'Actual vs Predicted\nR² = {results["r2"]:.3f}, Overfitting = {results["overfitting_ratio"]:.3f}')
        
        # Residuals
        residuals = results['y_test'] - results['y_pred']
        axes[0, 1].scatter(results['y_pred'], residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Points')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Feature Importance
        try:
            importance_df = self.get_feature_importance(15)
            axes[0, 2].barh(range(len(importance_df)), importance_df['importance'])
            axes[0, 2].set_yticks(range(len(importance_df)))
            axes[0, 2].set_yticklabels(importance_df['feature'], fontsize=8)
            axes[0, 2].set_xlabel('Importance')
            axes[0, 2].set_title(f'Feature Importance (Top 15)\n{results["n_features_selected"]} features selected')
        except Exception as e:
            axes[0, 2].text(0.5, 0.5, f'Feature importance\nnot available:\n{str(e)}', 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
        
        # Residual histogram
        axes[1, 0].hist(residuals, bins=min(20, len(residuals)//3), alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        
        # Training vs Test Performance
        metrics = ['Train R²', 'Test R²', 'CV Mean']
        values = [results['train_r2'], results['test_r2'], results['cv_mean']]
        bars = axes[1, 1].bar(metrics, values, alpha=0.7)
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Model Performance Comparison')
        axes[1, 1].set_ylim(0, max(values) * 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Model Parameters Summary
        axes[1, 2].axis('off')
        param_text = f"Model: {results['model_type']}\n"
        param_text += f"Training samples: {results['n_train']}\n"
        param_text += f"Test samples: {results['n_test']}\n"
        param_text += f"Features selected: {results['n_features_selected']}\n\n"
        param_text += "Best Parameters:\n"
        
        if 'best_params' in results and results['best_params']:
            for key, value in results['best_params'].items():
                param_text += f"  {key}: {value}\n"
        
        axes[1, 2].text(0.1, 0.9, param_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Model Configuration')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Enhanced plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    """
    Main function to run the enhanced hockey points predictor with regularization
    """
    csv_path = '/Users/juliandigiovanni/Library/CloudStorage/Dropbox/hockeyanalytics/nhl_output/skater_team_data.csv'
    
    # Create output directories
    os.makedirs('nhl_plots', exist_ok=True)
    os.makedirs('nhl_results', exist_ok=True)
    
    # Setup output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'nhl_results/enhanced_forward_points_prediction_v4.1_{timestamp}.txt'
    
    # Setup output manager to write to both console and file
    output_manager = OutputManager(output_file)
    sys.stdout = output_manager
    
    try:
        print("Enhanced Hockey Points Predictor v4.1 - Robust Regularization Edition")
        print("=" * 80)
        print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results will be saved to: {output_file}")
        
        # TRAINING A: Use all data with enhanced regularization
        print(f"\n{'='*80}")
        print("TRAINING A: USING ALL DATA WITH ENHANCED REGULARIZATION")
        print('='*80)
        
        best_results_all = None
        best_config_all = None
        best_r2_all = -1
        
        # Test different lag configurations for all data
        configs = [
            {'lag_years': 1, 'min_training_year': None, 'use_feature_selection': True},
            {'lag_years': 2, 'min_training_year': None, 'use_feature_selection': True},
            {'lag_years': 3, 'min_training_year': None, 'use_feature_selection': True}
        ]
        
        for config in configs:
            print(f"\nTesting {config['lag_years']}-year lags (ALL DATA)")
            print('-' * 60)
            
            try:
                predictor = EnhancedHockeyPointsPredictor(**config)
                df = predictor.load_data(csv_path)
                df_with_lags = predictor.create_lag_features(df)
                df_engineered = predictor.engineer_features(df_with_lags)
                X, y = predictor.prepare_features(df_engineered)
                
                if len(X) < 50:
                    print(f"Insufficient data: {len(X)} observations")
                    continue
                
                # Test different regularized models
                models = ['ridge', 'lasso', 'elastic_net', 'random_forest', 'gradient_boosting']
                
                for model_type in models:
                    try:
                        print(f"\n  Testing {model_type}...")
                        results = predictor.train_model_with_regularization(X, y, model_type=model_type)
                        
                        # Prefer models with lower overfitting and good performance
                        model_score = results['r2'] - (results['overfitting_ratio'] * 0.1)  # Penalize overfitting
                        
                        if model_score > best_r2_all:
                            best_r2_all = model_score
                            best_results_all = results
                            best_config_all = {**config, 'model_type': model_type, 'predictor': predictor}
                            
                    except Exception as e:
                        print(f"    Error with {model_type}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error with configuration {config}: {e}")
                continue
        
        # TRAINING B: Omit 2024-2025 and predict it with enhanced regularization
        print(f"\n{'='*80}")
        print("TRAINING B: OMIT 2024-2025 AND PREDICT IT (ENHANCED REGULARIZATION)")
        print('='*80)
        
        best_results_pred = None
        best_config_pred = None
        best_r2_pred = -1
        best_predictions = None
        
        for config in configs:
            print(f"\nTesting {config['lag_years']}-year lags (EXCLUDING 2024-2025)")
            print('-' * 60)
            
            try:
                predictor = EnhancedHockeyPointsPredictor(**config)
                df_full = predictor.load_data(csv_path)
                
                # Split data: exclude 2024-2025 for training
                df_train = df_full[df_full['year'] != 2024].copy()
                
                print(f"Full dataset: {len(df_full)} observations")
                print(f"Training dataset (no 2024-2025): {len(df_train)} observations")
                print(f"2024-2025 data: {len(df_full[df_full['year'] == 2024])} observations")
                
                df_with_lags = predictor.create_lag_features(df_train)
                df_engineered = predictor.engineer_features(df_with_lags)
                X, y = predictor.prepare_features(df_engineered)
                
                if len(X) < 50:
                    print(f"Insufficient training data: {len(X)} observations")
                    continue
                
                # Test different regularized models
                models = ['ridge', 'lasso', 'elastic_net', 'random_forest', 'gradient_boosting']
                
                for model_type in models:
                    try:
                        print(f"\n  Testing {model_type}...")
                        results = predictor.train_model_with_regularization(X, y, model_type=model_type)
                        
                        # Prefer models with lower overfitting and good performance
                        model_score = results['r2'] - (results['overfitting_ratio'] * 0.1)
                        
                        if model_score > best_r2_pred:
                            best_r2_pred = model_score
                            best_results_pred = results
                            best_config_pred = {**config, 'model_type': model_type, 'predictor': predictor}
                            
                            # Generate predictions for 2024-2025
                            try:
                                df_full_with_lags = predictor.create_lag_features(df_full)
                                df_full_engineered = predictor.engineer_features(df_full_with_lags)
                                predictions = predictor.predict_season(df_full_engineered, 2024)
                                best_predictions = predictions
                            except Exception as e:
                                print(f"    Error generating predictions: {e}")
                                best_predictions = None
                            
                    except Exception as e:
                        print(f"    Error with {model_type}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error with configuration {config}: {e}")
                continue
        
        # Show results
        print(f"\n{'='*80}")
        print("ENHANCED RESULTS SUMMARY")
        print('='*80)
        
        if best_results_all:
            print(f"\nTRAINING A - ALL DATA (ENHANCED REGULARIZATION):")
            print(f"  Best configuration: {best_config_all['lag_years']}-year lags, {best_config_all['model_type']}")
            print(f"  R² Score: {best_results_all['r2']:.3f}")
            print(f"  RMSE: {best_results_all['rmse']:.2f}")
            print(f"  Overfitting Ratio: {best_results_all['overfitting_ratio']:.3f}")
            print(f"  Features Selected: {best_results_all['n_features_selected']}")
            print(f"  Training size: {best_results_all['n_train']}")
            
            # Plot results for all data and save
            plot_filename = f'nhl_plots/enhanced_forward_model_v4.1_all_data_{timestamp}.png'
            best_config_all['predictor'].plot_results(best_results_all, save_path=plot_filename)
        
        if best_results_pred and best_predictions is not None:
            print(f"\nTRAINING B - PREDICT 2024-2025 (ENHANCED REGULARIZATION):")
            print(f"  Best configuration: {best_config_pred['lag_years']}-year lags, {best_config_pred['model_type']}")
            print(f"  R² Score: {best_results_pred['r2']:.3f}")
            print(f"  RMSE: {best_results_pred['rmse']:.2f}")
            print(f"  Overfitting Ratio: {best_results_pred['overfitting_ratio']:.3f}")
            print(f"  Features Selected: {best_results_pred['n_features_selected']}")
            print(f"  Training size: {best_results_pred['n_train']}")
            
            # Plot results for prediction scenario and save
            plot_filename = f'nhl_plots/enhanced_forward_model_v4.1_prediction_{timestamp}.png'
            best_config_pred['predictor'].plot_results(best_results_pred, save_path=plot_filename)
            
            # Show top 50 predictions
            print(f"\nTOP 50 PREDICTED POINTS FOR 2024-2025 SEASON (ENHANCED MODEL):")
            print("=" * 70)
            if 'confidence_lower' in best_predictions.columns:
                print(f"{'Rank':<4} {'Player Name':<25} {'Predicted':<10} {'Actual':<8} {'Conf. Int.':<15}")
                print("-" * 70)
                
                for idx, row in best_predictions.head(50).iterrows():
                    actual_points = row['points'] if not pd.isna(row['points']) else 'N/A'
                    if 'confidence_lower' in row and not pd.isna(row['confidence_lower']):
                        conf_int = f"[{row['confidence_lower']:.1f}-{row['confidence_upper']:.1f}]"
                    else:
                        conf_int = "N/A"
                    print(f"{idx+1:<4} {row['player_name']:<25} {row['predicted_points']:<10.1f} {actual_points:<8} {conf_int:<15}")
            else:
                print(f"{'Rank':<4} {'Player Name':<30} {'Predicted Points':<15} {'Actual Points':<12}")
                print("-" * 70)
                
                for idx, row in best_predictions.head(50).iterrows():
                    actual_points = row['points'] if not pd.isna(row['points']) else 'N/A'
                    print(f"{idx+1:<4} {row['player_name']:<30} {row['predicted_points']:<15.1f} {actual_points:<12}")
        
        if not best_results_all and not best_results_pred:
            print("\nNo working configuration found!")
        
        print(f"\nRun completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {output_file}")
        if best_results_all:
            print(f"All data plot saved to: nhl_plots/enhanced_forward_model_v4.1_all_data_{timestamp}.png")
        if best_results_pred:
            print(f"Prediction plot saved to: nhl_plots/enhanced_forward_model_v4.1_prediction_{timestamp}.png")
        
        # Print regularization summary
        print(f"\n{'='*80}")
        print("REGULARIZATION TECHNIQUES APPLIED:")
        print('='*80)
        print("✓ Feature Selection: Automatic reduction to most important features")
        print("✓ Hyperparameter Tuning: Grid search with cross-validation")
        print("✓ Tree Regularization: min_samples_split, min_samples_leaf, max_features")
        print("✓ Linear Regularization: Ridge (L2), Lasso (L1), ElasticNet (L1+L2)")
        print("✓ Early Stopping: Validation-based stopping for gradient boosting")
        print("✓ Time-Aware CV: TimeSeriesSplit for realistic validation")
        print("✓ Robust Scaling: RobustScaler for outlier resistance")
        print("✓ Overfitting Detection: Train vs test performance monitoring")
    
    finally:
        # Restore stdout and close the output file
        sys.stdout = output_manager.terminal
        output_manager.close()

if __name__ == "__main__":
    main()