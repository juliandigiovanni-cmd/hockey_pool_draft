# Date: August 8, 2025
# Author: Julian di Giovanni w/Claude.ai
# Version: 4.0

"""
PROGRAM OVERVIEW: NHL Forward Points Prediction Using Historical Performance
============================================================================

This program builds machine learning models to predict NHL forward scoring using only 
historical (lagged) performance data. It implements a rigorous approach to avoid data 
leakage by excluding all contemporaneous statistics.

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

3. FEATURE ENGINEERING:
   - Creates advanced metrics from lagged data:
     * Historical averages (e.g., goals_hist_avg, assists_hist_avg)
     * Trend indicators (lag1 vs lag2 performance changes)
     * Per-game rates for all lagged seasons (goals_per_game_lag1, etc.)
     * Historical points totals and per-game averages
   - All engineered features use ONLY historical data (no current season info)

4. FEATURE PREPARATION:
   - EXCLUDES all contemporaneous variables to prevent data leakage:
     * Current season: goals, assists, shots, games_played, etc.
     * Current season power play stats
     * Any variables ending in numbers (1, 2, 3, etc.)
   - INCLUDES only historical and team variables:
     * All lagged statistics (goals_lag1, assists_lag2, etc.)
     * Team variables (team_1, team_2, etc.)
     * Engineered historical features
   - Displays complete list of features used in training
   - Handles missing values and ensures all features are numeric

5. TWO TRAINING SCENARIOS:

   TRAINING A - ALL DATA:
   - Uses complete dataset for training and validation
   - Tests multiple lag configurations (1, 2, 3 years)
   - Tests multiple models (Random Forest, Gradient Boosting, Linear Regression)
   - Selects best performing combination based on RÂ² score
   - Provides model evaluation plots and feature importance

   TRAINING B - PREDICTIVE VALIDATION:
   - Excludes 2024-2025 season from training data
   - Trains on historical data only (up to 2023-2024)
   - Uses trained model to predict 2024-2025 season performance
   - Validates predictions against actual 2024-2025 results
   - Tests realistic prediction scenario

6. MODEL TRAINING & EVALUATION:
   - Implements train/test split with appropriate validation
   - Uses cross-validation for robust performance estimation
   - Calculates multiple metrics: RÂ², RMSE, MAE
   - Handles feature scaling for linear models
   - Prevents overfitting with appropriate model parameters

7. PREDICTIONS & OUTPUT:
   - Generates point predictions for 2024-2025 season
   - Ranks all forwards by predicted scoring
   - Creates formatted table of top 50 predicted scorers
   - Compares predictions to actual results where available
   - Provides comprehensive model performance summary

KEY FEATURES:
============
- NO DATA LEAKAGE: Uses only historical data for predictions
- PRESERVES ELITE PERFORMANCE: No artificial capping of high-scoring seasons
- ROBUST VALIDATION: Tests on held-out future season
- COMPREHENSIVE: Includes basic stats, power play, and engineered features  
- FLEXIBLE: Tests multiple lag periods and model types
- INTERPRETABLE: Shows feature importance and model diagnostics
- PRACTICAL: Outputs actionable player rankings

TECHNICAL APPROACH:
==================
- Handles missing data intelligently with domain knowledge
- Preserves all legitimate elite performance data (no artificial capping)
- Creates meaningful lag features for time series nature of hockey data
- Uses ensemble methods and linear models for comparison
- Implements proper train/test splits to avoid overfitting
- Scales features appropriately for different model types
- Provides extensive logging for transparency and debugging

OUTPUT:
=======
1. Data loading and cleaning summary
2. Feature engineering details with complete feature list
3. Training results for both scenarios (A & B)
4. Model performance metrics and diagnostic plots
5. Top 50 predicted scorers for 2024-2025 season with actual results
6. Feature importance rankings for best performing model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
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
            # Create directory if it doesn't exist
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

class HockeyPointsPredictor:
    def __init__(self, lag_years: int = 2, min_training_year: int = None):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.target_column = 'points'
        self.lag_years = lag_years
        self.min_training_year = min_training_year  # Only use data from this year onwards
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load hockey player statistics from CSV file and remove defensemen.
        """
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
            # Remove defensemen (D, LD, RD, or any position containing 'D')
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
        """
        Clean and handle missing data issues.
        """
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
            # Create shooting percentage from goals/shots
            if 'goals' in df.columns and 'shots' in df.columns:
                df['shooting_pct'] = df['goals'] / df['shots'].replace(0, np.nan)
                df['shooting_pct'] = df['shooting_pct'].fillna(0.1)  # Fill with reasonable default
            else:
                df['shooting_pct'] = 0.1  # Default for forwards
        
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
            df['time_on_ice_per_game'] = 15.0  # Reasonable default for forwards
        
        # Clean up core statistical columns
        core_stats = ['goals', 'assists', 'shots', 'games_played', 'points']
        for stat in core_stats:
            if stat in df.columns:
                # Fill missing values with 0 (reasonable for counting stats)
                df[stat] = df[stat].fillna(0)
                # Ensure non-negative values
                df[stat] = df[stat].clip(lower=0)
        
        # Remove rows where essential data is completely missing
        essential_cols = ['player_id', 'year', 'games_played']
        before_essential = len(df)
        df = df.dropna(subset=essential_cols)
        after_essential = len(df)
        
        if before_essential != after_essential:
            print(f"Removed {before_essential - after_essential} rows missing essential data")
        
        # Remove players with 0 games played (likely data errors)
        if 'games_played' in df.columns:
            before_games = len(df)
            df = df[df['games_played'] > 0]
            after_games = len(df)
            if before_games != after_games:
                print(f"Removed {before_games - after_games} rows with 0 games played")
        
        print(f"Data shape after cleaning: {df.shape}")
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features for previous seasons.
        Only creates lags where historical data exists.
        """
        # Include PP variables for lagging but exclude numbered variables
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
        """
        Create additional engineered features from lag data.
        """
        df_eng = df.copy()
        
        print(f"\nEngineering features...")
        
        # Historical features from lag data ONLY
        lag_columns = [col for col in df_eng.columns if 'lag' in col]
        
        if lag_columns:
            print(f"Found {len(lag_columns)} lag columns")
            
            # Get all base features that have lags (including PP variables)
            lag_base_features = ['goals', 'assists', 'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game']
            # Add PP base features
            pp_base_features = list(set([col.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '') 
                                        for col in lag_columns if col.startswith('pp')]))
            lag_base_features.extend(pp_base_features)
            
            for base_feature in lag_base_features:
                lag_cols = [col for col in lag_columns if col.startswith(f'{base_feature}_lag')]
                
                if lag_cols:
                    # Average over available lags (handle missing values)
                    df_eng[f'{base_feature}_hist_avg'] = df_eng[lag_cols].mean(axis=1, skipna=True)
                    
                    # Recent trend (lag1 vs lag2) - only if we have both
                    if len(lag_cols) >= 2 and f'{base_feature}_lag1' in df_eng.columns and f'{base_feature}_lag2' in df_eng.columns:
                        trend = df_eng[f'{base_feature}_lag1'] - df_eng[f'{base_feature}_lag2']
                        df_eng[f'{base_feature}_trend'] = trend.fillna(0)
            
            # Historical per-game rates
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
            
            # Historical per-game averages
            per_game_features = ['goals_per_game', 'assists_per_game', 'shots_per_game', 'points_per_game']
            for feature in per_game_features:
                feature_lag_cols = [col for col in df_eng.columns if col.startswith(f'{feature}_lag')]
                if feature_lag_cols:
                    hist_avg = df_eng[feature_lag_cols].mean(axis=1, skipna=True)
                    df_eng[f'{feature}_hist_avg'] = hist_avg.fillna(0)
        
        # Clean up infinite and extreme values
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN values with appropriate defaults
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        df_eng[numeric_cols] = df_eng[numeric_cols].fillna(0)
        
        print(f"Final engineered data shape: {df_eng.shape}")
        
        return df_eng
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable using ONLY lagged features.
        Excludes all contemporaneous variables except team variables.
        """
        # Exclude ALL contemporaneous variables except team variables
        exclude_cols = [
            'player_id', 'season', 'season_str', 'year', 
            'points', 'goals', 'assists', 'total_points',
            'position', 'pos', 'Position', 'Pos',
            # Exclude ALL current season stats (contemporaneous)
            'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game'
        ]
        
        # Exclude contemporaneous PP variables and numbered variables
        for col in df.columns:
            # Exclude any column ending with just numbers (like variable1, variable2, etc.)
            if any(col.endswith(str(i)) for i in range(1, 10)):
                exclude_cols.append(col)
                print(f"Excluding numbered column: '{col}'")
            # Exclude contemporaneous power play variables (pp* but not pp*_lag*)
            elif col.startswith('pp') and 'lag' not in col:
                exclude_cols.append(col)
                print(f"Excluding contemporaneous power play column: '{col}'")
            # Exclude any other contemporaneous stats that might exist
            elif col in ['shots_per_game', 'goals_per_game', 'assists_per_game', 'points_per_game'] and 'lag' not in col and 'hist' not in col:
                exclude_cols.append(col)
                print(f"Excluding contemporaneous stat: '{col}'")
        
        # Exclude any columns that might contain player names or other text
        name_indicators = ['name', 'player', 'firstname', 'lastname', 'full_name', 'player_name']
        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in name_indicators):
                exclude_cols.append(col)
                print(f"Excluding text column: '{col}'")
        
        # Exclude any non-numeric columns
        for col in df.columns:
            if col not in exclude_cols:
                # Check if column contains non-numeric data
                if df[col].dtype == 'object' or df[col].dtype.name == 'string':
                    # Try to see if it's actually numeric stored as string
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except (ValueError, TypeError):
                        exclude_cols.append(col)
                        print(f"Excluding non-numeric column: '{col}' (dtype: {df[col].dtype})")
        
        # Remove any columns that are mostly missing
        for col in df.columns:
            if col not in exclude_cols:
                missing_pct = df[col].isnull().sum() / len(df) * 100
                if missing_pct > 50:  # Exclude columns with >50% missing data
                    exclude_cols.append(col)
                    print(f"Excluding '{col}' due to {missing_pct:.1f}% missing data")
        
        # Remove duplicates and get final feature list
        exclude_cols = list(set(exclude_cols))
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\nPreparing features (LAGGED ONLY):")
        print(f"  Total columns in data: {len(df.columns)}")
        print(f"  Excluded columns: {len(exclude_cols)}")
        print(f"  Feature columns: {len(self.feature_columns)}")
        
        # Show which features are being used
        print(f"\nFEATURES BEING USED IN TRAINING:")
        print("-" * 50)
        for i, feature in enumerate(self.feature_columns, 1):
            print(f"  {i:2d}. {feature}")
        
        if len(exclude_cols) < 30:  # Only show if not too many
            print(f"\nExcluded columns: {exclude_cols}")
        
        # Ensure all remaining columns are numeric
        X = df[self.feature_columns].copy()
        
        # Convert any remaining object columns to numeric (this will catch edge cases)
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    print(f"Converted '{col}' to numeric")
                except Exception as e:
                    print(f"Could not convert '{col}' to numeric: {e}")
                    # Remove this column if it can't be converted
                    X = X.drop(columns=[col])
                    self.feature_columns.remove(col)
        
        y = df[self.target_column].copy()
        
        # Final cleanup - remove any remaining missing values
        initial_size = len(X)
        
        # Check for any remaining missing values
        missing_mask = X.isnull().any(axis=1) | y.isnull()
        if missing_mask.any():
            print(f"  Removing {missing_mask.sum()} rows with missing values")
            X_clean = X[~missing_mask]
            y_clean = y[~missing_mask]
        else:
            X_clean = X
            y_clean = y
        
        print(f"  Observations before cleaning: {initial_size}")
        print(f"  Observations after cleaning: {len(X_clean)}")
        
        if initial_size > 0:
            data_loss_pct = (initial_size - len(X_clean))/initial_size*100
            print(f"  Data loss: {initial_size - len(X_clean)} ({data_loss_pct:.1f}%)")
        
        return X_clean, y_clean
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train the machine learning model.
        """
        print(f"\nTraining {model_type} model:")
        print(f"  Training data shape: {X.shape}")
        
        if len(X) < 20:
            raise ValueError(f"Insufficient data for training: {len(X)} observations")
        
        # Split data
        test_size = min(0.25, max(0.15, 30/len(X)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        print(f"  Train size: {len(X_train)}")
        print(f"  Test size: {len(X_test)}")
        
        # Scale features for linear regression
        if model_type == 'linear_regression':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        if model_type == 'random_forest':
            n_estimators = min(100, max(20, len(X_train) // 10))
            self.model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=10,
                random_state=42, 
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            n_estimators = min(100, max(20, len(X_train) // 10))
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        
        # Train model
        if model_type == 'linear_regression':
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
        else:
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_folds = min(5, max(3, len(X_train) // 20))
        try:
            if model_type == 'linear_regression':
                cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
            else:
                cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='r2')
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        except:
            cv_mean, cv_std = r2, 0  # Fallback if CV fails
        
        results = {
            'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
            'cv_mean': cv_mean, 'cv_std': cv_std,
            'y_test': y_test, 'y_pred': y_pred,
            'model_type': model_type,
            'n_train': len(X_train), 'n_test': len(X_test)
        }
        
        print(f"  RÂ² Score: {r2:.3f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            importance = np.abs(self.model.coef_)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def predict_season(self, df_full: pd.DataFrame, target_year: int) -> pd.DataFrame:
        """
        Predict points for a specific season using trained model.
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
            
        # Get target year data
        target_data = df_full[df_full['year'] == target_year].copy()
        
        if len(target_data) == 0:
            raise ValueError(f"No data found for year {target_year}")
        
        print(f"\nPredicting {target_year} season:")
        print(f"  Players to predict: {len(target_data)}")
        
        # Prepare features for prediction (same process as training)
        X_pred, _ = self.prepare_features(target_data)
        
        print(f"  Players with complete feature data: {len(X_pred)}")
        
        # Make predictions
        if hasattr(self.scaler, 'transform') and hasattr(self.model, 'predict'):
            try:
                if hasattr(self.model, 'feature_importances_'):  # Tree-based models
                    predictions = self.model.predict(X_pred)
                else:  # Linear models
                    X_pred_scaled = self.scaler.transform(X_pred)
                    predictions = self.model.predict(X_pred_scaled)
            except:
                predictions = self.model.predict(X_pred)
        else:
            predictions = self.model.predict(X_pred)
        
        # Create results dataframe
        results_df = target_data.loc[X_pred.index].copy()
        results_df['predicted_points'] = predictions
        
        # Sort by predicted points descending
        results_df = results_df.sort_values('predicted_points', ascending=False)
        
        return results_df[['player_name', 'predicted_points', 'points']].reset_index(drop=True)
    
    def plot_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot model results and optionally save to file."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(results['y_test'], results['y_pred'], alpha=0.6)
        min_val, max_val = results['y_test'].min(), results['y_test'].max()
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Points')
        axes[0, 0].set_ylabel('Predicted Points')
        axes[0, 0].set_title(f'Actual vs Predicted (Forwards Only, Lagged Features)\nRÂ² = {results["r2"]:.3f}')
        
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
            axes[1, 0].barh(range(len(importance_df)), importance_df['importance'])
            axes[1, 0].set_yticks(range(len(importance_df)))
            axes[1, 0].set_yticklabels(importance_df['feature'], fontsize=8)
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Feature Importance (Top 15)')
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Feature importance\nnot available:\n{str(e)}', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Residual histogram
        axes[1, 1].hist(residuals, bins=min(20, len(residuals)//3), alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residual Distribution')
        
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    """
    Main function to run the hockey points predictor with two training scenarios
    """
    csv_path = '/Users/juliandigiovanni/Library/CloudStorage/Dropbox/hockeyanalytics/nhl_output/skater_team_data.csv'
    
    # Create output directories
    os.makedirs('nhl_plots', exist_ok=True)
    os.makedirs('nhl_results', exist_ok=True)
    
    # Setup output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'nhl_results/forward_points_prediction_v4_{timestamp}.txt'
    
    # Setup output manager to write to both console and file
    output_manager = OutputManager(output_file)
    sys.stdout = output_manager
    
    try:
        print("Hockey Points Predictor v4 - Lagged Features Only (Forwards)")
        print("=" * 70)
        print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results will be saved to: {output_file}")
        
        # TRAINING A: Use all data
        print(f"\n{'='*70}")
        print("TRAINING A: USING ALL DATA")
        print('='*70)
        
        best_results_all = None
        best_config_all = None
        best_r2_all = -1
        
        # Test different lag configurations for all data
        configs = [
            {'lag_years': 1, 'min_training_year': None},
            {'lag_years': 2, 'min_training_year': None},
            {'lag_years': 3, 'min_training_year': None}
        ]
        
        for config in configs:
            print(f"\nTesting {config['lag_years']}-year lags (ALL DATA)")
            print('-' * 50)
            
            try:
                predictor = HockeyPointsPredictor(**config)
                df = predictor.load_data(csv_path)
                df_with_lags = predictor.create_lag_features(df)
                df_engineered = predictor.engineer_features(df_with_lags)
                X, y = predictor.prepare_features(df_engineered)
                
                if len(X) < 50:
                    print(f"Insufficient data: {len(X)} observations")
                    continue
                
                # Test different models
                models = ['random_forest', 'gradient_boosting', 'linear_regression']
                
                for model_type in models:
                    try:
                        results = predictor.train_model(X, y, model_type=model_type)
                        
                        if results['r2'] > best_r2_all:
                            best_r2_all = results['r2']
                            best_results_all = results
                            best_config_all = {**config, 'model_type': model_type, 'predictor': predictor}
                            
                    except Exception as e:
                        print(f"Error with {model_type}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error with configuration {config}: {e}")
                continue
        
        # TRAINING B: Omit 2024-2025 and predict it
        print(f"\n{'='*70}")
        print("TRAINING B: OMIT 2024-2025 AND PREDICT IT")
        print('='*70)
        
        best_results_pred = None
        best_config_pred = None
        best_r2_pred = -1
        best_predictions = None
        
        for config in configs:
            print(f"\nTesting {config['lag_years']}-year lags (EXCLUDING 2024-2025)")
            print('-' * 50)
            
            try:
                predictor = HockeyPointsPredictor(**config)
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
                
                # Test different models
                models = ['random_forest', 'gradient_boosting', 'linear_regression']
                
                for model_type in models:
                    try:
                        results = predictor.train_model(X, y, model_type=model_type)
                        
                        if results['r2'] > best_r2_pred:
                            best_r2_pred = results['r2']
                            best_results_pred = results
                            best_config_pred = {**config, 'model_type': model_type, 'predictor': predictor}
                            
                            # Generate predictions for 2024-2025
                            try:
                                df_full_with_lags = predictor.create_lag_features(df_full)
                                df_full_engineered = predictor.engineer_features(df_full_with_lags)
                                predictions = predictor.predict_season(df_full_engineered, 2024)
                                best_predictions = predictions
                            except Exception as e:
                                print(f"Error generating predictions: {e}")
                                best_predictions = None
                            
                    except Exception as e:
                        print(f"Error with {model_type}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error with configuration {config}: {e}")
                continue
        
        # Show results
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print('='*70)
        
        if best_results_all:
            print(f"\nTRAINING A - ALL DATA:")
            print(f"  Best configuration: {best_config_all['lag_years']}-year lags, {best_config_all['model_type']}")
            print(f"  RÂ² Score: {best_results_all['r2']:.3f}")
            print(f"  RMSE: {best_results_all['rmse']:.2f}")
            print(f"  Training size: {best_results_all['n_train']}")
            
            # Plot results for all data and save
            plot_filename = f'nhl_plots/forward_model_v4_all_data_{timestamp}.png'
            best_config_all['predictor'].plot_results(best_results_all, save_path=plot_filename)
        
        if best_results_pred and best_predictions is not None:
            print(f"\nTRAINING B - PREDICT 2024-2025:")
            print(f"  Best configuration: {best_config_pred['lag_years']}-year lags, {best_config_pred['model_type']}")
            print(f"  RÂ² Score: {best_results_pred['r2']:.3f}")
            print(f"  RMSE: {best_results_pred['rmse']:.2f}")
            print(f"  Training size: {best_results_pred['n_train']}")
            
            # Plot results for prediction scenario and save
            plot_filename = f'nhl_plots/forward_model_v4_prediction_{timestamp}.png'
            best_config_pred['predictor'].plot_results(best_results_pred, save_path=plot_filename)
            
            # Show top 50 predictions
            print(f"\nTOP 50 PREDICTED POINTS FOR 2024-2025 SEASON:")
            print("=" * 60)
            print(f"{'Rank':<4} {'Player Name':<30} {'Predicted Points':<15} {'Actual Points':<12}")
            print("-" * 60)
            
            for idx, row in best_predictions.head(50).iterrows():
                actual_points = row['points'] if not pd.isna(row['points']) else 'N/A'
                print(f"{idx+1:<4} {row['player_name']:<30} {row['predicted_points']:<15.1f} {actual_points:<12}")
        
        if not best_results_all and not best_results_pred:
            print("\nNo working configuration found!")
        
        print(f"\nRun completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {output_file}")
        if best_results_all:
            print(f"All data plot saved to: nhl_plots/forward_model_v4_all_data_{timestamp}.png")
        if best_results_pred:
            print(f"Prediction plot saved to: nhl_plots/forward_model_v4_prediction_{timestamp}.png")
    
    finally:
        # Restore stdout and close the output file
        sys.stdout = output_manager.terminal
        output_manager.close()

if __name__ == "__main__":
    main()