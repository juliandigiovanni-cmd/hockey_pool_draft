# Date: August 8, 2025
# Author: Julian di Giovanni w/Claude.ai.
# Version: 5.0

"""
PROGRAM OVERVIEW: NHL Forward Points Prediction Using Historical Performance + Team Context
===========================================================================================

This program builds machine learning models to predict NHL forward scoring using 
historical (lagged) performance data including both individual stats and team context.
It implements a rigorous approach to avoid data leakage by excluding all 
contemporaneous statistics while leveraging historical team performance.

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
     * Individual stats: goals, assists, shots, games_played, shooting_pct, time_on_ice_per_game
     * Power play stats: pp_goals, pp_assists, pp_shots, etc.
     * Team statistics: variables ending with 1, 2, 3, etc. (team performance metrics)
   - Only creates lags where sufficient historical data exists
   - Filters data to years where complete lag history is available
   - Tracks players with sufficient history for reliable predictions

3. FEATURE ENGINEERING:
   - Creates advanced metrics from lagged data:
     * Historical averages (e.g., goals_hist_avg, assists_hist_avg, team_goals1_hist_avg)
     * Trend indicators (lag1 vs lag2 performance changes)
     * Per-game rates for all lagged seasons (goals_per_game_lag1, etc.)
     * Historical points totals and per-game averages
     * Team context metrics from lagged team statistics
   - All engineered features use ONLY historical data (no current season info)

4. FEATURE PREPARATION:
   - EXCLUDES contemporaneous variables to prevent data leakage:
     * Current season individual: goals, assists, shots, games_played, etc.
     * Current season power play stats (pp* without lag)
     * Current season team stats (variables ending in 1,2,3 without lag)
   - INCLUDES historical and team context:
     * All lagged individual statistics (goals_lag1, assists_lag2, etc.)
     * All lagged power play statistics (pp_goals_lag1, etc.)
     * All lagged team statistics (team_goals1_lag1, team_assists2_lag1, etc.)
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
   - Tests realistic prediction scenario with team context

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
- TEAM CONTEXT: Includes historical team performance as predictive features
- ROBUST VALIDATION: Tests on held-out future season
- COMPREHENSIVE: Includes individual stats, power play, and team statistics
- FLEXIBLE: Tests multiple lag periods and model types
- INTERPRETABLE: Shows feature importance and model diagnostics
- PRACTICAL: Outputs actionable player rankings

TECHNICAL APPROACH:
==================
- Handles missing data intelligently with domain knowledge
- Preserves all legitimate elite performance data (no artificial capping)
- Creates meaningful lag features for time series nature of hockey data
- Incorporates team context while avoiding contemporaneous data leakage
- Uses ensemble methods and linear models for comparison
- Implements proper train/test splits to avoid overfitting
- Scales features appropriately for different model types
- Provides extensive logging for transparency and debugging

OUTPUT:
=======
1. Data loading and cleaning summary
2. Feature engineering details with complete feature list (including team stats)
3. Training results for both scenarios (A & B)
4. Model performance metrics and diagnostic plots
5. Top 50 predicted scorers for 2024-2025 season with actual results
6. Feature importance rankings showing individual vs team stat contributions
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
        Create lagged features for previous seasons including selective team statistics.
        Only creates lags where sufficient data exists.
        """
        # Include individual stats and PP variables for lagging
        lag_features = ['goals', 'assists', 'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game']
        
        # Add PP variables to lag features (exclude numbered ones from PP)
        pp_features = [col for col in df.columns if col.startswith('pp') and not any(col.endswith(str(i)) for i in range(1, 10))]
        lag_features.extend(pp_features)
        
        # Add SELECTIVE team statistics (only most important ones to avoid feature explosion)
        team_features = [col for col in df.columns if any(col.endswith(str(i)) for i in range(1, 10))]
        # Filter to likely important team stats (goals, assists, points, shots)
        important_team_features = []
        for tf in team_features:
            tf_lower = tf.lower()
            if any(keyword in tf_lower for keyword in ['goal', 'assist', 'point', 'shot']):
                important_team_features.append(tf)
        
        # Limit to first 10 team features to prevent feature explosion
        selected_team_features = important_team_features[:10]
        lag_features.extend(selected_team_features)
        
        existing_features = [f for f in lag_features if f in df.columns]
        
        print(f"\nCreating {self.lag_years}-year lags for {len(existing_features)} features:")
        individual_features = [f for f in existing_features if not f.startswith('pp') and not any(f.endswith(str(i)) for i in range(1, 10))]
        pp_lag_features = [f for f in existing_features if f.startswith('pp')]
        team_lag_features = [f for f in existing_features if any(f.endswith(str(i)) for i in range(1, 10))]
        
        print(f"  Individual stats ({len(individual_features)}): {individual_features}")
        print(f"  Power play stats ({len(pp_lag_features)}): {pp_lag_features}")
        print(f"  Selected team statistics ({len(team_lag_features)}): {team_lag_features}")
        
        df_with_lags = df.copy()
        
        # Initialize lag columns
        lag_columns = []
        for feature in existing_features:
            for lag in range(1, self.lag_years + 1):
                lag_col = f'{feature}_lag{lag}'
                lag_columns.append(lag_col)
                df_with_lags[lag_col] = np.nan
        
        print(f"  Total lag columns created: {len(lag_columns)}")
        
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
        print(f"Training data shape after year filter: {training_data.shape}")
        
        # More flexible approach to missing data - don't require ALL lags to be complete
        individual_lag_cols = [col for col in lag_columns if not col.startswith('pp') and not any(col.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '').endswith(str(i)) for i in range(1, 10))]
        
        # Require only individual stats to be complete (more flexible)
        if individual_lag_cols:
            complete_lag_mask = ~training_data[individual_lag_cols].isnull().any(axis=1)
            complete_observations = complete_lag_mask.sum()
            
            print(f"Complete observations with individual lags: {complete_observations}")
            
            if complete_observations > 0:
                complete_years = training_data[complete_lag_mask]['year'].value_counts().sort_index()
                print(f"Complete observations by year:")
                for year, count in complete_years.items():
                    print(f"  {year}: {count}")
        else:
            print("No individual lag columns found")
        
        return training_data
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features from lag data including team context.
        """
        df_eng = df.copy()
        
        print(f"\nEngineering features...")
        
        # Historical features from lag data ONLY
        lag_columns = [col for col in df_eng.columns if 'lag' in col]
        
        if lag_columns:
            print(f"Found {len(lag_columns)} lag columns")
            
            # Get all base features that have lags (individual, PP, and team variables)
            lag_base_features = ['goals', 'assists', 'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game']
            
            # Add PP base features
            pp_base_features = list(set([col.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '') 
                                        for col in lag_columns if col.startswith('pp')]))
            lag_base_features.extend(pp_base_features)
            
            # Add team base features
            team_base_features = list(set([col.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '') 
                                          for col in lag_columns if any(col.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '').endswith(str(i)) for i in range(1, 10))]))
            lag_base_features.extend(team_base_features)
            
            print(f"Creating historical features for {len(lag_base_features)} base features")
            
            for base_feature in lag_base_features:
                lag_cols = [col for col in lag_columns if col.startswith(f'{base_feature}_lag')]
                
                if lag_cols:
                    # Average over available lags (handle missing values)
                    df_eng[f'{base_feature}_hist_avg'] = df_eng[lag_cols].mean(axis=1, skipna=True)
                    
                    # Recent trend (lag1 vs lag2) - only if we have both
                    if len(lag_cols) >= 2 and f'{base_feature}_lag1' in df_eng.columns and f'{base_feature}_lag2' in df_eng.columns:
                        trend = df_eng[f'{base_feature}_lag1'] - df_eng[f'{base_feature}_lag2']
                        df_eng[f'{base_feature}_trend'] = trend.fillna(0)
            
            # Historical per-game rates (for non-team features)
            individual_base_features = [f for f in lag_base_features if not any(f.endswith(str(i)) for i in range(1, 10))]
            
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
        Excludes all contemporaneous variables but includes lagged team statistics.
        """
        print(f"\nStarting feature preparation with {len(df)} observations and {len(df.columns)} columns")
        
        # Exclude ALL contemporaneous variables 
        exclude_cols = [
            'player_id', 'season', 'season_str', 'year', 
            'points', 'goals', 'assists', 'total_points',
            'position', 'pos', 'Position', 'Pos',
            # Exclude ALL current season stats (contemporaneous)
            'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game'
        ]
        
        # Exclude contemporaneous PP variables and contemporaneous team statistics
        for col in df.columns:
            # Exclude contemporaneous team statistics (ending with numbers but no lag)
            if any(col.endswith(str(i)) for i in range(1, 10)) and 'lag' not in col:
                exclude_cols.append(col)
            # Exclude contemporaneous power play variables (pp* but not pp*_lag*)
            elif col.startswith('pp') and 'lag' not in col:
                exclude_cols.append(col)
            # Exclude any other contemporaneous stats that might exist
            elif col in ['shots_per_game', 'goals_per_game', 'assists_per_game', 'points_per_game'] and 'lag' not in col and 'hist' not in col:
                exclude_cols.append(col)
        
        # Exclude any columns that might contain player names or other text
        name_indicators = ['name', 'player', 'firstname', 'lastname', 'full_name', 'player_name']
        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in name_indicators):
                exclude_cols.append(col)
        
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
        
        # Remove duplicates and get initial feature list
        exclude_cols = list(set(exclude_cols))
        potential_features = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Potential features after exclusions: {len(potential_features)}")
        
        # Check for features with too much missing data and remove them
        good_features = []
        for col in potential_features:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct <= 80:  # More lenient threshold
                good_features.append(col)
            else:
                print(f"Excluding '{col}' due to {missing_pct:.1f}% missing data")
        
        self.feature_columns = good_features
        
        print(f"Final feature count: {len(self.feature_columns)}")
        
        # Categorize and show which features are being used
        individual_features = [f for f in self.feature_columns if 'lag' in f and not f.startswith('pp') and not any(f.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '').endswith(str(i)) for i in range(1, 10))]
        pp_features = [f for f in self.feature_columns if 'lag' in f and f.startswith('pp')]
        team_features = [f for f in self.feature_columns if 'lag' in f and any(f.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '').endswith(str(i)) for i in range(1, 10))]
        engineered_features = [f for f in self.feature_columns if 'hist' in f or 'trend' in f]
        other_features = [f for f in self.feature_columns if f not in individual_features + pp_features + team_features + engineered_features]
        
        print(f"\nFEATURE BREAKDOWN:")
        print("-" * 50)
        print(f"Individual lagged features: {len(individual_features)}")
        print(f"Power play lagged features: {len(pp_features)}")
        print(f"Team lagged features: {len(team_features)}")
        print(f"Engineered features: {len(engineered_features)}")
        print(f"Other features: {len(other_features)}")
        
        # Show first few from each category
        if individual_features:
            print(f"\nSample individual features: {individual_features[:5]}")
        if team_features:
            print(f"Sample team features: {team_features[:5]}")
        if engineered_features:
            print(f"Sample engineered features: {engineered_features[:5]}")
        
        # Get feature matrix and target
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        print(f"\nBefore missing data removal: {len(X)} observations")
        
        # More flexible missing data handling
        # Instead of requiring ALL features to be non-missing, require at least some core features
        core_individual_features = [f for f in individual_features if any(core in f for core in ['goals_lag', 'assists_lag', 'games_played_lag'])]
        
        if core_individual_features:
            # Require at least the core individual features to be non-missing
            missing_mask = X[core_individual_features].isnull().all(axis=1) | y.isnull()
        else:
            # Fallback to any feature being non-missing
            missing_mask = X.isnull().all(axis=1) | y.isnull()
        
        # Remove rows with missing core data
        if missing_mask.any():
            X_clean = X[~missing_mask]
            y_clean = y[~missing_mask]
            print(f"Removed {missing_mask.sum()} rows with missing core data")
        else:
            X_clean = X
            y_clean = y
        
        # Fill remaining missing values with 0 (for non-core features)
        X_clean = X_clean.fillna(0)
        
        print(f"After missing data handling: {len(X_clean)} observations")
        print(f"Final feature matrix shape: {X_clean.shape}")
        
        # Check for reasonable feature to observation ratio
        if len(X_clean) > 0:
            feature_to_obs_ratio = len(self.feature_columns) / len(X_clean)
            print(f"Feature to observation ratio: {feature_to_obs_ratio:.2f}")
            if feature_to_obs_ratio > 0.5:
                print("WARNING: High feature to observation ratio - consider reducing features")
        
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
        axes[0, 0].set_title(f'Actual vs Predicted (Forwards + Team Context)\nRÂ² = {results["r2"]:.3f}')
        
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
    output_file = f'nhl_results/forward_points_prediction_v5_{timestamp}.txt'
    
    # Setup output manager to write to both console and file
    output_manager = OutputManager(output_file)
    sys.stdout = output_manager
    
    try:
        print("Hockey Points Predictor v5 - With Team Statistics (Forwards)")
        print("=" * 75)
        print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results will be saved to: {output_file}")
        
        # TRAINING A: Use all data
        print(f"\n{'='*75}")
        print("TRAINING A: USING ALL DATA (INCLUDING TEAM CONTEXT)")
        print('='*75)
        
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
                
                print("Loading data...")
                df = predictor.load_data(csv_path)
                
                print("Creating lag features...")
                df_with_lags = predictor.create_lag_features(df)
                
                print("Engineering features...")
                df_engineered = predictor.engineer_features(df_with_lags)
                
                print("Preparing features...")
                X, y = predictor.prepare_features(df_engineered)
                
                print(f"Final dataset: {len(X)} observations, {X.shape[1] if len(X) > 0 else 0} features")
                
                if len(X) < 50:
                    print(f"Insufficient data: {len(X)} observations (need at least 50)")
                    continue
                
                # Test different models
                models = ['random_forest', 'gradient_boosting', 'linear_regression']
                
                for model_type in models:
                    try:
                        print(f"\nTrying {model_type}...")
                        results = predictor.train_model(X, y, model_type=model_type)
                        
                        print(f"Model {model_type} succeeded with RÂ² = {results['r2']:.3f}")
                        
                        if results['r2'] > best_r2_all:
                            best_r2_all = results['r2']
                            best_results_all = results
                            best_config_all = {**config, 'model_type': model_type, 'predictor': predictor}
                            print(f"New best model: {model_type} with RÂ² = {results['r2']:.3f}")
                            
                    except Exception as e:
                        print(f"Error with {model_type}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                        
            except Exception as e:
                print(f"Error with configuration {config}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # TRAINING B: Omit 2024-2025 and predict it
        print(f"\n{'='*75}")
        print("TRAINING B: OMIT 2024-2025 AND PREDICT IT (WITH TEAM CONTEXT)")
        print('='*75)
        
        best_results_pred = None
        best_config_pred = None
        best_r2_pred = -1
        best_predictions = None
        
        for config in configs:
            print(f"\nTesting {config['lag_years']}-year lags (EXCLUDING 2024-2025)")
            print('-' * 50)
            
            try:
                predictor = HockeyPointsPredictor(**config)
                
                print("Loading data...")
                df_full = predictor.load_data(csv_path)
                
                # Split data: exclude 2024-2025 for training
                df_train = df_full[df_full['year'] != 2024].copy()
                
                print(f"Full dataset: {len(df_full)} observations")
                print(f"Training dataset (no 2024-2025): {len(df_train)} observations")
                print(f"2024-2025 data: {len(df_full[df_full['year'] == 2024])} observations")
                
                print("Creating lag features...")
                df_with_lags = predictor.create_lag_features(df_train)
                
                print("Engineering features...")
                df_engineered = predictor.engineer_features(df_with_lags)
                
                print("Preparing features...")
                X, y = predictor.prepare_features(df_engineered)
                
                print(f"Final training dataset: {len(X)} observations, {X.shape[1] if len(X) > 0 else 0} features")
                
                if len(X) < 50:
                    print(f"Insufficient training data: {len(X)} observations (need at least 50)")
                    continue
                
                # Test different models
                models = ['random_forest', 'gradient_boosting', 'linear_regression']
                
                for model_type in models:
                    try:
                        print(f"\nTrying {model_type}...")
                        results = predictor.train_model(X, y, model_type=model_type)
                        
                        print(f"Model {model_type} succeeded with RÂ² = {results['r2']:.3f}")
                        
                        if results['r2'] > best_r2_pred:
                            best_r2_pred = results['r2']
                            best_results_pred = results
                            best_config_pred = {**config, 'model_type': model_type, 'predictor': predictor}
                            print(f"New best prediction model: {model_type} with RÂ² = {results['r2']:.3f}")
                            
                            # Generate predictions for 2024-2025
                            try:
                                print("Generating 2024-2025 predictions...")
                                df_full_with_lags = predictor.create_lag_features(df_full)
                                df_full_engineered = predictor.engineer_features(df_full_with_lags)
                                predictions = predictor.predict_season(df_full_engineered, 2024)
                                best_predictions = predictions
                                print(f"Successfully generated predictions for {len(predictions)} players")
                            except Exception as e:
                                print(f"Error generating predictions: {e}")
                                import traceback
                                traceback.print_exc()
                                best_predictions = None
                            
                    except Exception as e:
                        print(f"Error with {model_type}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                        
            except Exception as e:
                print(f"Error with configuration {config}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Show results
        print(f"\n{'='*75}")
        print("RESULTS SUMMARY")
        print('='*75)
        
        if best_results_all:
            print(f"\nTRAINING A - ALL DATA (WITH TEAM CONTEXT):")
            print(f"  Best configuration: {best_config_all['lag_years']}-year lags, {best_config_all['model_type']}")
            print(f"  RÂ² Score: {best_results_all['r2']:.3f}")
            print(f"  RMSE: {best_results_all['rmse']:.2f}")
            print(f"  Training size: {best_results_all['n_train']}")
            
            # Plot results for all data and save
            plot_filename = f'nhl_plots/forward_model_v5_all_data_{timestamp}.png'
            best_config_all['predictor'].plot_results(best_results_all, save_path=plot_filename)
            
            # Show feature importance with categories
            print(f"\nTop 15 Most Important Features:")
            print("-" * 60)
            try:
                importance_df = best_config_all['predictor'].get_feature_importance(15)
                for idx, row in importance_df.iterrows():
                    feature_type = "TEAM" if any(row['feature'].replace('_lag1', '').replace('_lag2', '').replace('_lag3', '').endswith(str(i)) for i in range(1, 10)) else "INDIVIDUAL"
                    print(f"{idx+1:2d}. {row['feature']:35} {row['importance']:.4f} ({feature_type})")
            except Exception as e:
                print(f"Could not display feature importance: {e}")
        
        if best_results_pred and best_predictions is not None:
            print(f"\nTRAINING B - PREDICT 2024-2025 (WITH TEAM CONTEXT):")
            print(f"  Best configuration: {best_config_pred['lag_years']}-year lags, {best_config_pred['model_type']}")
            print(f"  RÂ² Score: {best_results_pred['r2']:.3f}")
            print(f"  RMSE: {best_results_pred['rmse']:.2f}")
            print(f"  Training size: {best_results_pred['n_train']}")
            
            # Plot results for prediction scenario and save
            plot_filename = f'nhl_plots/forward_model_v5_prediction_{timestamp}.png'
            best_config_pred['predictor'].plot_results(best_results_pred, save_path=plot_filename)
            
            # Show top 50 predictions
            print(f"\nTOP 50 PREDICTED POINTS FOR 2024-2025 SEASON (WITH TEAM CONTEXT):")
            print("=" * 70)
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
            print(f"All data plot saved to: nhl_plots/forward_model_v5_all_data_{timestamp}.png")
        if best_results_pred:
            print(f"Prediction plot saved to: nhl_plots/forward_model_v5_prediction_{timestamp}.png")
    
    finally:
        # Restore stdout and close the output file
        sys.stdout = output_manager.terminal
        output_manager.close()

if __name__ == "__main__":
    main()