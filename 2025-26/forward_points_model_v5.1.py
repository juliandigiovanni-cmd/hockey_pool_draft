# Date: August 18, 2025
# Author: Julian di Giovanni w/Claude.ai - Enhanced Regularization Edition
# Version: 5.1 - Robust Regularization Edition with Team Context

"""
ENHANCED PROGRAM OVERVIEW: NHL Forward Points Prediction with Team Context + Robust Regularization
==================================================================================================

This enhanced program builds machine learning models to predict NHL forward scoring using 
historical (lagged) performance data including both individual stats and team context,
with comprehensive regularization to prevent overfitting. It implements a rigorous approach 
to avoid data leakage by excluding all contemporaneous statistics while leveraging historical 
team performance.

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

2. LAG FEATURE CREATION WITH TEAM CONTEXT:
   - Creates 1, 2, or 3-year historical lags for key statistics:
     * Individual stats: goals, assists, shots, games_played, shooting_pct, time_on_ice_per_game
     * Power play stats: pp_goals, pp_assists, pp_shots, etc.
     * Team statistics: variables ending with 1, 2, 3, etc. (team performance metrics)
   - Only creates lags where sufficient historical data exists
   - Selectively includes most important team statistics to prevent feature explosion
   - Filters data to years where complete lag history is available
   - Tracks players with sufficient history for reliable predictions

3. ENHANCED FEATURE ENGINEERING WITH TEAM CONTEXT:
   - Creates advanced metrics from lagged data:
     * Historical averages for individual stats (e.g., goals_hist_avg, assists_hist_avg)
     * Historical averages for team stats (e.g., team_goals1_hist_avg)
     * Trend indicators (lag1 vs lag2 performance changes)
     * Momentum indicators (lag1 vs historical average)
     * Per-game rates for all lagged seasons (goals_per_game_lag1, etc.)
     * Historical points totals and per-game averages
     * Team context metrics from lagged team statistics
     * Consistency measures (historical standard deviations)
   - All engineered features use ONLY historical data (no current season info)

4. FEATURE PREPARATION WITH SELECTION:
   - EXCLUDES contemporaneous variables to prevent data leakage:
     * Current season individual: goals, assists, shots, games_played, etc.
     * Current season power play stats (pp* without lag)
     * Current season team stats (variables ending in 1,2,3 without lag)
   - INCLUDES historical and team context:
     * All lagged individual statistics (goals_lag1, assists_lag2, etc.)
     * All lagged power play statistics (pp_goals_lag1, etc.)
     * All lagged team statistics (team_goals1_lag1, team_assists2_lag1, etc.)
     * Engineered historical features
   - AUTOMATIC FEATURE SELECTION:
     * Uses SelectKBest with f_regression scoring
     * Reduces feature set from 100+ to ~50 most important features
     * Prevents curse of dimensionality and reduces overfitting
   - Displays complete list of features used in training with categorization
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
   - Tests realistic prediction scenario with team context and overfitting prevention

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
   - Feature importance breakdown showing individual vs team stat contributions

KEY ENHANCED FEATURES:
=====================
- NO DATA LEAKAGE: Uses only historical data for predictions
- ROBUST REGULARIZATION: Comprehensive overfitting prevention across all model types
- TEAM CONTEXT: Includes historical team performance as predictive features with regularization
- AUTOMATIC FEATURE SELECTION: Reduces dimensionality and improves generalization
- PRESERVES ELITE PERFORMANCE: No artificial capping of high-scoring seasons
- ENHANCED VALIDATION: Tests on held-out future season with overfitting monitoring
- COMPREHENSIVE: Includes individual stats, power play, team statistics, and consistency measures
- FLEXIBLE: Tests multiple lag periods and regularized model types
- INTERPRETABLE: Shows feature importance categorized by individual vs team contributions
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

TEAM CONTEXT INTEGRATION:
========================
- Historical team performance metrics as lagged features
- Selective inclusion of most important team statistics to prevent feature explosion
- Team-based engineered features (trends, averages, consistency measures)
- Feature importance analysis showing individual vs team stat contributions
- Balanced approach to individual and team context without data leakage

TECHNICAL APPROACH:
==================
- Handles missing data intelligently with domain knowledge
- Preserves all legitimate elite performance data (no artificial capping)
- Creates meaningful lag features for time series nature of hockey data
- Incorporates team context while avoiding contemporaneous data leakage
- Uses ensemble methods and regularized linear models for comparison
- Implements proper train/test splits to avoid overfitting
- Scales features appropriately for different model types with robust scaling
- Provides extensive logging for transparency and debugging
- Monitors and prevents overfitting through multiple complementary techniques

OUTPUT:
=======
1. Data loading and cleaning summary with team context integration
2. Enhanced feature engineering details with complete feature list and team statistics
3. Training results for both scenarios (A & B) with regularization metrics
4. Enhanced model performance metrics and diagnostic plots including overfitting analysis
5. Top 50 predicted scorers for 2024-2025 season with actual results and confidence intervals
6. Feature importance rankings categorized by individual vs team contributions
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

class EnhancedHockeyPointsPredictorWithTeamContext:
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
        """Create lagged features for previous seasons including selective team statistics."""
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
        """Create additional engineered features from lag data including team context."""
        df_eng = df.copy()
        
        print(f"\nEngineering features with team context...")
        
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
            individual_base = [f for f in lag_base_features if not any(f.endswith(str(i)) for i in range(1, 10)) and not f.startswith('pp')]
            team_base = [f for f in lag_base_features if any(f.endswith(str(i)) for i in range(1, 10))]
            pp_base = [f for f in lag_base_features if f.startswith('pp')]
            print(f"  Individual: {len(individual_base)}, Team: {len(team_base)}, PP: {len(pp_base)}")
            
            for base_feature in lag_base_features:
                lag_cols = [col for col in lag_columns if col.startswith(f'{base_feature}_lag')]
                
                if lag_cols:
                    # Average over available lags (handle missing values)
                    df_eng[f'{base_feature}_hist_avg'] = df_eng[lag_cols].mean(axis=1, skipna=True)
                    
                    # Historical variance (stability measure)
                    if len(lag_cols) >= 2:
                        df_eng[f'{base_feature}_hist_std'] = df_eng[lag_cols].std(axis=1, skipna=True).fillna(0)
                    
                    # Recent trend (lag1 vs lag2) - only if we have both
                    if len(lag_cols) >= 2 and f'{base_feature}_lag1' in df_eng.columns and f'{base_feature}_lag2' in df_eng.columns:
                        trend = df_eng[f'{base_feature}_lag1'] - df_eng[f'{base_feature}_lag2']
                        df_eng[f'{base_feature}_trend'] = trend.fillna(0)
                        
                    # Momentum (lag1 vs historical average)
                    if f'{base_feature}_lag1' in df_eng.columns:
                        momentum = df_eng[f'{base_feature}_lag1'] - df_eng[f'{base_feature}_hist_avg']
                        df_eng[f'{base_feature}_momentum'] = momentum.fillna(0)
            
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
        """Prepare feature matrix and target variable using ONLY lagged features including team context."""
        print(f"\nStarting feature preparation with {len(df)} observations and {len(df.columns)} columns")
        
        # Exclude ALL contemporaneous variables 
        exclude_cols = [
            'player_id', 'season', 'season_str', 'year', 
            'points', 'goals', 'assists', 'total_points',
            'position', 'pos', 'Position', 'Pos',
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
        engineered_features = [f for f in self.feature_columns if 'hist' in f or 'trend' in f or 'momentum' in f or 'consistency' in f]
        other_features = [f for f in self.feature_columns if f not in individual_features + pp_features + team_features + engineered_features]
        
        print(f"\nFEATURE BREAKDOWN WITH TEAM CONTEXT:")
        print("-" * 60)
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
                print("WARNING: High feature to observation ratio - feature selection recommended")
        
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
        print(f"\nTraining {model_type} with enhanced regularization and team context:")
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
        """Get feature importance from trained model with team context categorization."""
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
        
        # Add feature type categorization
        def categorize_feature(feature_name):
            if any(feature_name.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '').endswith(str(i)) for i in range(1, 10)):
                return "TEAM"
            elif feature_name.startswith('pp'):
                return "POWERPLAY"
            else:
                return "INDIVIDUAL"
        
        importance_df['type'] = importance_df['feature'].apply(categorize_feature)
        
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
        """Plot enhanced model results with regularization metrics and team context."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(results['y_test'], results['y_pred'], alpha=0.6)
        min_val, max_val = results['y_test'].min(), results['y_test'].max()
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Points')
        axes[0, 0].set_ylabel('Predicted Points')
        axes[0, 0].set_title(f'Actual vs Predicted (Team Context + Regularization)\nR² = {results["r2"]:.3f}, Overfitting = {results["overfitting_ratio"]:.3f}')
        
        # Residuals
        residuals = results['y_test'] - results['y_pred']
        axes[0, 1].scatter(results['y_pred'], residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Points')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Feature Importance with categorization
        try:
            importance_df = self.get_feature_importance(15)
            
            # Color-code by feature type
            colors = {'INDIVIDUAL': 'blue', 'TEAM': 'red', 'POWERPLAY': 'green'}
            bar_colors = [colors.get(t, 'gray') for t in importance_df['type']]
            
            bars = axes[0, 2].barh(range(len(importance_df)), importance_df['importance'], color=bar_colors, alpha=0.7)
            axes[0, 2].set_yticks(range(len(importance_df)))
            axes[0, 2].set_yticklabels(importance_df['feature'], fontsize=8)
            axes[0, 2].set_xlabel('Importance')
            axes[0, 2].set_title(f'Feature Importance (Top 15)\n{results["n_features_selected"]} features selected')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Individual'),
                             Patch(facecolor='red', alpha=0.7, label='Team'),
                             Patch(facecolor='green', alpha=0.7, label='PowerPlay')]
            axes[0, 2].legend(handles=legend_elements, loc='lower right', fontsize=8)
            
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
    Main function to run the enhanced hockey points predictor with team context and regularization
    """
    csv_path = '/Users/juliandigiovanni/Library/CloudStorage/Dropbox/hockeyanalytics/nhl_output/skater_team_data.csv'
    
    # Create output directories
    os.makedirs('nhl_plots', exist_ok=True)
    os.makedirs('nhl_results', exist_ok=True)
    
    # Setup output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'nhl_results/enhanced_forward_points_prediction_v5.1_{timestamp}.txt'
    
    # Setup output manager to write to both console and file
    output_manager = OutputManager(output_file)
    sys.stdout = output_manager
    
    try:
        print("Enhanced Hockey Points Predictor v5.1 - Team Context + Robust Regularization")
        print("=" * 85)
        print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results will be saved to: {output_file}")
        
        # TRAINING A: Use all data with enhanced regularization and team context
        print(f"\n{'='*85}")
        print("TRAINING A: USING ALL DATA (TEAM CONTEXT + ENHANCED REGULARIZATION)")
        print('='*85)
        
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
            print('-' * 70)
            
            try:
                predictor = EnhancedHockeyPointsPredictorWithTeamContext(**config)
                
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
                            print(f"    New best model: {model_type} with adjusted score = {model_score:.3f}")
                            
                    except Exception as e:
                        print(f"    Error with {model_type}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error with configuration {config}: {e}")
                continue
        
        # TRAINING B: Omit 2024-2025 and predict it with enhanced regularization
        print(f"\n{'='*85}")
        print("TRAINING B: OMIT 2024-2025 AND PREDICT IT (TEAM CONTEXT + REGULARIZATION)")
        print('='*85)
        
        best_results_pred = None
        best_config_pred = None
        best_r2_pred = -1
        best_predictions = None
        
        for config in configs:
            print(f"\nTesting {config['lag_years']}-year lags (EXCLUDING 2024-2025)")
            print('-' * 70)
            
            try:
                predictor = EnhancedHockeyPointsPredictorWithTeamContext(**config)
                
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
                            print(f"    New best prediction model: {model_type} with adjusted score = {model_score:.3f}")
                            
                            # Generate predictions for 2024-2025
                            try:
                                print("    Generating 2024-2025 predictions...")
                                df_full_with_lags = predictor.create_lag_features(df_full)
                                df_full_engineered = predictor.engineer_features(df_full_with_lags)
                                predictions = predictor.predict_season(df_full_engineered, 2024)
                                best_predictions = predictions
                                print(f"    Successfully generated predictions for {len(predictions)} players")
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
        print(f"\n{'='*85}")
        print("ENHANCED RESULTS SUMMARY WITH TEAM CONTEXT")
        print('='*85)
        
        if best_results_all:
            print(f"\nTRAINING A - ALL DATA (TEAM CONTEXT + ENHANCED REGULARIZATION):")
            print(f"  Best configuration: {best_config_all['lag_years']}-year lags, {best_config_all['model_type']}")
            print(f"  R² Score: {best_results_all['r2']:.3f}")
            print(f"  RMSE: {best_results_all['rmse']:.2f}")
            print(f"  Overfitting Ratio: {best_results_all['overfitting_ratio']:.3f}")
            print(f"  Features Selected: {best_results_all['n_features_selected']}")
            print(f"  Training size: {best_results_all['n_train']}")
            
            # Plot results for all data and save
            plot_filename = f'nhl_plots/enhanced_forward_model_v5.1_all_data_{timestamp}.png'
            best_config_all['predictor'].plot_results(best_results_all, save_path=plot_filename)
            
            # Show feature importance with categories
            print(f"\nTop 15 Most Important Features (with Team Context):")
            print("-" * 75)
            try:
                importance_df = best_config_all['predictor'].get_feature_importance(15)
                for idx, row in importance_df.iterrows():
                    print(f"{idx+1:2d}. {row['feature']:35} {row['importance']:.4f} ({row['type']})")
            except Exception as e:
                print(f"Could not display feature importance: {e}")
        
        if best_results_pred and best_predictions is not None:
            print(f"\nTRAINING B - PREDICT 2024-2025 (TEAM CONTEXT + ENHANCED REGULARIZATION):")
            print(f"  Best configuration: {best_config_pred['lag_years']}-year lags, {best_config_pred['model_type']}")
            print(f"  R² Score: {best_results_pred['r2']:.3f}")
            print(f"  RMSE: {best_results_pred['rmse']:.2f}")
            print(f"  Overfitting Ratio: {best_results_pred['overfitting_ratio']:.3f}")
            print(f"  Features Selected: {best_results_pred['n_features_selected']}")
            print(f"  Training size: {best_results_pred['n_train']}")
            
            # Plot results for prediction scenario and save
            plot_filename = f'nhl_plots/enhanced_forward_model_v5.1_prediction_{timestamp}.png'
            best_config_pred['predictor'].plot_results(best_results_pred, save_path=plot_filename)
            
            # Show top 50 predictions
            print(f"\nTOP 50 PREDICTED POINTS FOR 2024-2025 SEASON (TEAM CONTEXT + ENHANCED MODEL):")
            print("=" * 85)
            if 'confidence_lower' in best_predictions.columns:
                print(f"{'Rank':<4} {'Player Name':<25} {'Predicted':<10} {'Actual':<8} {'Conf. Int.':<15}")
                print("-" * 85)
                
                for idx, row in best_predictions.head(50).iterrows():
                    actual_points = row['points'] if not pd.isna(row['points']) else 'N/A'
                    if 'confidence_lower' in row and not pd.isna(row['confidence_lower']):
                        conf_int = f"[{row['confidence_lower']:.1f}-{row['confidence_upper']:.1f}]"
                    else:
                        conf_int = "N/A"
                    print(f"{idx+1:<4} {row['player_name']:<25} {row['predicted_points']:<10.1f} {actual_points:<8} {conf_int:<15}")
            else:
                print(f"{'Rank':<4} {'Player Name':<30} {'Predicted Points':<15} {'Actual Points':<12}")
                print("-" * 85)
                
                for idx, row in best_predictions.head(50).iterrows():
                    actual_points = row['points'] if not pd.isna(row['points']) else 'N/A'
                    print(f"{idx+1:<4} {row['player_name']:<30} {row['predicted_points']:<15.1f} {actual_points:<12}")
        
        if not best_results_all and not best_results_pred:
            print("\nNo working configuration found!")
        
        print(f"\nRun completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {output_file}")
        if best_results_all:
            print(f"All data plot saved to: nhl_plots/enhanced_forward_model_v5.1_all_data_{timestamp}.png")
        if best_results_pred:
            print(f"Prediction plot saved to: nhl_plots/enhanced_forward_model_v5.1_prediction_{timestamp}.png")
        
        # Print comprehensive summary
        print(f"\n{'='*85}")
        print("COMPREHENSIVE ENHANCEMENT SUMMARY:")
        print('='*85)
        print("✓ Team Context Integration: Historical team performance as predictive features")
        print("✓ Feature Selection: Automatic reduction to most important features")
        print("✓ Hyperparameter Tuning: Grid search with cross-validation")
        print("✓ Tree Regularization: min_samples_split, min_samples_leaf, max_features")
        print("✓ Linear Regularization: Ridge (L2), Lasso (L1), ElasticNet (L1+L2)")
        print("✓ Early Stopping: Validation-based stopping for gradient boosting")
        print("✓ Time-Aware CV: TimeSeriesSplit for realistic validation")
        print("✓ Robust Scaling: RobustScaler for outlier resistance")
        print("✓ Overfitting Detection: Train vs test performance monitoring")
        print("✓ Feature Categorization: Individual vs Team vs PowerPlay contributions")
    
    finally:
        # Restore stdout and close the output file
        sys.stdout = output_manager.terminal
        output_manager.close()

if __name__ == "__main__":
    main()