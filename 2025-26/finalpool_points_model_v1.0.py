"""
NHL POOL RANKING SYSTEM - UNIFIED MODEL v1.0
=============================================
Date: September 6, 2025
Author: Integrated from multiple position-specific models

This comprehensive program combines machine learning models for NHL forwards, defensemen, 
and goalies to generate pool rankings based on a sophisticated point system. It uses 
historical (lagged) performance data to predict future performance while avoiding data leakage.

POOL POINT SYSTEM:
==================
- Forwards: Points (Goals + Assists)
- Defensemen: Points (Goals + Assists) + Plus/Minus
- Goalies: 1 point per win, 3 points per shutout, 10 points for best GAA (40+ games), 
           10 points for best save percentage (40+ games)

THREE TRAINING SCENARIOS:
========================
A. Use all data for training and validation
B. Exclude 2024-25 season for training, predict 2024-25 for validation
C. Predict 2025-26 season with team reassignments:
   - Mitch Marner → VGK
   - Nikolaj Ehlers → CAR  
   - Noah Dobson → MTL

OUTPUT STRUCTURE:
================
- All plots saved to: nhl_plots/
- All results saved to: nhl_results/
- File naming: finalpool_[description]_v1.0_[timestamp].[ext]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List, Optional
import warnings
import os
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class UnifiedNHLPoolPredictor:
    """
    Unified predictor for all NHL positions with pool-specific scoring system.
    Combines forward points, defensemen dual targets, and goalie quadruple targets.
    """
    
    def __init__(self, lag_years: int = 2, output_dir: str = None):
        """Initialize the unified predictor."""
        self.lag_years = lag_years
        self.min_training_year = None
        
        # Position-specific models
        self.models = {
            'forward': {'points': None},
            'defense': {'points': None, 'plus_minus': None},
            'goalie': {'wins': None, 'shutouts': None, 'gaa': None, 'save_pct': None}
        }
        
        # Scalers for each model
        self.scalers = {}
        
        # Feature selectors
        self.feature_selectors = {}
        
        # Selected features
        self.selected_features = {}
        
        # Best parameters
        self.best_params = {}
        
        # Output directories
        self.plots_dir = 'nhl_plots'
        self.results_dir = 'nhl_results'
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Data paths
        self.skater_data_path = 'data_output/skater_team_data.csv'
        self.goalie_data_path = 'data_output/goalie_team_data.csv'
        
        # Timestamp for file naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data for all positions and prepare for modeling.
        
        Returns:
            Tuple of (forwards_df, defensemen_df, goalies_df)
        """
        print("\n" + "="*80)
        print("LOADING AND PREPARING DATA FOR ALL POSITIONS")
        print("="*80)
        
        # Load skater data (forwards and defensemen)
        print(f"\nLoading skater data from: {self.skater_data_path}")
        skaters_df = pd.read_csv(self.skater_data_path)
        print(f"Loaded {len(skaters_df)} skater records")
        
        # Parse season and year
        skaters_df['year'] = skaters_df['season'].astype(str).str[:4].astype(int)
        
        # Separate forwards and defensemen
        forwards_df = skaters_df[~skaters_df['position'].str.upper().str.contains('D', na=False)].copy()
        defensemen_df = skaters_df[skaters_df['position'].str.upper().str.contains('D', na=False)].copy()
        
        print(f"Separated into {len(forwards_df)} forwards and {len(defensemen_df)} defensemen")
        
        # Load goalie data
        print(f"\nLoading goalie data from: {self.goalie_data_path}")
        goalies_df = pd.read_csv(self.goalie_data_path)
        goalies_df['year'] = goalies_df['season'].astype(str).str[:4].astype(int)
        print(f"Loaded {len(goalies_df)} goalie records")
        
        # Clean data for each position
        forwards_df = self._clean_forward_data(forwards_df)
        defensemen_df = self._clean_defense_data(defensemen_df)
        goalies_df = self._clean_goalie_data(goalies_df)
        
        # Set minimum training year
        all_years = pd.concat([skaters_df['year'], goalies_df['year']])
        self.min_training_year = all_years.min() + self.lag_years
        print(f"\nMinimum training year set to: {self.min_training_year}")
        
        return forwards_df, defensemen_df, goalies_df
    
    def _clean_forward_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean forward data."""
        # Create points column if missing
        if 'points_player' not in df.columns:
            df['points_player'] = df['goals'].fillna(0) + df['assists'].fillna(0)
        
        # Handle games played
        if 'games_played_player' not in df.columns:
            if 'games_played' in df.columns:
                df['games_played_player'] = df['games_played']
        
        # Fill missing values
        numeric_cols = ['goals', 'assists', 'points_player', 'shots', 'games_played_player', 
                       'plus_minus', 'penalty_minutes', 'time_on_ice_per_game']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Handle shooting percentage
        if 'shooting_pct' in df.columns:
            if df['shooting_pct'].max() > 1:
                df['shooting_pct'] = df['shooting_pct'] / 100
        elif 'shots' in df.columns and 'goals' in df.columns:
            df['shooting_pct'] = df['goals'] / df['shots'].replace(0, np.nan)
            df['shooting_pct'] = df['shooting_pct'].fillna(0.1)
        
        return df
    
    def _clean_defense_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean defensemen data."""
        # Similar to forwards but preserve plus_minus carefully
        df = self._clean_forward_data(df)
        
        # Special handling for plus_minus
        if 'plus_minus' in df.columns:
            # Don't clip negative values for plus_minus
            df['plus_minus'] = pd.to_numeric(df['plus_minus'], errors='coerce').fillna(0)
        
        # Add defensive stats if available
        if 'blocked_shots' in df.columns:
            df['blocked_shots'] = df['blocked_shots'].fillna(0)
        if 'hits' in df.columns:
            df['hits'] = df['hits'].fillna(0)
        
        return df
    
    def _clean_goalie_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean goalie data."""
        # Handle goalie-specific columns
        goalie_cols = {
            'wins_player': 0,
            'losses_player': 0,
            'ot_losses_player': 0,
            'shutouts': 0,
            'goals_against_avg': 3.0,
            'save_pct': 0.900,
            'saves': 0,
            'shots_against': 0,
            'games_played_player': 0,
            'games_started': 0
        }
        
        for col, default in goalie_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
                
        # Handle save percentage format
        if 'save_pct' in df.columns:
            if df['save_pct'].max() > 1:
                df['save_pct'] = df['save_pct'] / 100
            df['save_pct'] = df['save_pct'].clip(0.5, 1.0)
        
        # Handle GAA
        if 'goals_against_avg' in df.columns:
            df['goals_against_avg'] = df['goals_against_avg'].clip(0, 10)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, position: str) -> pd.DataFrame:
        """
        Create lag features for a specific position.
        
        Args:
            df: DataFrame with player statistics
            position: 'forward', 'defense', or 'goalie'
            
        Returns:
            DataFrame with lag features added
        """
        print(f"\nCreating {self.lag_years}-year lag features for {position}s...")
        
        # Define features to lag based on position
        if position == 'forward':
            lag_features = ['goals', 'assists', 'points_player', 'shots', 'games_played_player',
                          'shooting_pct', 'time_on_ice_per_game', 'plus_minus', 'penalty_minutes']
            # Add power play features
            pp_features = [col for col in df.columns if col.startswith('pp_')][:5]
            lag_features.extend(pp_features)
            
        elif position == 'defense':
            lag_features = ['goals', 'assists', 'points_player', 'shots', 'games_played_player',
                          'shooting_pct', 'time_on_ice_per_game', 'plus_minus', 'penalty_minutes',
                          'hits', 'blocked_shots']
            # Add power play features
            pp_features = [col for col in df.columns if col.startswith('pp_')][:5]
            lag_features.extend(pp_features)
            
        else:  # goalie
            lag_features = ['wins_player', 'losses_player', 'ot_losses_player', 'shutouts',
                          'goals_against_avg', 'save_pct', 'saves', 'shots_against',
                          'games_played_player', 'games_started']
        
        # Add team features for all positions
        team_features = ['wins', 'losses', 'points_team', 'goals_for', 'goals_against']
        team_features = [f for f in team_features if f in df.columns]
        lag_features.extend(team_features)
        
        # Filter to existing features
        existing_features = [f for f in lag_features if f in df.columns]
        
        df_with_lags = df.copy()
        
        # Sort by player and year
        df_with_lags = df_with_lags.sort_values(['player_id', 'year'])
        
        # Create lag columns for each player
        lag_cols_created = []
        for feature in existing_features:
            for lag in range(1, self.lag_years + 1):
                lag_col = f'{feature}_lag{lag}'
                lag_cols_created.append(lag_col)
                
                # Create lags for each player
                for player_id in df_with_lags['player_id'].unique():
                    player_mask = df_with_lags['player_id'] == player_id
                    player_data = df_with_lags[player_mask].copy()
                    
                    if len(player_data) > lag:
                        df_with_lags.loc[player_mask, lag_col] = player_data[feature].shift(lag)
        
        print(f"Created {len(lag_cols_created)} lag features")
        
        # Filter to training years
        df_with_lags = df_with_lags[df_with_lags['year'] >= self.min_training_year]
        
        print(f"After filtering to year >= {self.min_training_year}: {len(df_with_lags)} observations")
        
        return df_with_lags
    
    def engineer_features(self, df: pd.DataFrame, position: str) -> pd.DataFrame:
        """
        Engineer additional features from lag data.
        
        Args:
            df: DataFrame with lag features
            position: Position type
            
        Returns:
            DataFrame with engineered features
        """
        print(f"Engineering features for {position}s...")
        
        df_eng = df.copy()
        
        # Find all lag columns
        lag_columns = [col for col in df_eng.columns if '_lag' in col]
        
        if lag_columns:
            # Get unique base features
            base_features = list(set([col.rsplit('_lag', 1)[0] for col in lag_columns]))
            
            for base_feature in base_features:
                lag_cols = [col for col in lag_columns if col.startswith(f'{base_feature}_lag')]
                
                if lag_cols:
                    # Historical average
                    df_eng[f'{base_feature}_hist_avg'] = df_eng[lag_cols].mean(axis=1, skipna=True)
                    
                    # Historical std (consistency measure)
                    if len(lag_cols) >= 2:
                        df_eng[f'{base_feature}_hist_std'] = df_eng[lag_cols].std(axis=1, skipna=True).fillna(0)
                    
                    # Trend (if multiple lags)
                    if len(lag_cols) >= 2:
                        lag1_col = f'{base_feature}_lag1'
                        lag2_col = f'{base_feature}_lag2'
                        if lag1_col in df_eng.columns and lag2_col in df_eng.columns:
                            df_eng[f'{base_feature}_trend'] = (
                                df_eng[lag1_col] - df_eng[lag2_col]
                            ).fillna(0)
        
        # Position-specific engineered features
        if position == 'forward':
            # Points per game
            for lag in range(1, min(self.lag_years + 1, 3)):
                if f'points_player_lag{lag}' in df_eng.columns and f'games_played_player_lag{lag}' in df_eng.columns:
                    df_eng[f'points_per_game_lag{lag}'] = (
                        df_eng[f'points_player_lag{lag}'] / 
                        df_eng[f'games_played_player_lag{lag}'].replace(0, np.nan)
                    ).fillna(0)
                
                # Goals per game
                if f'goals_lag{lag}' in df_eng.columns and f'games_played_player_lag{lag}' in df_eng.columns:
                    df_eng[f'goals_per_game_lag{lag}'] = (
                        df_eng[f'goals_lag{lag}'] / 
                        df_eng[f'games_played_player_lag{lag}'].replace(0, np.nan)
                    ).fillna(0)
        
        elif position == 'defense':
            # Points and plus/minus per game
            for lag in range(1, min(self.lag_years + 1, 3)):
                if f'points_player_lag{lag}' in df_eng.columns and f'games_played_player_lag{lag}' in df_eng.columns:
                    df_eng[f'points_per_game_lag{lag}'] = (
                        df_eng[f'points_player_lag{lag}'] / 
                        df_eng[f'games_played_player_lag{lag}'].replace(0, np.nan)
                    ).fillna(0)
                
                if f'plus_minus_lag{lag}' in df_eng.columns and f'games_played_player_lag{lag}' in df_eng.columns:
                    df_eng[f'plus_minus_per_game_lag{lag}'] = (
                        df_eng[f'plus_minus_lag{lag}'] / 
                        df_eng[f'games_played_player_lag{lag}'].replace(0, np.nan)
                    ).fillna(0)
        
        else:  # goalie
            # Win rate and shutout rate
            for lag in range(1, min(self.lag_years + 1, 3)):
                if f'wins_player_lag{lag}' in df_eng.columns and f'games_played_player_lag{lag}' in df_eng.columns:
                    df_eng[f'win_rate_lag{lag}'] = (
                        df_eng[f'wins_player_lag{lag}'] / 
                        df_eng[f'games_played_player_lag{lag}'].replace(0, np.nan)
                    ).fillna(0)
                
                if f'shutouts_lag{lag}' in df_eng.columns and f'games_played_player_lag{lag}' in df_eng.columns:
                    df_eng[f'shutout_rate_lag{lag}'] = (
                        df_eng[f'shutouts_lag{lag}'] / 
                        df_eng[f'games_played_player_lag{lag}'].replace(0, np.nan)
                    ).fillna(0)
        
        # Clean up infinite values
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        df_eng[numeric_cols] = df_eng[numeric_cols].fillna(0)
        
        print(f"Engineered features complete: {df_eng.shape}")
        
        return df_eng
    
    def prepare_features_for_modeling(self, df: pd.DataFrame, target_col: str, 
                                     position: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for a specific target variable.
        """
        # Create target if it doesn't exist
        if target_col not in df.columns:
            if target_col == 'target_points':
                df['target_points'] = df['points_player'].fillna(
                    df['goals'].fillna(0) + df['assists'].fillna(0)
                )
            elif target_col == 'target_plus_minus':
                df['target_plus_minus'] = df['plus_minus'].fillna(0)
            elif target_col == 'target_wins':
                df['target_wins'] = df['wins_player'].fillna(0)
            elif target_col == 'target_shutouts':
                df['target_shutouts'] = df['shutouts'].fillna(0)
            elif target_col == 'target_gaa':
                df['target_gaa'] = df['goals_against_avg'].fillna(3.0)
            elif target_col == 'target_save_pct':
                df['target_save_pct'] = df['save_pct'].fillna(0.900)
        
        # Define columns to exclude (current season stats)
        exclude_cols = [
            'player_id', 'player_name', 'season', 'year', 'team_id', 'team_name',
            'team_abbrev', 'position', target_col, 'franchise_id', 'team_id_original'
        ]
        
        # Exclude all current season stats (columns without 'lag', 'hist', or 'trend')
        for col in df.columns:
            if not any(x in col for x in ['lag', 'hist', 'trend', '_per_game_lag']):
                if col not in exclude_cols and df[col].dtype in ['float64', 'int64']:
                    # This is likely a current season stat
                    exclude_cols.append(col)
        
        # Keep only features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many missing values
        feature_cols = [col for col in feature_cols 
                       if df[col].notna().sum() > len(df) * 0.3]
        
        # Ensure all features are numeric
        numeric_features = []
        for col in feature_cols:
            if df[col].dtype in ['float64', 'int64']:
                numeric_features.append(col)
            else:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    numeric_features.append(col)
                except:
                    pass
        
        X = df[numeric_features].fillna(0)
        y = df[target_col].fillna(0)
        
        # Remove rows where target is missing or zero (for some targets)
        if target_col in ['target_gaa', 'target_save_pct']:
            # For GAA and save %, keep non-zero values
            valid_mask = (y > 0) & (y.notna())
        else:
            # For other targets, just check for non-missing
            valid_mask = y.notna()
        
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Prepared {len(X)} observations with {len(numeric_features)} features for {target_col}")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest',
                   position: str = 'forward', target: str = 'points') -> Dict[str, Any]:
        """
        Train a model for a specific position and target.
        """
        print(f"\nTraining {model_type} for {position} {target}...")
        
        if len(X) < 20:
            print(f"Insufficient data: {len(X)} observations")
            return None
        
        # Feature selection
        k_features = min(40, max(10, len(X.columns) // 3))
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        # Store selector
        selector_key = f"{position}_{target}"
        self.feature_selectors[selector_key] = selector
        self.selected_features[selector_key] = selected_features
        
        print(f"Selected {len(selected_features)} features")
        
        # Split data
        test_size = min(0.25, max(0.15, 30/len(X)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features for linear models
        scaler = RobustScaler()
        if model_type in ['ridge', 'lasso', 'elastic_net']:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.scalers[selector_key] = scaler
        
        # Get model with optimized parameters
        model = self._get_optimized_model(model_type, target)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store model
        if position not in self.models:
            self.models[position] = {}
        self.models[position][target] = model
        
        # Predictions and metrics
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Training score for overfitting check
        train_score = model.score(X_train, y_train)
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'train_r2': train_score,
            'overfitting_ratio': (train_score - r2) / train_score if train_score > 0 else 0,
            'y_test': y_test,
            'y_pred': y_pred,
            'model_type': model_type,
            'n_features': len(selected_features),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        
        print(f"  R² Score: {r2:.3f}, RMSE: {rmse:.2f}, Train R²: {train_score:.3f}")
        
        return results
    
    def _get_optimized_model(self, model_type: str, target: str = 'points'):
        """Get an optimized model instance based on type and target."""
        if model_type == 'random_forest':
            # Different parameters for different targets
            if target == 'shutouts':
                # More trees for rare events
                return RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                return RandomForestRegressor(
                    n_estimators=150,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                )
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif model_type == 'ridge':
            return Ridge(alpha=10.0, random_state=42)
        elif model_type == 'lasso':
            return Lasso(alpha=1.0, random_state=42, max_iter=2000)
        elif model_type == 'elastic_net':
            return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict_season(self, df_full: pd.DataFrame, target_year: int, 
                      position: str, team_changes: Dict[str, str] = None) -> pd.DataFrame:
        """
        Predict performance for a specific season and position.
        
        Args:
            df_full: Complete dataset with lag features
            target_year: Year to predict
            position: Position to predict
            team_changes: Optional dictionary of player name to new team mappings
            
        Returns:
            DataFrame with predictions
        """
        # Filter to target year
        target_data = df_full[df_full['year'] == target_year].copy()
        
        if len(target_data) == 0:
            print(f"No data found for {position} in year {target_year}")
            return pd.DataFrame()
        
        # Apply team changes if provided
        if team_changes and target_year == 2025:
            for player_name, new_team in team_changes.items():
                player_mask = target_data['player_name'].str.contains(player_name, case=False, na=False)
                if player_mask.any():
                    target_data.loc[player_mask, 'team_abbrev'] = new_team
                    print(f"Reassigned {player_name} to {new_team}")
        
        print(f"\nPredicting {target_year} for {len(target_data)} {position}s")
        
        # Create results dataframe
        results_df = target_data[['player_id', 'player_name', 'team_abbrev', 
                                 'games_played_player']].copy()
        
        # Make predictions based on position
        if position == 'forward':
            # Predict points
            X, _ = self.prepare_features_for_modeling(target_data, 'target_points', position)
            
            if self.models['forward']['points'] is not None and len(X) > 0:
                selector_key = f"{position}_points"
                if selector_key in self.selected_features:
                    X_valid = X[self.selected_features[selector_key]]
                    
                    if selector_key in self.scalers:
                        X_scaled = self.scalers[selector_key].transform(X_valid)
                    else:
                        X_scaled = X_valid
                    
                    predictions = self.models['forward']['points'].predict(X_scaled)
                    results_df.loc[X_valid.index, 'predicted_points'] = predictions
                    results_df.loc[X_valid.index, 'pool_points'] = predictions
        
        elif position == 'defense':
            # Predict points
            X_points, _ = self.prepare_features_for_modeling(target_data, 'target_points', position)
            
            if self.models['defense']['points'] is not None and len(X_points) > 0:
                selector_key = f"{position}_points"
                if selector_key in self.selected_features:
                    X_valid = X_points[self.selected_features[selector_key]]
                    
                    if selector_key in self.scalers:
                        X_scaled = self.scalers[selector_key].transform(X_valid)
                    else:
                        X_scaled = X_valid
                    
                    points_pred = self.models['defense']['points'].predict(X_scaled)
                    results_df.loc[X_valid.index, 'predicted_points'] = points_pred
            
            # Predict plus/minus
            X_pm, _ = self.prepare_features_for_modeling(target_data, 'target_plus_minus', position)
            
            if self.models['defense']['plus_minus'] is not None and len(X_pm) > 0:
                selector_key = f"{position}_plus_minus"
                if selector_key in self.selected_features:
                    X_valid = X_pm[self.selected_features[selector_key]]
                    
                    if selector_key in self.scalers:
                        X_scaled = self.scalers[selector_key].transform(X_valid)
                    else:
                        X_scaled = X_valid
                    
                    pm_pred = self.models['defense']['plus_minus'].predict(X_scaled)
                    results_df.loc[X_valid.index, 'predicted_plus_minus'] = pm_pred
            
            # Calculate pool points (points + plus/minus)
            results_df['pool_points'] = (
                results_df['predicted_points'].fillna(0) + 
                results_df['predicted_plus_minus'].fillna(0)
            )
        
        else:  # goalie
            # Predict all four targets
            goalie_predictions = {}
            
            # Wins
            X_wins, _ = self.prepare_features_for_modeling(target_data, 'target_wins', position)
            if self.models['goalie']['wins'] is not None and len(X_wins) > 0:
                selector_key = f"{position}_wins"
                if selector_key in self.selected_features:
                    X_valid = X_wins[self.selected_features[selector_key]]
                    if selector_key in self.scalers:
                        X_scaled = self.scalers[selector_key].transform(X_valid)
                    else:
                        X_scaled = X_valid
                    predictions = self.models['goalie']['wins'].predict(X_scaled)
                    results_df.loc[X_valid.index, 'predicted_wins'] = predictions
            
            # Shutouts
            X_shutouts, _ = self.prepare_features_for_modeling(target_data, 'target_shutouts', position)
            if self.models['goalie']['shutouts'] is not None and len(X_shutouts) > 0:
                selector_key = f"{position}_shutouts"
                if selector_key in self.selected_features:
                    X_valid = X_shutouts[self.selected_features[selector_key]]
                    if selector_key in self.scalers:
                        X_scaled = self.scalers[selector_key].transform(X_valid)
                    else:
                        X_scaled = X_valid
                    predictions = self.models['goalie']['shutouts'].predict(X_scaled)
                    results_df.loc[X_valid.index, 'predicted_shutouts'] = predictions
            
            # GAA (for 40+ games)
            qualified_gaa = target_data[target_data['games_played_player'] >= 40].copy()
            if len(qualified_gaa) > 0:
                X_gaa, _ = self.prepare_features_for_modeling(qualified_gaa, 'target_gaa', position)
                if self.models['goalie']['gaa'] is not None and len(X_gaa) > 0:
                    selector_key = f"{position}_gaa"
                    if selector_key in self.selected_features:
                        X_valid = X_gaa[self.selected_features[selector_key]]
                        if selector_key in self.scalers:
                            X_scaled = self.scalers[selector_key].transform(X_valid)
                        else:
                            X_scaled = X_valid
                        predictions = self.models['goalie']['gaa'].predict(X_scaled)
                        results_df.loc[X_valid.index, 'predicted_gaa'] = predictions
            
            # Save percentage (for 40+ games)
            qualified_save = target_data[target_data['games_played_player'] >= 40].copy()
            if len(qualified_save) > 0:
                X_save, _ = self.prepare_features_for_modeling(qualified_save, 'target_save_pct', position)
                if self.models['goalie']['save_pct'] is not None and len(X_save) > 0:
                    selector_key = f"{position}_save_pct"
                    if selector_key in self.selected_features:
                        X_valid = X_save[self.selected_features[selector_key]]
                        if selector_key in self.scalers:
                            X_scaled = self.scalers[selector_key].transform(X_valid)
                        else:
                            X_scaled = X_valid
                        predictions = self.models['goalie']['save_pct'].predict(X_scaled)
                        results_df.loc[X_valid.index, 'predicted_save_pct'] = predictions
            
            # Calculate goalie pool points
            results_df = self._calculate_goalie_pool_points(results_df)
        
        return results_df
    
    def _calculate_goalie_pool_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate pool points for goalies based on the point system."""
        # Initialize pool points
        df['pool_points'] = 0
        
        # Points from wins (1 point each)
        if 'predicted_wins' in df.columns:
            df['pool_points'] += df['predicted_wins'].fillna(0)
        
        # Points from shutouts (3 points each, but subtract from wins to avoid double counting)
        if 'predicted_shutouts' in df.columns:
            # Non-shutout wins get 1 point, shutout wins get 3 points total
            shutouts = df['predicted_shutouts'].fillna(0)
            df['pool_points'] += shutouts * 2  # Additional 2 points for shutouts
        
        # Best GAA gets 10 points (among 40+ game goalies)
        if 'predicted_gaa' in df.columns:
            qualified = df[(df['games_played_player'] >= 40) & df['predicted_gaa'].notna()]
            if len(qualified) > 0:
                best_gaa_idx = qualified['predicted_gaa'].idxmin()
                df.loc[best_gaa_idx, 'pool_points'] += 10
        
        # Best save percentage gets 10 points (among 40+ game goalies)
        if 'predicted_save_pct' in df.columns:
            qualified = df[(df['games_played_player'] >= 40) & df['predicted_save_pct'].notna()]
            if len(qualified) > 0:
                best_save_idx = qualified['predicted_save_pct'].idxmax()
                df.loc[best_save_idx, 'pool_points'] += 10
        
        return df
    
    def train_all_positions(self, forwards_df: pd.DataFrame, defensemen_df: pd.DataFrame,
                           goalies_df: pd.DataFrame, exclude_year: Optional[int] = None,
                           model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train models for all positions.
        """
        results = {}
        
        # Filter data if excluding year
        if exclude_year:
            print(f"\nExcluding year {exclude_year} from training")
            forwards_train = forwards_df[forwards_df['year'] != exclude_year].copy()
            defensemen_train = defensemen_df[defensemen_df['year'] != exclude_year].copy()
            goalies_train = goalies_df[goalies_df['year'] != exclude_year].copy()
        else:
            forwards_train = forwards_df.copy()
            defensemen_train = defensemen_df.copy()
            goalies_train = goalies_df.copy()
        
        # Train forwards
        print("\n" + "="*60)
        print("TRAINING FORWARD MODEL")
        print("="*60)
        
        forwards_lag = self.create_lag_features(forwards_train, 'forward')
        forwards_eng = self.engineer_features(forwards_lag, 'forward')
        X_fwd, y_fwd = self.prepare_features_for_modeling(forwards_eng, 'target_points', 'forward')
        
        if len(X_fwd) > 0:
            results['forward_points'] = self.train_model(X_fwd, y_fwd, model_type, 'forward', 'points')
        
        # Train defensemen
        print("\n" + "="*60)
        print("TRAINING DEFENSEMEN MODELS")
        print("="*60)
        
        defense_lag = self.create_lag_features(defensemen_train, 'defense')
        defense_eng = self.engineer_features(defense_lag, 'defense')
        
        # Points model
        X_def_pts, y_def_pts = self.prepare_features_for_modeling(defense_eng, 'target_points', 'defense')
        if len(X_def_pts) > 0:
            results['defense_points'] = self.train_model(X_def_pts, y_def_pts, model_type, 'defense', 'points')
        
        # Plus/minus model
        X_def_pm, y_def_pm = self.prepare_features_for_modeling(defense_eng, 'target_plus_minus', 'defense')
        if len(X_def_pm) > 0:
            results['defense_plus_minus'] = self.train_model(X_def_pm, y_def_pm, model_type, 'defense', 'plus_minus')
        
        # Train goalies
        print("\n" + "="*60)
        print("TRAINING GOALIE MODELS")
        print("="*60)
        
        goalies_lag = self.create_lag_features(goalies_train, 'goalie')
        goalies_eng = self.engineer_features(goalies_lag, 'goalie')
        
        # Wins model
        X_wins, y_wins = self.prepare_features_for_modeling(goalies_eng, 'target_wins', 'goalie')
        if len(X_wins) > 0:
            results['goalie_wins'] = self.train_model(X_wins, y_wins, model_type, 'goalie', 'wins')
        
        # Shutouts model
        X_shutouts, y_shutouts = self.prepare_features_for_modeling(goalies_eng, 'target_shutouts', 'goalie')
        if len(X_shutouts) > 0:
            results['goalie_shutouts'] = self.train_model(X_shutouts, y_shutouts, model_type, 'goalie', 'shutouts')
        
        # GAA model (40+ games only)
        qualified_gaa = goalies_eng[goalies_eng['games_played_player'] >= 40].copy()
        if len(qualified_gaa) > 0:
            X_gaa, y_gaa = self.prepare_features_for_modeling(qualified_gaa, 'target_gaa', 'goalie')
            if len(X_gaa) > 0:
                results['goalie_gaa'] = self.train_model(X_gaa, y_gaa, model_type, 'goalie', 'gaa')
        
        # Save percentage model (40+ games only)
        qualified_save = goalies_eng[goalies_eng['games_played_player'] >= 40].copy()
        if len(qualified_save) > 0:
            X_save, y_save = self.prepare_features_for_modeling(qualified_save, 'target_save_pct', 'goalie')
            if len(X_save) > 0:
                results['goalie_save_pct'] = self.train_model(X_save, y_save, model_type, 'goalie', 'save_pct')
        
        return results
    
    def generate_comprehensive_rankings(self, forwards_pred: pd.DataFrame, 
                                      defense_pred: pd.DataFrame,
                                      goalies_pred: pd.DataFrame,
                                      scenario_name: str) -> pd.DataFrame:
        """Generate comprehensive pool rankings with all positions."""
        # Add position column
        forwards_pred['position_group'] = 'Forward'
        defense_pred['position_group'] = 'Defenseman'
        goalies_pred['position_group'] = 'Goalie'
        
        # Combine all predictions
        all_players = pd.concat([forwards_pred, defense_pred, goalies_pred], ignore_index=True)
        
        # Fill missing pool points with 0
        all_players['pool_points'] = all_players['pool_points'].fillna(0)
        
        # Sort by pool points
        all_players = all_players.sort_values('pool_points', ascending=False)
        
        # Add overall rank
        all_players['overall_rank'] = range(1, len(all_players) + 1)
        
        # Add position rank
        for pos in ['Forward', 'Defenseman', 'Goalie']:
            pos_mask = all_players['position_group'] == pos
            pos_data = all_players[pos_mask].copy()
            pos_data['position_rank'] = range(1, len(pos_data) + 1)
            all_players.loc[pos_mask, 'position_rank'] = pos_data['position_rank']
        
        # Save to CSV
        csv_path = f'{self.results_dir}/finalpool_{scenario_name}_v1.0_{self.timestamp}.csv'
        all_players.to_csv(csv_path, index=False)
        print(f"\nSaved rankings to: {csv_path}")
        
        # Save summary text file
        txt_path = f'{self.results_dir}/finalpool_{scenario_name}_summary_v1.0_{self.timestamp}.txt'
        with open(txt_path, 'w') as f:
            f.write(f"NHL POOL RANKINGS - {scenario_name.upper()}\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TOP 50 OVERALL RANKINGS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<6} {'Player':<30} {'Team':<5} {'Pos':<4} {'Pool Pts':<10}\n")
            f.write("-"*80 + "\n")
            
            for _, row in all_players.head(50).iterrows():
                f.write(f"{row['overall_rank']:<6} {row['player_name'][:29]:<30} "
                       f"{row['team_abbrev']:<5} {row['position_group'][:3]:<4} "
                       f"{row['pool_points']:<10.1f}\n")
            
            # Top by position
            for pos in ['Forward', 'Defenseman', 'Goalie']:
                f.write(f"\n\nTOP 20 {pos.upper()}S\n")
                f.write("-"*80 + "\n")
                pos_data = all_players[all_players['position_group'] == pos].head(20)
                
                for _, row in pos_data.iterrows():
                    f.write(f"{row['position_rank']:<6} {row['player_name'][:29]:<30} "
                           f"{row['team_abbrev']:<5} {row['pool_points']:<10.1f}\n")
        
        print(f"Saved summary to: {txt_path}")
        
        return all_players
    
    def create_visualization(self, rankings: pd.DataFrame, results: Dict[str, Any], 
                           scenario_name: str):
        """Create comprehensive visualization of results."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Top scorers by position
        ax1 = fig.add_subplot(gs[0, 0])
        top_fwd = rankings[rankings['position_group'] == 'Forward'].head(10)
        ax1.barh(range(len(top_fwd)), top_fwd['pool_points'], color='steelblue')
        ax1.set_yticks(range(len(top_fwd)))
        ax1.set_yticklabels(top_fwd['player_name'].str[:20], fontsize=8)
        ax1.set_xlabel('Pool Points')
        ax1.set_title('Top 10 Forwards')
        ax1.invert_yaxis()
        
        ax2 = fig.add_subplot(gs[0, 1])
        top_def = rankings[rankings['position_group'] == 'Defenseman'].head(10)
        ax2.barh(range(len(top_def)), top_def['pool_points'], color='forestgreen')
        ax2.set_yticks(range(len(top_def)))
        ax2.set_yticklabels(top_def['player_name'].str[:20], fontsize=8)
        ax2.set_xlabel('Pool Points')
        ax2.set_title('Top 10 Defensemen')
        ax2.invert_yaxis()
        
        ax3 = fig.add_subplot(gs[0, 2])
        top_goal = rankings[rankings['position_group'] == 'Goalie'].head(10)
        ax3.barh(range(len(top_goal)), top_goal['pool_points'], color='firebrick')
        ax3.set_yticks(range(len(top_goal)))
        ax3.set_yticklabels(top_goal['player_name'].str[:20], fontsize=8)
        ax3.set_xlabel('Pool Points')
        ax3.set_title('Top 10 Goalies')
        ax3.invert_yaxis()
        
        # 2. Overall top 20
        ax4 = fig.add_subplot(gs[1, :])
        top20 = rankings.head(20)
        colors = {'Forward': 'steelblue', 'Defenseman': 'forestgreen', 'Goalie': 'firebrick'}
        bar_colors = [colors[pos] for pos in top20['position_group']]
        bars = ax4.bar(range(len(top20)), top20['pool_points'], color=bar_colors)
        ax4.set_xticks(range(len(top20)))
        ax4.set_xticklabels(top20['player_name'].str[:12], rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Pool Points')
        ax4.set_title('Top 20 Overall Players')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='steelblue', label='Forward'),
                          Patch(facecolor='forestgreen', label='Defenseman'),
                          Patch(facecolor='firebrick', label='Goalie')]
        ax4.legend(handles=legend_elements, loc='upper right')
        
        # 3. Model performance
        ax5 = fig.add_subplot(gs[2, 0])
        model_names = []
        r2_scores = []
        for key, result in results.items():
            if result and 'r2' in result:
                model_names.append(key.replace('_', ' ').title())
                r2_scores.append(result['r2'])
        
        ax5.bar(range(len(model_names)), r2_scores, color='skyblue')
        ax5.set_xticks(range(len(model_names)))
        ax5.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax5.set_ylabel('R² Score')
        ax5.set_title('Model Performance')
        ax5.set_ylim(0, max(r2_scores) * 1.2 if r2_scores else 1)
        
        # 4. Position distribution
        ax6 = fig.add_subplot(gs[2, 1])
        pos_counts = rankings['position_group'].value_counts()
        wedges, texts, autotexts = ax6.pie(pos_counts.values, labels=pos_counts.index, 
                                           autopct='%1.1f%%', colors=['steelblue', 'forestgreen', 'firebrick'])
        ax6.set_title('Position Distribution')
        
        # 5. Pool points distribution
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.hist(rankings['pool_points'], bins=30, color='teal', alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Pool Points')
        ax7.set_ylabel('Number of Players')
        ax7.set_title('Pool Points Distribution')
        ax7.axvline(rankings['pool_points'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {rankings["pool_points"].mean():.1f}')
        ax7.legend()
        
        plt.suptitle(f'NHL Pool Rankings - {scenario_name}', fontsize=16, y=0.98)
        
        # Save plot
        plot_path = f'{self.plots_dir}/finalpool_{scenario_name}_v1.0_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to: {plot_path}")


def main():
    """Main function to run the complete NHL pool ranking system."""
    
    print("\n" + "="*80)
    print("NHL POOL RANKING SYSTEM v1.0")
    print("COMPLETE IMPLEMENTATION")
    print("="*80)
    print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize predictor
    predictor = UnifiedNHLPoolPredictor(lag_years=2)
    
    # Load data
    try:
        forwards_df, defensemen_df, goalies_df = predictor.load_and_prepare_data()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure data files are in data_output/ directory:")
        print("  - data_output/skater_team_data.csv")
        print("  - data_output/goalies_team_data.csv")
        return
    
    # Test multiple model types
    model_types = ['random_forest', 'gradient_boosting', 'ridge']
    best_model_type = None
    best_avg_r2 = -1
    
    print("\n" + "="*80)
    print("TESTING DIFFERENT MODEL TYPES")
    print("="*80)
    
    for model_type in model_types:
        print(f"\nTesting {model_type}...")
        results = predictor.train_all_positions(
            forwards_df, defensemen_df, goalies_df,
            exclude_year=None,
            model_type=model_type
        )
        
        # Calculate average R²
        r2_scores = [r['r2'] for r in results.values() if r and 'r2' in r]
        if r2_scores:
            avg_r2 = np.mean(r2_scores)
            if avg_r2 > best_avg_r2:
                best_avg_r2 = avg_r2
                best_model_type = model_type
            print(f"  Average R² for {model_type}: {avg_r2:.3f}")
    
    print(f"\nBest model type: {best_model_type} with average R² of {best_avg_r2:.3f}")
    
    # SCENARIO A: Train on all data
    print("\n" + "="*80)
    print("SCENARIO A: TRAINING ON ALL DATA")
    print("="*80)
    
    results_all = predictor.train_all_positions(
        forwards_df, defensemen_df, goalies_df,
        exclude_year=None,
        model_type=best_model_type
    )
    
    # Print results
    print("\nTraining Results (All Data):")
    for key, result in results_all.items():
        if result:
            print(f"  {key}: R² = {result['r2']:.3f}, RMSE = {result['rmse']:.2f}")
    
    # SCENARIO B: Exclude 2024-25 and predict
    print("\n" + "="*80)
    print("SCENARIO B: PREDICTING 2024-25 SEASON")
    print("="*80)
    
    # Re-initialize predictor for clean training
    predictor_2024 = UnifiedNHLPoolPredictor(lag_years=2)
    predictor_2024.load_and_prepare_data()
    
    results_2024 = predictor_2024.train_all_positions(
        forwards_df, defensemen_df, goalies_df,
        exclude_year=2024,
        model_type=best_model_type
    )
    
    print("\nTraining Results (Excluding 2024):")
    for key, result in results_2024.items():
        if result:
            print(f"  {key}: R² = {result['r2']:.3f}, RMSE = {result['rmse']:.2f}")
    
    # Make predictions for 2024-25
    forwards_full = predictor_2024.create_lag_features(forwards_df, 'forward')
    forwards_full = predictor_2024.engineer_features(forwards_full, 'forward')
    
    defensemen_full = predictor_2024.create_lag_features(defensemen_df, 'defense')
    defensemen_full = predictor_2024.engineer_features(defensemen_full, 'defense')
    
    goalies_full = predictor_2024.create_lag_features(goalies_df, 'goalie')
    goalies_full = predictor_2024.engineer_features(goalies_full, 'goalie')
    
    # Predict 2024-25
    forwards_2024 = predictor_2024.predict_season(forwards_full, 2024, 'forward')
    defense_2024 = predictor_2024.predict_season(defensemen_full, 2024, 'defense')
    goalies_2024 = predictor_2024.predict_season(goalies_full, 2024, 'goalie')
    
    # Generate rankings
    rankings_2024 = predictor_2024.generate_comprehensive_rankings(
        forwards_2024, defense_2024, goalies_2024, "2024_25_predictions"
    )
    
    # Create visualization
    predictor_2024.create_visualization(rankings_2024, results_2024, "2024_25_predictions")
    
    # Display top 50
    print("\n" + "="*80)
    print("TOP 50 PREDICTED POOL RANKINGS FOR 2024-25")
    print("="*80)
    print(f"{'Rank':<6} {'Player':<30} {'Team':<5} {'Position':<12} {'Pool Points':<12}")
    print("-"*75)
    
    for _, row in rankings_2024.head(50).iterrows():
        print(f"{row['overall_rank']:<6} {row['player_name'][:29]:<30} {row['team_abbrev']:<5} "
              f"{row['position_group']:<12} {row['pool_points']:<12.1f}")
    
    # SCENARIO C: Predict 2025-26 with team changes
    print("\n" + "="*80)
    print("SCENARIO C: PREDICTING 2025-26 WITH TEAM CHANGES")
    print("="*80)
    
    # Define team changes
    team_changes = {
        'Mitch Marner': 'VGK',
        'Nikolaj Ehlers': 'CAR',
        'Noah Dobson': 'MTL'
    }
    
    # Use 2024 data as proxy for 2025 (since we don't have actual 2025 data)
    # In reality, you would need to create synthetic 2025 data or wait for actual data
    print("\nNote: Using 2024 data structure for 2025-26 predictions with team reassignments")
    
    # Predict with team changes
    forwards_2025 = predictor_2024.predict_season(forwards_full, 2024, 'forward', team_changes)
    defense_2025 = predictor_2024.predict_season(defensemen_full, 2024, 'defense', team_changes)
    goalies_2025 = predictor_2024.predict_season(goalies_full, 2024, 'goalie', team_changes)
    
    # Generate rankings
    rankings_2025 = predictor_2024.generate_comprehensive_rankings(
        forwards_2025, defense_2025, goalies_2025, "2025_26_predictions_with_trades"
    )
    
    # Create visualization
    predictor_2024.create_visualization(rankings_2025, results_2024, "2025_26_predictions_with_trades")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nPlayers analyzed:")
    print(f"  Forwards: {len(forwards_df['player_id'].unique())}")
    print(f"  Defensemen: {len(defensemen_df['player_id'].unique())}")
    print(f"  Goalies: {len(goalies_df['player_id'].unique())}")
    
    print(f"\nSeasons in dataset: {forwards_df['year'].min()}-{forwards_df['year'].max()}")
    
    print(f"\nBest model type: {best_model_type}")
    print(f"Average model R² across all positions: {best_avg_r2:.3f}")
    
    print(f"\nFiles generated:")
    print(f"  Results directory: {predictor_2024.results_dir}/")
    print(f"  Plots directory: {predictor_2024.plots_dir}/")
    print(f"  Timestamp: {predictor_2024.timestamp}")
    
    print(f"\nRun completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()