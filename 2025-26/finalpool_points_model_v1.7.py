""" 
NHL POOL RANKING SYSTEM - UNIFIED MODEL v1.7
=============================================
Complete working version with:
- Enhanced goalie statistics (shutouts, GAA, save %)
- 2024-25 validation analysis (predicted vs actual)
- Feature importance analysis
- Comprehensive outputs (CSV, TXT, plots)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings
import os
from datetime import datetime
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class UnifiedNHLPoolPredictor:
    def __init__(self, lag_years=2):
        self.lag_years = lag_years
        self.min_training_year = None
        self.min_goalie_games_for_bonus = 40
        self.models = {
            'forward': {'points': None},
            'defense': {'points': None, 'plus_minus': None},
            'goalie': {'wins': None, 'shutouts': None, 'gaa': None, 'save_pct': None}
        }
        self.scalers = {}
        self.feature_selectors = {}
        self.selected_features = {}
        self.model_results = {}
        self.feature_importances = {}
        self.goalie_3yr_gp = {}
        self.forward_3yr_gp = {}
        self.defense_3yr_gp = {}
        self.actual_2024_gp = {}
        self.actual_2024_stats = {}
        
        self.plots_dir = 'nhl_plots'
        self.results_dir = 'nhl_results'
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.skater_data_path = 'data_output/skater_team_data.csv'
        self.goalie_data_path = 'data_output/goalie_team_data.csv'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_and_prepare_data(self):
        print("\nLoading data...")
        skaters_df = pd.read_csv(self.skater_data_path)
        skaters_df['year'] = skaters_df['season'].astype(str).str[:4].astype(int)
        
        forwards_df = skaters_df[~skaters_df['position'].str.upper().str.contains('D', na=False)].copy()
        defensemen_df = skaters_df[skaters_df['position'].str.upper().str.contains('D', na=False)].copy()
        
        goalies_df = pd.read_csv(self.goalie_data_path)
        goalies_df['year'] = goalies_df['season'].astype(str).str[:4].astype(int)
        
        forwards_df = self._clean_forward_data(forwards_df)
        defensemen_df = self._clean_defense_data(defensemen_df)
        goalies_df = self._clean_goalie_data(goalies_df)
        
        self._calculate_3yr_avg_gp(forwards_df, 'forward')
        self._calculate_3yr_avg_gp(defensemen_df, 'defense')
        self._calculate_3yr_avg_gp(goalies_df, 'goalie')
        
        self._store_actual_2024_stats(forwards_df, defensemen_df, goalies_df)
        
        all_years = pd.concat([skaters_df['year'], goalies_df['year']])
        self.min_training_year = all_years.min() + self.lag_years
        
        print(f"  Forwards: {len(forwards_df['player_id'].unique())} players")
        print(f"  Defensemen: {len(defensemen_df['player_id'].unique())} players")
        print(f"  Goalies: {len(goalies_df['player_id'].unique())} players")
        
        return forwards_df, defensemen_df, goalies_df
    
    def _calculate_3yr_avg_gp(self, df, position):
        gp_dict = {}
        for player_id in df['player_id'].unique():
            player_data = df[df['player_id'] == player_id].sort_values('year')
            if 'games_played_player' in player_data.columns:
                recent_gp = player_data['games_played_player'].tail(3).values
            else:
                recent_gp = player_data.get('games_played', []).tail(3).values
            
            valid_gp = recent_gp[recent_gp > 0] if len(recent_gp) > 0 else []
            if len(valid_gp) > 0:
                gp_dict[player_id] = min(np.mean(valid_gp), 82)
            else:
                defaults = {'forward': 70, 'defense': 72, 'goalie': 20}
                gp_dict[player_id] = defaults.get(position, 50)
        
        if position == 'forward':
            self.forward_3yr_gp = gp_dict
        elif position == 'defense':
            self.defense_3yr_gp = gp_dict
        else:
            self.goalie_3yr_gp = gp_dict
    
    def _store_actual_2024_stats(self, forwards_df, defensemen_df, goalies_df):
        """Store actual 2024 statistics for validation"""
        # Store forward stats
        for _, row in forwards_df[forwards_df['year'] == 2024].iterrows():
            player_id = row['player_id']
            self.actual_2024_stats[player_id] = {
                'player_name': row['player_name'],
                'position': 'Forward',
                'team': row.get('team_abbrev', 'N/A'),
                'games_played': row.get('games_played_player', 0),
                'points': row.get('points_player', 0),
                'goals': row.get('goals', 0),
                'assists': row.get('assists', 0)
            }
        
        # Store defense stats
        for _, row in defensemen_df[defensemen_df['year'] == 2024].iterrows():
            player_id = row['player_id']
            self.actual_2024_stats[player_id] = {
                'player_name': row['player_name'],
                'position': 'Defense',
                'team': row.get('team_abbrev', 'N/A'),
                'games_played': row.get('games_played_player', 0),
                'points': row.get('points_player', 0),
                'plus_minus': row.get('plus_minus', 0),
                'goals': row.get('goals', 0),
                'assists': row.get('assists', 0)
            }
        
        # Store goalie stats
        for _, row in goalies_df[goalies_df['year'] == 2024].iterrows():
            player_id = row['player_id']
            self.actual_2024_stats[player_id] = {
                'player_name': row['player_name'],
                'position': 'Goalie',
                'team': row.get('team_abbrev', 'N/A'),
                'games_played': row.get('games_played_player', 0),
                'wins': row.get('wins_player', 0),
                'shutouts': row.get('shutouts', 0),
                'gaa': row.get('goals_against_avg', 0),
                'save_pct': row.get('save_pct', 0)
            }
    
    def _clean_forward_data(self, df):
        if 'points_player' not in df.columns:
            df['points_player'] = df['goals'].fillna(0) + df['assists'].fillna(0)
        if 'games_played_player' not in df.columns:
            if 'games_played' in df.columns:
                df['games_played_player'] = df['games_played']
        
        numeric_cols = ['goals', 'assists', 'points_player', 'shots', 'games_played_player', 
                       'plus_minus', 'penalty_minutes', 'time_on_ice_per_game']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df
    
    def _clean_defense_data(self, df):
        df = self._clean_forward_data(df)
        if 'plus_minus' in df.columns:
            df['plus_minus'] = pd.to_numeric(df['plus_minus'], errors='coerce').fillna(0)
        return df
    
    def _clean_goalie_data(self, df):
        goalie_cols = {
            'wins_player': 0, 'losses_player': 0, 'shutouts': 0,
            'goals_against_avg': 3.0, 'save_pct': 0.900,
            'games_played_player': 0, 'games_started': 0
        }
        for col, default in goalie_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)
        
        if 'save_pct' in df.columns:
            if df['save_pct'].max() > 1:
                df['save_pct'] = df['save_pct'] / 100
            df['save_pct'] = df['save_pct'].clip(0.5, 1.0)
        
        if 'goals_against_avg' in df.columns:
            df['goals_against_avg'] = df['goals_against_avg'].clip(0, 10)
        return df
    
    def create_lag_features(self, df, position):
        if position == 'forward':
            lag_features = ['goals', 'assists', 'points_player', 'shots', 'games_played_player']
        elif position == 'defense':
            lag_features = ['goals', 'assists', 'points_player', 'shots', 'games_played_player', 'plus_minus']
        else:
            lag_features = ['wins_player', 'shutouts', 'goals_against_avg', 'save_pct', 'games_played_player']
        
        existing_features = [f for f in lag_features if f in df.columns]
        df_with_lags = df.copy()
        df_with_lags = df_with_lags.sort_values(['player_id', 'year'])
        
        for feature in existing_features:
            for lag in range(1, self.lag_years + 1):
                lag_col = f'{feature}_lag{lag}'
                for player_id in df_with_lags['player_id'].unique():
                    player_mask = df_with_lags['player_id'] == player_id
                    player_data = df_with_lags[player_mask].copy()
                    if len(player_data) > lag:
                        df_with_lags.loc[player_mask, lag_col] = player_data[feature].shift(lag)
        
        df_with_lags = df_with_lags[df_with_lags['year'] >= self.min_training_year]
        return df_with_lags
    
    def create_synthetic_2024_rows(self, df, position):
        """Create synthetic 2024 rows for validation"""
        df_2023 = df[df['year'] == 2023].copy()
        df_2022 = df[df['year'] == 2022].copy()
        
        synthetic_rows = []
        for player_id in df_2023['player_id'].unique():
            player_2023 = df_2023[df_2023['player_id'] == player_id]
            if len(player_2023) == 0:
                continue
            
            synthetic_row = player_2023.iloc[0].to_dict()
            synthetic_row['year'] = 2024
            synthetic_row['season'] = '2024-25'
            
            player_2022 = df_2022[df_2022['player_id'] == player_id]
            
            if position == 'forward':
                lag_features = ['goals', 'assists', 'points_player', 'shots', 'games_played_player']
            elif position == 'defense':
                lag_features = ['goals', 'assists', 'points_player', 'shots', 'games_played_player', 'plus_minus']
            else:
                lag_features = ['wins_player', 'shutouts', 'goals_against_avg', 'save_pct', 'games_played_player']
            
            for feature in lag_features:
                if feature in df_2023.columns:
                    lag1_col = f'{feature}_lag1'
                    if len(player_2023) > 0 and feature in player_2023.columns:
                        synthetic_row[lag1_col] = player_2023.iloc[0][feature]
                    else:
                        synthetic_row[lag1_col] = 0
                    
                    lag2_col = f'{feature}_lag2'
                    if len(player_2022) > 0 and feature in player_2022.columns:
                        synthetic_row[lag2_col] = player_2022.iloc[0][feature]
                    else:
                        synthetic_row[lag2_col] = 0
            
            synthetic_rows.append(synthetic_row)
        
        if synthetic_rows:
            return pd.DataFrame(synthetic_rows)
        return pd.DataFrame()
    
    def create_synthetic_2025_rows(self, df, position):
        df_2024 = df[df['year'] == 2024].copy()
        df_2023 = df[df['year'] == 2023].copy()
        
        synthetic_rows = []
        for player_id in df_2024['player_id'].unique():
            player_2024 = df_2024[df_2024['player_id'] == player_id]
            if len(player_2024) == 0:
                continue
            
            synthetic_row = player_2024.iloc[0].to_dict()
            synthetic_row['year'] = 2025
            synthetic_row['season'] = '2025-26'
            
            player_2023 = df_2023[df_2023['player_id'] == player_id]
            
            if position == 'forward':
                lag_features = ['goals', 'assists', 'points_player', 'shots', 'games_played_player']
            elif position == 'defense':
                lag_features = ['goals', 'assists', 'points_player', 'shots', 'games_played_player', 'plus_minus']
            else:
                lag_features = ['wins_player', 'shutouts', 'goals_against_avg', 'save_pct', 'games_played_player']
            
            for feature in lag_features:
                if feature in df_2024.columns:
                    lag1_col = f'{feature}_lag1'
                    if len(player_2024) > 0 and feature in player_2024.columns:
                        synthetic_row[lag1_col] = player_2024.iloc[0][feature]
                    else:
                        synthetic_row[lag1_col] = 0
                    
                    lag2_col = f'{feature}_lag2'
                    if len(player_2023) > 0 and feature in player_2023.columns:
                        synthetic_row[lag2_col] = player_2023.iloc[0][feature]
                    else:
                        synthetic_row[lag2_col] = 0
            
            synthetic_rows.append(synthetic_row)
        
        if synthetic_rows:
            return pd.DataFrame(synthetic_rows)
        return pd.DataFrame()
    
    def prepare_features_for_modeling(self, df, target_col, position):
        if target_col not in df.columns:
            if target_col == 'target_points':
                total_points = df['points_player'].fillna(0)
                games = df['games_played_player'].replace(0, np.nan)
                df['target_points'] = (total_points / games).fillna(0)
            elif target_col == 'target_plus_minus':
                games = df['games_played_player'].replace(0, np.nan)
                df['target_plus_minus'] = (df['plus_minus'].fillna(0) / games).fillna(0)
            elif target_col == 'target_wins':
                games = df['games_played_player'].replace(0, np.nan)
                df['target_wins'] = (df['wins_player'].fillna(0) / games).fillna(0)
            elif target_col == 'target_shutouts':
                games = df['games_played_player'].replace(0, np.nan)
                df['target_shutouts'] = (df['shutouts'].fillna(0) / games).fillna(0)
            elif target_col == 'target_gaa':
                df['target_gaa'] = df['goals_against_avg'].fillna(3.0)
            elif target_col == 'target_save_pct':
                df['target_save_pct'] = df['save_pct'].fillna(0.900)
        
        exclude_cols = ['player_id', 'player_name', 'season', 'year', 'team_id', 'team_name',
                       'team_abbrev', 'position', target_col]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols and 'lag' in col]
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y
    
    def train_model(self, X, y, position, target):
        if len(X) < 20:
            return None
        
        k_features = min(20, max(5, len(X.columns) // 2))
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        selector_key = f"{position}_{target}"
        self.feature_selectors[selector_key] = selector
        self.selected_features[selector_key] = selected_features
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        if position not in self.models:
            self.models[position] = {}
        self.models[position][target] = model
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"  {position} {target}: R² = {r2:.3f}, MAE = {mae:.3f}")
        
        # Store feature importances
        feature_key = f"{position}_{target}"
        self.feature_importances[feature_key] = dict(zip(selected_features, model.feature_importances_))
        self.model_results[feature_key] = {
            'r2': r2,
            'mae': mae,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        return {'r2': r2, 'mae': mae, 'model': model}
    
    def calculate_goalie_pool_points(self, results_df):
        """
        Calculate goalie pool points with full bonus system
        """
        if len(results_df) == 0:
            return results_df
        
        # Initialize columns
        results_df['gaa_bonus'] = False
        results_df['sv_pct_bonus'] = False
        results_df['qualified_for_bonus'] = False
        
        # Calculate base points
        wins = results_df.get('predicted_wins', 0).fillna(0)
        shutouts = results_df.get('predicted_shutouts', 0).fillna(0)
        
        # Ensure shutouts don't exceed wins
        shutouts = np.minimum(shutouts, wins)
        
        # Calculate points
        non_shutout_wins = np.maximum(wins - shutouts, 0)
        results_df['pool_points'] = non_shutout_wins + (shutouts * 3)
        
        # Store breakdown
        results_df['points_from_wins'] = non_shutout_wins
        results_df['points_from_shutouts'] = shutouts * 3
        
        # Determine qualified goalies
        gp_col = 'projected_games' if 'projected_games' in results_df.columns else 'games_played_player'
        qualified_mask = results_df[gp_col] >= self.min_goalie_games_for_bonus
        results_df['qualified_for_bonus'] = qualified_mask
        
        # Award GAA bonus
        if 'predicted_gaa' in results_df.columns and qualified_mask.sum() > 0:
            qualified = results_df[qualified_mask].copy()
            if len(qualified) > 0:
                best_gaa_idx = qualified['predicted_gaa'].idxmin()
                if pd.notna(best_gaa_idx):
                    results_df.loc[best_gaa_idx, 'pool_points'] += 10
                    results_df.loc[best_gaa_idx, 'gaa_bonus'] = True
        
        # Award Save Percentage bonus
        if 'predicted_save_pct' in results_df.columns and qualified_mask.sum() > 0:
            qualified = results_df[qualified_mask].copy()
            if len(qualified) > 0:
                best_sv_idx = qualified['predicted_save_pct'].idxmax()
                if pd.notna(best_sv_idx):
                    results_df.loc[best_sv_idx, 'pool_points'] += 10
                    results_df.loc[best_sv_idx, 'sv_pct_bonus'] = True
        
        return results_df
    
    def validate_2024_predictions(self, forwards_df, defensemen_df, goalies_df):
        """Validate 2024 predictions against actual results"""
        print("\nValidating 2024-25 predictions...")
        
        # Generate 2024 predictions
        forwards_2024 = self.create_synthetic_2024_rows(forwards_df, 'forward')
        defense_2024 = self.create_synthetic_2024_rows(defensemen_df, 'defense')
        goalies_2024 = self.create_synthetic_2024_rows(goalies_df, 'goalie')
        
        # Predict for each position
        validation_results = []
        
        # Forward validation
        if len(forwards_2024) > 0:
            fwd_results = forwards_2024[['player_id', 'player_name', 'team_abbrev']].copy()
            fwd_results['projected_games'] = fwd_results['player_id'].map(self.forward_3yr_gp).fillna(70)
            
            if self.models['forward'].get('points'):
                X, _ = self.prepare_features_for_modeling(forwards_2024, 'target_points', 'forward')
                if len(X) > 0 and 'forward_points' in self.selected_features:
                    X_selected = X[self.selected_features['forward_points']]
                    per_game_pred = self.models['forward']['points'].predict(X_selected)
                    for i, idx in enumerate(X_selected.index):
                        if idx in fwd_results.index:
                            player_id = fwd_results.loc[idx, 'player_id']
                            gp = fwd_results.loc[idx, 'projected_games']
                            predicted_points = per_game_pred[i] * gp
                            
                            if player_id in self.actual_2024_stats:
                                actual = self.actual_2024_stats[player_id]
                                validation_results.append({
                                    'player_name': actual['player_name'],
                                    'position': 'Forward',
                                    'predicted_points': predicted_points,
                                    'actual_points': actual.get('points', 0),
                                    'predicted_gp': gp,
                                    'actual_gp': actual.get('games_played', 0),
                                    'error': predicted_points - actual.get('points', 0)
                                })
        
        # Defense validation
        if len(defense_2024) > 0:
            def_results = defense_2024[['player_id', 'player_name', 'team_abbrev']].copy()
            def_results['projected_games'] = def_results['player_id'].map(self.defense_3yr_gp).fillna(72)
            
            if self.models['defense'].get('points') and self.models['defense'].get('plus_minus'):
                # Points prediction
                X, _ = self.prepare_features_for_modeling(defense_2024, 'target_points', 'defense')
                points_pred = None
                if len(X) > 0 and 'defense_points' in self.selected_features:
                    X_selected = X[self.selected_features['defense_points']]
                    points_pred = self.models['defense']['points'].predict(X_selected)
                
                # Plus/minus prediction
                X, _ = self.prepare_features_for_modeling(defense_2024, 'target_plus_minus', 'defense')
                if len(X) > 0 and 'defense_plus_minus' in self.selected_features and points_pred is not None:
                    X_selected = X[self.selected_features['defense_plus_minus']]
                    pm_pred = self.models['defense']['plus_minus'].predict(X_selected)
                    
                    for i, idx in enumerate(X_selected.index):
                        if idx in def_results.index:
                            player_id = def_results.loc[idx, 'player_id']
                            gp = def_results.loc[idx, 'projected_games']
                            predicted_points = points_pred[i] * gp if i < len(points_pred) else 0
                            predicted_pm = pm_pred[i] * gp
                            predicted_total = predicted_points + predicted_pm
                            
                            if player_id in self.actual_2024_stats:
                                actual = self.actual_2024_stats[player_id]
                                actual_total = actual.get('points', 0) + actual.get('plus_minus', 0)
                                validation_results.append({
                                    'player_name': actual['player_name'],
                                    'position': 'Defense',
                                    'predicted_points': predicted_total,
                                    'actual_points': actual_total,
                                    'predicted_gp': gp,
                                    'actual_gp': actual.get('games_played', 0),
                                    'error': predicted_total - actual_total
                                })
        
        # Goalie validation
        if len(goalies_2024) > 0:
            goal_results = goalies_2024[['player_id', 'player_name', 'team_abbrev']].copy()
            goal_results['projected_games'] = goal_results['player_id'].map(self.goalie_3yr_gp).fillna(20)
            
            for stat in ['wins', 'shutouts', 'gaa', 'save_pct']:
                goal_results[f'predicted_{stat}'] = 0
                if self.models['goalie'].get(stat):
                    target = f'target_{stat}'
                    X, _ = self.prepare_features_for_modeling(goalies_2024, target, 'goalie')
                    key = f'goalie_{stat}'
                    if len(X) > 0 and key in self.selected_features:
                        X_selected = X[self.selected_features[key]]
                        predictions = self.models['goalie'][stat].predict(X_selected)
                        
                        for i, idx in enumerate(X_selected.index):
                            if idx in goal_results.index:
                                if stat in ['wins', 'shutouts']:
                                    gp = goal_results.loc[idx, 'projected_games']
                                    goal_results.loc[idx, f'predicted_{stat}'] = predictions[i] * gp
                                else:
                                    goal_results.loc[idx, f'predicted_{stat}'] = predictions[i]
            
            goal_results = self.calculate_goalie_pool_points(goal_results)
            
            for _, row in goal_results.iterrows():
                player_id = row['player_id']
                if player_id in self.actual_2024_stats:
                    actual = self.actual_2024_stats[player_id]
                    actual_wins = actual.get('wins', 0)
                    actual_shutouts = actual.get('shutouts', 0)
                    actual_pool_points = (actual_wins - actual_shutouts) + (actual_shutouts * 3)
                    
                    validation_results.append({
                        'player_name': actual['player_name'],
                        'position': 'Goalie',
                        'predicted_points': row['pool_points'],
                        'actual_points': actual_pool_points,
                        'predicted_gp': row['projected_games'],
                        'actual_gp': actual.get('games_played', 0),
                        'predicted_wins': row.get('predicted_wins', 0),
                        'actual_wins': actual_wins,
                        'predicted_shutouts': row.get('predicted_shutouts', 0),
                        'actual_shutouts': actual_shutouts,
                        'predicted_gaa': row.get('predicted_gaa', 0),
                        'actual_gaa': actual.get('gaa', 0),
                        'predicted_sv_pct': row.get('predicted_save_pct', 0),
                        'actual_sv_pct': actual.get('save_pct', 0),
                        'error': row['pool_points'] - actual_pool_points
                    })
        
        return pd.DataFrame(validation_results)
    
    def predict_2025_26_season(self, forwards_df, defensemen_df, goalies_df):
        print("\nGenerating 2025-26 predictions...")
        
        # Create synthetic 2025 rows
        forwards_2025 = self.create_synthetic_2025_rows(forwards_df, 'forward')
        defense_2025 = self.create_synthetic_2025_rows(defensemen_df, 'defense')
        goalies_2025 = self.create_synthetic_2025_rows(goalies_df, 'goalie')
        
        # Forward predictions
        fwd_results = pd.DataFrame()
        if len(forwards_2025) > 0:
            fwd_results = forwards_2025[['player_id', 'player_name', 'team_abbrev']].copy()
            fwd_results['projected_games'] = fwd_results['player_id'].map(self.forward_3yr_gp).fillna(70)
            fwd_results['pool_points'] = 0
            
            if self.models['forward'].get('points'):
                X, _ = self.prepare_features_for_modeling(forwards_2025, 'target_points', 'forward')
                if len(X) > 0 and 'forward_points' in self.selected_features:
                    X_selected = X[self.selected_features['forward_points']]
                    per_game_pred = self.models['forward']['points'].predict(X_selected)
                    for i, idx in enumerate(X_selected.index):
                        if idx in fwd_results.index:
                            gp = fwd_results.loc[idx, 'projected_games']
                            fwd_results.loc[idx, 'pool_points'] = per_game_pred[i] * gp
        
        # Defense predictions
        def_results = pd.DataFrame()
        if len(defense_2025) > 0:
            def_results = defense_2025[['player_id', 'player_name', 'team_abbrev']].copy()
            def_results['projected_games'] = def_results['player_id'].map(self.defense_3yr_gp).fillna(72)
            def_results['predicted_points'] = 0
            def_results['predicted_plus_minus'] = 0
            
            if self.models['defense'].get('points'):
                X, _ = self.prepare_features_for_modeling(defense_2025, 'target_points', 'defense')
                if len(X) > 0 and 'defense_points' in self.selected_features:
                    X_selected = X[self.selected_features['defense_points']]
                    per_game_pred = self.models['defense']['points'].predict(X_selected)
                    for i, idx in enumerate(X_selected.index):
                        if idx in def_results.index:
                            gp = def_results.loc[idx, 'projected_games']
                            def_results.loc[idx, 'predicted_points'] = per_game_pred[i] * gp
            
            if self.models['defense'].get('plus_minus'):
                X, _ = self.prepare_features_for_modeling(defense_2025, 'target_plus_minus', 'defense')
                if len(X) > 0 and 'defense_plus_minus' in self.selected_features:
                    X_selected = X[self.selected_features['defense_plus_minus']]
                    per_game_pred = self.models['defense']['plus_minus'].predict(X_selected)
                    for i, idx in enumerate(X_selected.index):
                        if idx in def_results.index:
                            gp = def_results.loc[idx, 'projected_games']
                            def_results.loc[idx, 'predicted_plus_minus'] = per_game_pred[i] * gp
            
            def_results['pool_points'] = def_results['predicted_points'] + def_results['predicted_plus_minus']
        
        # Goalie predictions with all stats
        goal_results = pd.DataFrame()
        if len(goalies_2025) > 0:
            goal_results = goalies_2025[['player_id', 'player_name', 'team_abbrev']].copy()
            goal_results['projected_games'] = goal_results['player_id'].map(self.goalie_3yr_gp).fillna(20)
            
            # Predict all goalie stats
            for stat in ['wins', 'shutouts', 'gaa', 'save_pct']:
                goal_results[f'predicted_{stat}'] = 0
                if self.models['goalie'].get(stat):
                    target = f'target_{stat}'
                    X, _ = self.prepare_features_for_modeling(goalies_2025, target, 'goalie')
                    key = f'goalie_{stat}'
                    if len(X) > 0 and key in self.selected_features:
                        X_selected = X[self.selected_features[key]]
                        predictions = self.models['goalie'][stat].predict(X_selected)
                        
                        for i, idx in enumerate(X_selected.index):
                            if idx in goal_results.index:
                                if stat in ['wins', 'shutouts']:
                                    gp = goal_results.loc[idx, 'projected_games']
                                    goal_results.loc[idx, f'predicted_{stat}'] = predictions[i] * gp
                                else:
                                    goal_results.loc[idx, f'predicted_{stat}'] = predictions[i]
            
            # Calculate pool points with bonus system
            goal_results = self.calculate_goalie_pool_points(goal_results)
        
        return fwd_results, def_results, goal_results
    
    def train_all_positions(self, forwards_df, defensemen_df, goalies_df):
        print("\nTraining models...")
        results = {}
        
        # Train forwards
        forwards_lag = self.create_lag_features(forwards_df, 'forward')
        X_fwd, y_fwd = self.prepare_features_for_modeling(forwards_lag, 'target_points', 'forward')
        if len(X_fwd) > 0:
            results['forward_points'] = self.train_model(X_fwd, y_fwd, 'forward', 'points')
        
        # Train defense
        defense_lag = self.create_lag_features(defensemen_df, 'defense')
        X_def, y_def = self.prepare_features_for_modeling(defense_lag, 'target_points', 'defense')
        if len(X_def) > 0:
            results['defense_points'] = self.train_model(X_def, y_def, 'defense', 'points')
        
        X_pm, y_pm = self.prepare_features_for_modeling(defense_lag, 'target_plus_minus', 'defense')
        if len(X_pm) > 0:
            results['defense_plus_minus'] = self.train_model(X_pm, y_pm, 'defense', 'plus_minus')
        
        # Train all goalie models
        print("\nTraining goalie models...")
        goalies_lag = self.create_lag_features(goalies_df, 'goalie')
        
        for stat, target in [('wins', 'target_wins'), ('shutouts', 'target_shutouts'),
                            ('gaa', 'target_gaa'), ('save_pct', 'target_save_pct')]:
            X, y = self.prepare_features_for_modeling(goalies_lag, target, 'goalie')
            if len(X) > 0:
                results[f'goalie_{stat}'] = self.train_model(X, y, 'goalie', stat)
        
        return results
    
    def generate_feature_importance_report(self):
        """Generate comprehensive feature importance analysis"""
        csv_path = f'{self.results_dir}/feature_importance_{self.timestamp}.csv'
        txt_path = f'{self.results_dir}/feature_importance_{self.timestamp}.txt'
        
        importance_data = []
        
        with open(txt_path, 'w') as f:
            f.write("FEATURE IMPORTANCE ANALYSIS - NHL POOL RANKING SYSTEM v1.7\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model_key, importances in self.feature_importances.items():
                f.write(f"\nModel: {model_key.upper()}\n")
                f.write("-" * 40 + "\n")
                
                # Sort features by importance
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                
                for rank, (feature, importance) in enumerate(sorted_features[:10], 1):
                    f.write(f"{rank:2}. {feature:30} {importance:.4f}\n")
                    importance_data.append({
                        'model': model_key,
                        'feature': feature,
                        'importance': importance,
                        'rank': rank
                    })
                
                # Add model performance metrics
                if model_key in self.model_results:
                    metrics = self.model_results[model_key]
                    f.write(f"\nModel Performance:\n")
                    f.write(f"  R² Score: {metrics['r2']:.3f}\n")
                    f.write(f"  MAE: {metrics['mae']:.3f}\n")
        
        # Save to CSV
        importance_df = pd.DataFrame(importance_data)
        importance_df.to_csv(csv_path, index=False)
        
        print(f"Saved feature importance analysis to:")
        print(f"  - CSV: {csv_path}")
        print(f"  - TXT: {txt_path}")
        
        return importance_df
    
    def generate_validation_report(self, validation_df):
        """Generate validation report comparing 2024 predictions to actuals"""
        csv_path = f'{self.results_dir}/validation_2024_{self.timestamp}.csv'
        txt_path = f'{self.results_dir}/validation_2024_{self.timestamp}.txt'
        
        # Save CSV
        validation_df.to_csv(csv_path, index=False)
        
        # Generate text report
        with open(txt_path, 'w') as f:
            f.write("2024-25 SEASON VALIDATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            f.write("OVERALL ACCURACY METRICS\n")
            f.write("-" * 40 + "\n")
            
            mae_overall = validation_df['error'].abs().mean()
            rmse_overall = np.sqrt((validation_df['error'] ** 2).mean())
            
            f.write(f"Mean Absolute Error: {mae_overall:.2f} points\n")
            f.write(f"RMSE: {rmse_overall:.2f} points\n")
            f.write(f"Total Players Validated: {len(validation_df)}\n\n")
            
            # By position
            for position in ['Forward', 'Defense', 'Goalie']:
                pos_data = validation_df[validation_df['position'] == position]
                if len(pos_data) > 0:
                    f.write(f"\n{position.upper()} VALIDATION\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Players: {len(pos_data)}\n")
                    f.write(f"MAE: {pos_data['error'].abs().mean():.2f}\n")
                    f.write(f"RMSE: {np.sqrt((pos_data['error'] ** 2).mean()):.2f}\n")
                    
                    # Create absolute error column for sorting
                    pos_data_copy = pos_data.copy()
                    pos_data_copy['abs_error'] = pos_data_copy['error'].abs()
                    
                    # Top 5 most accurate predictions
                    f.write(f"\nMost Accurate Predictions:\n")
                    accurate = pos_data_copy.nsmallest(5, 'abs_error')
                    for _, row in accurate.iterrows():
                        f.write(f"  {row['player_name'][:25]:25} Pred: {row['predicted_points']:.1f} "
                               f"Act: {row['actual_points']:.1f} Err: {row['error']:.1f}\n")
                    
                    # Top 5 largest errors
                    f.write(f"\nLargest Prediction Errors:\n")
                    errors = pos_data_copy.nlargest(5, 'abs_error')
                    for _, row in errors.iterrows():
                        f.write(f"  {row['player_name'][:25]:25} Pred: {row['predicted_points']:.1f} "
                               f"Act: {row['actual_points']:.1f} Err: {row['error']:.1f}\n")
            
            # Goalie-specific validation
            goalie_data = validation_df[validation_df['position'] == 'Goalie']
            if len(goalie_data) > 0 and 'predicted_wins' in goalie_data.columns:
                f.write(f"\n\nGOALIE DETAILED STATISTICS\n")
                f.write("-" * 40 + "\n")
                
                for stat in ['wins', 'shutouts', 'gaa', 'sv_pct']:
                    pred_col = f'predicted_{stat}'
                    actual_col = f'actual_{stat}'
                    if pred_col in goalie_data.columns and actual_col in goalie_data.columns:
                        pred_values = goalie_data[pred_col].dropna()
                        actual_values = goalie_data[actual_col].dropna()
                        if len(pred_values) > 0 and len(actual_values) > 0:
                            mae = np.mean(np.abs(pred_values - actual_values))
                            f.write(f"{stat.upper()}: MAE = {mae:.3f}\n")
        
        print(f"Saved validation report to:")
        print(f"  - CSV: {csv_path}")
        print(f"  - TXT: {txt_path}")
        
        return validation_df
    
    def generate_validation_plots(self, validation_df):
        """Generate plots for 2024 validation analysis"""
        
        # 1. Predicted vs Actual Scatter Plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('2024-25 Season: Predicted vs Actual Performance', fontsize=16)
        
        # Overall scatter
        ax = axes[0, 0]
        ax.scatter(validation_df['actual_points'], validation_df['predicted_points'], alpha=0.5)
        max_val = max(validation_df[['actual_points', 'predicted_points']].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Prediction')
        ax.set_xlabel('Actual Pool Points')
        ax.set_ylabel('Predicted Pool Points')
        ax.set_title('All Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # By position
        positions = ['Forward', 'Defense', 'Goalie']
        colors = ['blue', 'green', 'red']
        
        for idx, (pos, color) in enumerate(zip(positions, colors), 1):
            ax = axes.flatten()[idx]
            pos_data = validation_df[validation_df['position'] == pos]
            if len(pos_data) > 0:
                ax.scatter(pos_data['actual_points'], pos_data['predicted_points'], 
                         alpha=0.5, color=color)
                max_val = max(pos_data[['actual_points', 'predicted_points']].max())
                ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
                ax.set_xlabel('Actual Pool Points')
                ax.set_ylabel('Predicted Pool Points')
                ax.set_title(f'{pos}')
                
                # Add R² score
                if len(pos_data) > 1:
                    r, _ = pearsonr(pos_data['actual_points'], pos_data['predicted_points'])
                    ax.text(0.05, 0.95, f'R² = {r**2:.3f}', transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f'{self.plots_dir}/validation_scatter_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved validation scatter plot to: {plot_path}")
        
        # 2. Error Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prediction Error Analysis', fontsize=16)
        
        # Overall error distribution
        ax = axes[0, 0]
        ax.hist(validation_df['error'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
        ax.set_xlabel('Prediction Error (Predicted - Actual)')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Error Distribution')
        ax.legend()
        
        # Error by position
        ax = axes[0, 1]
        position_errors = [validation_df[validation_df['position'] == pos]['error'].values 
                          for pos in positions]
        bp = ax.boxplot(position_errors, labels=positions)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel('Prediction Error')
        ax.set_title('Error Distribution by Position')
        ax.grid(True, alpha=0.3)
        
        # Games played accuracy
        ax = axes[1, 0]
        ax.scatter(validation_df['actual_gp'], validation_df['predicted_gp'], alpha=0.5)
        max_gp = max(validation_df[['actual_gp', 'predicted_gp']].max())
        ax.plot([0, max_gp], [0, max_gp], 'r--', alpha=0.5)
        ax.set_xlabel('Actual Games Played')
        ax.set_ylabel('Predicted Games Played')
        ax.set_title('Games Played Prediction Accuracy')
        ax.grid(True, alpha=0.3)
        
        # Top errors - create absolute error column for sorting
        validation_df_copy = validation_df.copy()
        validation_df_copy['abs_error'] = validation_df_copy['error'].abs()
        top_errors = validation_df_copy.nlargest(10, 'abs_error')
        
        ax = axes[1, 1]
        y_pos = np.arange(len(top_errors))
        ax.barh(y_pos, top_errors['error'].values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name[:20] for name in top_errors['player_name'].values])
        ax.set_xlabel('Prediction Error')
        ax.set_title('Top 10 Prediction Errors')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plot_path = f'{self.plots_dir}/validation_errors_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved error analysis plot to: {plot_path}")
        
        # 3. Feature Importance Visualization
        if len(self.feature_importances) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Feature Importance by Model', fontsize=16)
            axes = axes.flatten()
            
            for idx, (model_key, importances) in enumerate(self.feature_importances.items()):
                if idx < len(axes):
                    ax = axes[idx]
                    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                    features, importance_values = zip(*sorted_features)
                    
                    y_pos = np.arange(len(features))
                    ax.barh(y_pos, importance_values)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([f[:20] for f in features])
                    ax.set_xlabel('Importance')
                    ax.set_title(model_key.replace('_', ' ').title())
                    ax.invert_yaxis()
            
            # Hide unused subplots
            for idx in range(len(self.feature_importances), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plot_path = f'{self.plots_dir}/feature_importance_{self.timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved feature importance plot to: {plot_path}")
    
    def generate_rankings(self, fwd_pred, def_pred, goal_pred, scenario):
        """Generate comprehensive rankings with all statistics"""
        # Add position column
        fwd_pred['position_group'] = 'Forward'
        def_pred['position_group'] = 'Defenseman'
        goal_pred['position_group'] = 'Goalie'
        
        # Combine all
        all_players = pd.concat([fwd_pred, def_pred, goal_pred], ignore_index=True)
        all_players['pool_points'] = all_players['pool_points'].fillna(0)
        all_players = all_players.sort_values('pool_points', ascending=False)
        all_players['overall_rank'] = range(1, len(all_players) + 1)
        
        # Save comprehensive CSV
        csv_path = f'{self.results_dir}/rankings_{scenario}_{self.timestamp}.csv'
        all_players.to_csv(csv_path, index=False)
        print(f"\nSaved rankings to: {csv_path}")
        
        # Generate detailed text report
        self.generate_text_report(all_players, fwd_pred, def_pred, goal_pred, scenario)
        
        # Generate visualization plots
        self.generate_plots(all_players, fwd_pred, def_pred, goal_pred, scenario)
        
        return all_players
    
    def generate_text_report(self, all_players, fwd_pred, def_pred, goal_pred, scenario):
        """Generate comprehensive text report with all statistics"""
        txt_path = f'{self.results_dir}/report_{scenario}_{self.timestamp}.txt'
        
        with open(txt_path, 'w') as f:
            f.write("NHL POOL RANKING SYSTEM v1.7 - DETAILED REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Season: {scenario.replace('_', '-')}\n\n")
            
            # Overall Top 30
            f.write("TOP 30 OVERALL RANKINGS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6} {'Player':<25} {'Pos':<4} {'Team':<5} {'Points':>8}\n")
            f.write("-" * 80 + "\n")
            for _, row in all_players.head(30).iterrows():
                f.write(f"{row['overall_rank']:<6} {row['player_name'][:24]:<25} "
                       f"{row['position_group'][:3]:<4} {row.get('team_abbrev', 'N/A'):<5} "
                       f"{row['pool_points']:>8.1f}\n")
            
            # Top Forwards
            f.write("\n\nTOP 20 FORWARDS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6} {'Player':<25} {'Team':<5} {'GP':>4} {'Points':>8}\n")
            f.write("-" * 80 + "\n")
            top_fwd = fwd_pred.nlargest(20, 'pool_points')
            for i, (_, row) in enumerate(top_fwd.iterrows(), 1):
                f.write(f"{i:<6} {row['player_name'][:24]:<25} "
                       f"{row.get('team_abbrev', 'N/A'):<5} "
                       f"{int(row['projected_games']):>4} {row['pool_points']:>8.1f}\n")
            
            # Top Defensemen
            f.write("\n\nTOP 20 DEFENSEMEN\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6} {'Player':<25} {'Team':<5} {'GP':>4} {'Pts':>6} {'+/-':>6} {'Total':>8}\n")
            f.write("-" * 80 + "\n")
            top_def = def_pred.nlargest(20, 'pool_points')
            for i, (_, row) in enumerate(top_def.iterrows(), 1):
                f.write(f"{i:<6} {row['player_name'][:24]:<25} "
                       f"{row.get('team_abbrev', 'N/A'):<5} "
                       f"{int(row['projected_games']):>4} "
                       f"{row['predicted_points']:>6.1f} "
                       f"{row['predicted_plus_minus']:>6.1f} "
                       f"{row['pool_points']:>8.1f}\n")
            
            # Top Goalies with detailed stats
            f.write("\n\nTOP 20 GOALIES (WITH DETAILED STATISTICS)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6} {'Player':<25} {'Team':<5} {'GP':>4} {'Total':>8}\n")
            f.write(f"{'':6} {'W':>8} {'SO':>6} {'GAA':>6} {'SV%':>6} {'Bonus':>12}\n")
            f.write("-" * 80 + "\n")
            
            top_goal = goal_pred.nlargest(20, 'pool_points')
            for i, (_, row) in enumerate(top_goal.iterrows(), 1):
                f.write(f"{i:<6} {row['player_name'][:24]:<25} "
                       f"{row.get('team_abbrev', 'N/A'):<5} "
                       f"{int(row['projected_games']):>4} "
                       f"{row['pool_points']:>8.1f}\n")
                
                # Detailed stats line
                bonus_str = ""
                if row.get('gaa_bonus', False):
                    bonus_str += "GAA "
                if row.get('sv_pct_bonus', False):
                    bonus_str += "SV% "
                
                f.write(f"{'':6} {row.get('predicted_wins', 0):>8.1f} "
                       f"{row.get('predicted_shutouts', 0):>6.1f} "
                       f"{row.get('predicted_gaa', 0):>6.3f} "
                       f"{row.get('predicted_save_pct', 0):>6.3f} "
                       f"{bonus_str:>12}\n")
            
            # Goalie Bonus Recipients
            f.write("\n\nGOALIE BONUS AWARDS (40+ GAMES REQUIRED)\n")
            f.write("-" * 80 + "\n")
            
            qualified_goalies = goal_pred[goal_pred['qualified_for_bonus'] == True]
            if len(qualified_goalies) > 0:
                gaa_winner = goal_pred[goal_pred['gaa_bonus'] == True]
                if len(gaa_winner) > 0:
                    winner = gaa_winner.iloc[0]
                    f.write(f"Best GAA (10 pts): {winner['player_name']} - "
                           f"{winner['predicted_gaa']:.3f} GAA\n")
                
                sv_winner = goal_pred[goal_pred['sv_pct_bonus'] == True]
                if len(sv_winner) > 0:
                    winner = sv_winner.iloc[0]
                    f.write(f"Best SV% (10 pts): {winner['player_name']} - "
                           f"{winner['predicted_save_pct']:.3f} SV%\n")
                
                f.write(f"\nTotal Qualified Goalies (40+ GP): {len(qualified_goalies)}\n")
            else:
                f.write("No goalies projected to play 40+ games\n")
            
            # Summary Statistics
            f.write("\n\nSUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Players Ranked: {len(all_players)}\n")
            f.write(f"  - Forwards: {len(fwd_pred)}\n")
            f.write(f"  - Defensemen: {len(def_pred)}\n")
            f.write(f"  - Goalies: {len(goal_pred)}\n")
            f.write(f"\nAverage Pool Points by Position:\n")
            f.write(f"  - Forwards: {fwd_pred['pool_points'].mean():.1f}\n")
            f.write(f"  - Defensemen: {def_pred['pool_points'].mean():.1f}\n")
            f.write(f"  - Goalies: {goal_pred['pool_points'].mean():.1f}\n")
            
        print(f"Saved text report to: {txt_path}")
    
    def generate_plots(self, all_players, fwd_pred, def_pred, goal_pred, scenario):
        """Generate comprehensive visualization plots"""
        
        # 1. Top Players by Position
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NHL Pool Rankings 2025-26 - Top Players Analysis', fontsize=16)
        
        # Top 15 Overall
        ax = axes[0, 0]
        top_15 = all_players.head(15)
        colors = {'Forward': 'blue', 'Defenseman': 'green', 'Goalie': 'red'}
        bar_colors = [colors[pos] for pos in top_15['position_group']]
        ax.barh(range(len(top_15)), top_15['pool_points'], color=bar_colors)
        ax.set_yticks(range(len(top_15)))
        ax.set_yticklabels([f"{row['player_name'][:20]}" for _, row in top_15.iterrows()])
        ax.set_xlabel('Pool Points')
        ax.set_title('Top 15 Overall')
        ax.invert_yaxis()
        
        # Top 10 Forwards
        ax = axes[0, 1]
        top_fwd = fwd_pred.nlargest(10, 'pool_points')
        ax.barh(range(len(top_fwd)), top_fwd['pool_points'], color='blue')
        ax.set_yticks(range(len(top_fwd)))
        ax.set_yticklabels([f"{row['player_name'][:20]}" for _, row in top_fwd.iterrows()])
        ax.set_xlabel('Pool Points')
        ax.set_title('Top 10 Forwards')
        ax.invert_yaxis()
        
        # Top 10 Defensemen
        ax = axes[1, 0]
        top_def = def_pred.nlargest(10, 'pool_points')
        ax.barh(range(len(top_def)), top_def['pool_points'], color='green')
        ax.set_yticks(range(len(top_def)))
        ax.set_yticklabels([f"{row['player_name'][:20]}" for _, row in top_def.iterrows()])
        ax.set_xlabel('Pool Points')
        ax.set_title('Top 10 Defensemen')
        ax.invert_yaxis()
        
        # Top 10 Goalies
        ax = axes[1, 1]
        top_goal = goal_pred.nlargest(10, 'pool_points')
        ax.barh(range(len(top_goal)), top_goal['pool_points'], color='red')
        ax.set_yticks(range(len(top_goal)))
        ax.set_yticklabels([f"{row['player_name'][:20]}" for _, row in top_goal.iterrows()])
        ax.set_xlabel('Pool Points')
        ax.set_title('Top 10 Goalies')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plot_path = f'{self.plots_dir}/top_players_{scenario}_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved top players plot to: {plot_path}")
        
        # 2. Distribution Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Pool Points Distribution Analysis', fontsize=16)
        
        # Overall distribution
        ax = axes[0, 0]
        ax.hist(all_players['pool_points'], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Pool Points')
        ax.set_ylabel('Number of Players')
        ax.set_title('Overall Distribution')
        ax.axvline(all_players['pool_points'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {all_players["pool_points"].mean():.1f}')
        ax.legend()
        
        # Position comparison boxplot
        ax = axes[0, 1]
        position_data = [fwd_pred['pool_points'], def_pred['pool_points'], goal_pred['pool_points']]
        bp = ax.boxplot(position_data, labels=['Forwards', 'Defensemen', 'Goalies'])
        ax.set_ylabel('Pool Points')
        ax.set_title('Distribution by Position')
        ax.grid(True, alpha=0.3)
        
        # Goalie statistics breakdown
        ax = axes[1, 0]
        if len(goal_pred) > 0:
            top_20_goalies = goal_pred.nlargest(20, 'pool_points')
            x = range(len(top_20_goalies))
            width = 0.35
            
            wins_points = top_20_goalies.get('points_from_wins', 0)
            shutout_points = top_20_goalies.get('points_from_shutouts', 0)
            
            ax.bar([i - width/2 for i in x], wins_points, width, label='Win Points', color='lightblue')
            ax.bar([i + width/2 for i in x], shutout_points, width, label='Shutout Points', color='darkblue')
            
            ax.set_xlabel('Goalie Rank')
            ax.set_ylabel('Points')
            ax.set_title('Top 20 Goalies - Points Breakdown')
            ax.legend()
        
        # Games Played vs Points Scatter
        ax = axes[1, 1]
        for df, label, color in [(fwd_pred, 'Forwards', 'blue'),
                                 (def_pred, 'Defensemen', 'green'),
                                 (goal_pred, 'Goalies', 'red')]:
            if 'projected_games' in df.columns:
                ax.scatter(df['projected_games'], df['pool_points'], 
                         alpha=0.5, s=20, label=label, color=color)
        ax.set_xlabel('Projected Games Played')
        ax.set_ylabel('Pool Points')
        ax.set_title('Games Played vs Pool Points')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f'{self.plots_dir}/distribution_analysis_{scenario}_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved distribution plot to: {plot_path}")
        
        # 3. Goalie-specific analysis
        if len(goal_pred) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Goalie Statistics Analysis', fontsize=16)
            
            top_goalies = goal_pred.nlargest(15, 'pool_points')
            
            # Wins vs Shutouts
            ax = axes[0, 0]
            ax.scatter(top_goalies['predicted_wins'], top_goalies['predicted_shutouts'], s=100, alpha=0.6)
            for idx, row in top_goalies.iterrows():
                ax.annotate(row['player_name'][:10], 
                          (row['predicted_wins'], row['predicted_shutouts']),
                          fontsize=8, alpha=0.7)
            ax.set_xlabel('Predicted Wins')
            ax.set_ylabel('Predicted Shutouts')
            ax.set_title('Wins vs Shutouts - Top 15 Goalies')
            ax.grid(True, alpha=0.3)
            
            # GAA Distribution
            ax = axes[0, 1]
            ax.hist(goal_pred['predicted_gaa'], bins=20, edgecolor='black', alpha=0.7, color='orange')
            ax.set_xlabel('Goals Against Average')
            ax.set_ylabel('Number of Goalies')
            ax.set_title('GAA Distribution')
            if goal_pred['gaa_bonus'].any():
                best_gaa = goal_pred[goal_pred['gaa_bonus']]['predicted_gaa'].iloc[0]
                ax.axvline(best_gaa, color='red', linestyle='--', label=f'Best: {best_gaa:.3f}')
                ax.legend()
            
            # Save Percentage Distribution
            ax = axes[1, 0]
            ax.hist(goal_pred['predicted_save_pct'], bins=20, edgecolor='black', alpha=0.7, color='green')
            ax.set_xlabel('Save Percentage')
            ax.set_ylabel('Number of Goalies')
            ax.set_title('Save Percentage Distribution')
            if goal_pred['sv_pct_bonus'].any():
                best_sv = goal_pred[goal_pred['sv_pct_bonus']]['predicted_save_pct'].iloc[0]
                ax.axvline(best_sv, color='red', linestyle='--', label=f'Best: {best_sv:.3f}')
                ax.legend()
            
            # Pool Points vs Games Played (Goalies)
            ax = axes[1, 1]
            scatter = ax.scatter(top_goalies['projected_games'], top_goalies['pool_points'], 
                               c=top_goalies['predicted_save_pct'], cmap='viridis', s=100)
            ax.set_xlabel('Projected Games')
            ax.set_ylabel('Pool Points')
            ax.set_title('Goalie Performance Overview')
            ax.axvline(40, color='red', linestyle='--', alpha=0.5, label='40 GP Threshold')
            ax.legend()
            plt.colorbar(scatter, ax=ax, label='Save %')
            
            plt.tight_layout()
            plot_path = f'{self.plots_dir}/goalie_analysis_{scenario}_{self.timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()            
            print(f"Saved goalie analysis plot to: {plot_path}")

def main():
    print("NHL POOL RANKING SYSTEM v1.7")
    print("="*50)
    print("Enhanced Features:")
    print("  - Complete goalie statistics (shutouts, GAA, save %)")
    print("  - 2024-25 validation analysis")
    print("  - Feature importance analysis")
    print("  - Comprehensive outputs (CSV, TXT, plots)")
    print("="*50)
    
    predictor = UnifiedNHLPoolPredictor(lag_years=2)
    
    try:
        forwards_df, defensemen_df, goalies_df = predictor.load_and_prepare_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure data files exist in data_output/")
        return
    
    # Train models
    predictor.train_all_positions(forwards_df, defensemen_df, goalies_df)
    
    # Generate feature importance analysis
    print("\nGenerating feature importance analysis...")
    predictor.generate_feature_importance_report()
    
    # Validate 2024-25 season
    validation_df = predictor.validate_2024_predictions(forwards_df, defensemen_df, goalies_df)
    if len(validation_df) > 0:
        predictor.generate_validation_report(validation_df)
        predictor.generate_validation_plots(validation_df)
        
        print("\n2024-25 VALIDATION SUMMARY:")
        print("-"*50)
        print(f"Players Validated: {len(validation_df)}")
        print(f"Overall MAE: {validation_df['error'].abs().mean():.2f} points")
        print(f"Overall RMSE: {np.sqrt((validation_df['error'] ** 2).mean()):.2f} points")
        
        for position in ['Forward', 'Defense', 'Goalie']:
            pos_data = validation_df[validation_df['position'] == position]
            if len(pos_data) > 0:
                print(f"\n{position}:")
                print(f"  MAE: {pos_data['error'].abs().mean():.2f}")
                print(f"  RMSE: {np.sqrt((pos_data['error'] ** 2).mean()):.2f}")
    
    # Generate 2025-26 predictions
    fwd_2025, def_2025, goal_2025 = predictor.predict_2025_26_season(
        forwards_df, defensemen_df, goalies_df
    )
    
    # Generate rankings with full output
    if len(fwd_2025) > 0 or len(def_2025) > 0 or len(goal_2025) > 0:
        rankings = predictor.generate_rankings(fwd_2025, def_2025, goal_2025, "2025_26")
        
        print("\n" + "="*80)
        print("TOP 30 POOL RANKINGS FOR 2025-26 SEASON")
        print("="*80)
        print(f"{'Rank':<6} {'Player':<25} {'Position':<12} {'Team':<6} {'Points':>10}")
        print("-"*80)
        for _, row in rankings.head(30).iterrows():
            print(f"{row['overall_rank']:<6} {row['player_name'][:24]:<25} "
                  f"{row['position_group']:<12} {row.get('team_abbrev', 'N/A'):<6} "
                  f"{row['pool_points']:>10.1f}")
        
        print("\n" + "="*80)
        print("TOP GOALIES WITH DETAILED STATISTICS")
        print("="*80)
        top_goalies = goal_2025.nlargest(10, 'pool_points')
        for _, goalie in top_goalies.iterrows():
            print(f"\n{goalie['player_name']} ({goalie.get('team_abbrev', 'N/A')})")
            print(f"  Total Pool Points: {goalie['pool_points']:.1f}")
            print(f"  Projected Games: {goalie['projected_games']:.0f}")
            print(f"  Predicted Wins: {goalie.get('predicted_wins', 0):.1f}")
            print(f"  Predicted Shutouts: {goalie.get('predicted_shutouts', 0):.1f}")
            print(f"  Predicted GAA: {goalie.get('predicted_gaa', 0):.3f}")
            print(f"  Predicted Save %: {goalie.get('predicted_save_pct', 0):.3f}")
            if goalie.get('gaa_bonus', False):
                print("  *** GAA BONUS: +10 points ***")
            if goalie.get('sv_pct_bonus', False):
                print("  *** SAVE % BONUS: +10 points ***")
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE - TOP FEATURES BY MODEL")
        print("="*80)
        for model_key, importances in predictor.feature_importances.items():
            print(f"\n{model_key.upper()}:")
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            for rank, (feature, importance) in enumerate(sorted_features, 1):
                print(f"  {rank}. {feature[:30]:30} {importance:.4f}")
        
        print("\n" + "="*80)
        print("Analysis complete! Output files generated:")
        print("-"*80)
        print("Rankings & Predictions:")
        print(f"  - {predictor.results_dir}/rankings_2025_26_{predictor.timestamp}.csv")
        print(f"  - {predictor.results_dir}/report_2025_26_{predictor.timestamp}.txt")
        print("\n2024-25 Validation:")
        print(f"  - {predictor.results_dir}/validation_2024_{predictor.timestamp}.csv")
        print(f"  - {predictor.results_dir}/validation_2024_{predictor.timestamp}.txt")
        print("\nFeature Analysis:")
        print(f"  - {predictor.results_dir}/feature_importance_{predictor.timestamp}.csv")
        print(f"  - {predictor.results_dir}/feature_importance_{predictor.timestamp}.txt")
        print("\nVisualizations:")
        print(f"  - {predictor.plots_dir}/top_players_2025_26_{predictor.timestamp}.png")
        print(f"  - {predictor.plots_dir}/distribution_analysis_2025_26_{predictor.timestamp}.png")
        print(f"  - {predictor.plots_dir}/goalie_analysis_2025_26_{predictor.timestamp}.png")
        print(f"  - {predictor.plots_dir}/validation_scatter_{predictor.timestamp}.png")
        print(f"  - {predictor.plots_dir}/validation_errors_{predictor.timestamp}.png")
        print(f"  - {predictor.plots_dir}/feature_importance_{predictor.timestamp}.png")
        print("="*80)

if __name__ == "__main__":
    main()