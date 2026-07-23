""" 
NHL POOL RANKING SYSTEM - UNIFIED MODEL v1.6
=============================================
Complete working version with enhanced goalie statistics
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
        
        self._store_actual_2024_gp(forwards_df, defensemen_df, goalies_df)
        
        all_years = pd.concat([skaters_df['year'], goalies_df['year']])
        self.min_training_year = all_years.min() + self.lag_years
        
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
    
    def _store_actual_2024_gp(self, forwards_df, defensemen_df, goalies_df):
        for df in [forwards_df, defensemen_df, goalies_df]:
            data_2024 = df[df['year'] == 2024]
            for _, row in data_2024.iterrows():
                player_id = row['player_id']
                gp = row.get('games_played_player', row.get('games_played', 0))
                self.actual_2024_gp[player_id] = gp
    
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
        print(f"  {position} {target}: R² = {r2:.3f}")
        
        return {'r2': r2, 'model': model}
    
    def calculate_goalie_pool_points(self, results_df):
        if len(results_df) == 0:
            return results_df
        
        wins = results_df.get('predicted_wins', 0).fillna(0)
        shutouts = results_df.get('predicted_shutouts', 0).fillna(0)
        
        non_shutout_wins = np.maximum(wins - shutouts, 0)
        results_df['pool_points'] = non_shutout_wins + (shutouts * 3)
        
        gp_col = 'projected_games' if 'projected_games' in results_df.columns else 'games_played_player'
        qualified_mask = results_df[gp_col] >= self.min_goalie_games_for_bonus
        
        if 'predicted_gaa' in results_df.columns and qualified_mask.sum() > 0:
            qualified = results_df[qualified_mask]
            best_gaa_idx = qualified['predicted_gaa'].idxmin()
            if pd.notna(best_gaa_idx):
                results_df.loc[best_gaa_idx, 'pool_points'] += 10
                results_df.loc[best_gaa_idx, 'gaa_bonus'] = True
        
        if 'predicted_save_pct' in results_df.columns and qualified_mask.sum() > 0:
            qualified = results_df[qualified_mask]
            best_sv_idx = qualified['predicted_save_pct'].idxmax()
            if pd.notna(best_sv_idx):
                results_df.loc[best_sv_idx, 'pool_points'] += 10
                results_df.loc[best_sv_idx, 'sv_pct_bonus'] = True
        
        return results_df
    
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
        
        # Goalie predictions
        goal_results = pd.DataFrame()
        if len(goalies_2025) > 0:
            goal_results = goalies_2025[['player_id', 'player_name', 'team_abbrev']].copy()
            goal_results['projected_games'] = goal_results['player_id'].map(self.goalie_3yr_gp).fillna(20)
            
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
        
        # Train goalies
        goalies_lag = self.create_lag_features(goalies_df, 'goalie')
        
        for stat, target in [('wins', 'target_wins'), ('shutouts', 'target_shutouts'),
                            ('gaa', 'target_gaa'), ('save_pct', 'target_save_pct')]:
            X, y = self.prepare_features_for_modeling(goalies_lag, target, 'goalie')
            if len(X) > 0:
                results[f'goalie_{stat}'] = self.train_model(X, y, 'goalie', stat)
        
        return results
    
    def generate_rankings(self, fwd_pred, def_pred, goal_pred, scenario):
        # Add position column
        fwd_pred['position_group'] = 'Forward'
        def_pred['position_group'] = 'Defenseman'
        goal_pred['position_group'] = 'Goalie'
        
        # Combine all
        all_players = pd.concat([fwd_pred, def_pred, goal_pred], ignore_index=True)
        all_players['pool_points'] = all_players['pool_points'].fillna(0)
        all_players = all_players.sort_values('pool_points', ascending=False)
        all_players['overall_rank'] = range(1, len(all_players) + 1)
        
        # Save CSV
        csv_path = f'{self.results_dir}/rankings_{scenario}_{self.timestamp}.csv'
        all_players.to_csv(csv_path, index=False)
        print(f"Saved rankings to: {csv_path}")
        
        return all_players

def main():
    print("NHL POOL RANKING SYSTEM v1.6")
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
    
    # Generate 2025-26 predictions
    fwd_2025, def_2025, goal_2025 = predictor.predict_2025_26_season(
        forwards_df, defensemen_df, goalies_df
    )
    
    # Generate rankings
    if len(fwd_2025) > 0 or len(def_2025) > 0 or len(goal_2025) > 0:
        rankings = predictor.generate_rankings(fwd_2025, def_2025, goal_2025, "2025_26")
        
        print("\nTOP 20 POOL RANKINGS FOR 2025-26:")
        print("-"*50)
        for _, row in rankings.head(20).iterrows():
            print(f"{row['overall_rank']:3}. {row['player_name'][:20]:20} "
                  f"{row['position_group'][:3]:3} {row['pool_points']:6.1f}")
    
    print("\nComplete!")

if __name__ == "__main__":
    main()