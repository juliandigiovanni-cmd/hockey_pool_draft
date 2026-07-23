# Date: August 8, 2025
# Program: ML model for points regardless of position - FINAL VERSION (FIXED)
# Author: Julian di Giovanni w/Claude.AI
# Updates: Removes defensemen, handles missing data properly, optimized for real data

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
warnings.filterwarnings('ignore')

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
        
        # Remove completely empty team columns (team_2, team_3, etc.)
        team_cols = [col for col in df.columns if col.startswith('team_') and col != 'team_1']
        if team_cols:
            for col in team_cols:
                missing_pct = df[col].isnull().sum() / len(df) * 100
                if missing_pct > 95:  # Remove columns that are >95% missing
                    df = df.drop(columns=[col])
                    print(f"Removed column '{col}' ({missing_pct:.1f}% missing)")
        
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
        
        # Handle outliers in key statistics
        for stat in ['goals', 'assists', 'points', 'shots']:
            if stat in df.columns:
                # Cap at 99th percentile to handle data entry errors
                cap_value = df[stat].quantile(0.99)
                outliers = (df[stat] > cap_value).sum()
                if outliers > 0:
                    df[stat] = df[stat].clip(upper=cap_value)
                    print(f"Capped {outliers} outliers in '{stat}' at {cap_value}")
        
        print(f"Data shape after cleaning: {df.shape}")
        return df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features for previous seasons.
        Only creates lags where historical data exists.
        """
        lag_features = ['goals', 'assists', 'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game']
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
        
        # Current season features (safe divisions)
        if 'shots' in df_eng.columns and 'games_played' in df_eng.columns:
            df_eng['shots_per_game'] = df_eng['shots'] / df_eng['games_played'].replace(0, np.nan)
            df_eng['shots_per_game'] = df_eng['shots_per_game'].fillna(0)
        
        # Historical features from lag data
        lag_columns = [col for col in df_eng.columns if 'lag' in col]
        
        if lag_columns:
            print(f"Found {len(lag_columns)} lag columns")
            
            # Create historical averages
            lag_base_features = ['goals', 'assists', 'shots', 'games_played', 'shooting_pct', 'time_on_ice_per_game']
            
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
        Prepare feature matrix and target variable with robust missing data handling.
        """
        # Exclude current year goals/assists and meta columns
        exclude_cols = [
            'player_id', 'team_1', 'season', 'season_str', 'year', 
            'points', 'goals', 'assists', 'total_points',
            'position', 'pos', 'Position', 'Pos'  # Also exclude position columns
        ]
        
        # Exclude any columns that might contain player names or other text
        name_indicators = ['name', 'player', 'firstname', 'lastname', 'full_name', 'player_name']
        for col in df.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in name_indicators):
                exclude_cols.append(col)
                print(f"Excluding text column: '{col}'")
        
        # Also exclude any remaining team columns
        team_cols = [col for col in df.columns if col.startswith('team_')]
        exclude_cols.extend(team_cols)
        
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
        
        print(f"\nPreparing features:")
        print(f"  Total columns in data: {len(df.columns)}")
        print(f"  Excluded columns: {len(exclude_cols)}")
        print(f"  Feature columns: {len(self.feature_columns)}")
        
        if len(exclude_cols) < 20:  # Only show if not too many
            print(f"  Excluded: {exclude_cols}")
        
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
        
        print(f"  R² Score: {r2:.3f}")
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
    
    def plot_results(self, results: Dict[str, Any]):
        """Plot model results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(results['y_test'], results['y_pred'], alpha=0.6)
        min_val, max_val = results['y_test'].min(), results['y_test'].max()
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Points')
        axes[0, 0].set_ylabel('Predicted Points')
        axes[0, 0].set_title(f'Actual vs Predicted (Forwards Only)\nR² = {results["r2"]:.3f}')
        
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
        plt.show()

def main():
    """
    Main function to run the hockey points predictor
    """
    csv_path = '/Users/juliandigiovanni/Library/CloudStorage/Dropbox/hockeyanalytics/nhl_output/skater_team_data.csv'
    
    print("Hockey Points Predictor - Fixed Version (Forwards Only)")
    print("=" * 60)
    
    best_results = None
    best_config = None
    best_r2 = -1
    
    # Test different lag configurations
    configs = [
        {'lag_years': 1, 'min_training_year': None},
        {'lag_years': 2, 'min_training_year': None},
        {'lag_years': 3, 'min_training_year': None}
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"TESTING: {config['lag_years']}-year lags")
        print('='*60)
        
        try:
            # Initialize predictor
            predictor = HockeyPointsPredictor(**config)
            
            # Load and process data
            df = predictor.load_data(csv_path)
            df_with_lags = predictor.create_lag_features(df)
            df_engineered = predictor.engineer_features(df_with_lags)
            
            # Prepare features
            X, y = predictor.prepare_features(df_engineered)
            
            if len(X) < 50:
                print(f"Insufficient data: {len(X)} observations")
                continue
            
            # Test different models
            models = ['random_forest', 'gradient_boosting', 'linear_regression']
            
            for model_type in models:
                try:
                    results = predictor.train_model(X, y, model_type=model_type)
                    
                    if results['r2'] > best_r2:
                        best_r2 = results['r2']
                        best_results = results
                        best_config = {**config, 'model_type': model_type, 'predictor': predictor}
                        
                except Exception as e:
                    print(f"Error with {model_type}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error with configuration {config}: {e}")
            continue
    
    # Show best results
    if best_results:
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION FOUND (FORWARDS ONLY)")
        print('='*70)
        print(f"Lag years: {best_config['lag_years']}")
        print(f"Model: {best_config['model_type']}")
        print(f"R² Score: {best_results['r2']:.3f}")
        print(f"RMSE: {best_results['rmse']:.2f}")
        print(f"Training size: {best_results['n_train']}")
        print(f"Test size: {best_results['n_test']}")
        
        # Plot results
        best_config['predictor'].plot_results(best_results)
        
        # Show top features
        print(f"\nTop 10 Most Important Features:")
        print("-" * 40)
        try:
            importance_df = best_config['predictor'].get_feature_importance(10)
            for idx, row in importance_df.iterrows():
                print(f"{row['feature']:30}: {row['importance']:.4f}")
        except Exception as e:
            print(f"Could not display feature importance: {e}")
            
    else:
        print("\nNo working configuration found!")

if __name__ == "__main__":
    main()