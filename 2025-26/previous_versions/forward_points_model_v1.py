
# Date: August 6, 2025
# Program: ML model for points regardless of position. v1
# Author: Julian di Giovanni w/Claude.AI

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
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.target_column = 'points'  # Changed from 'total_points' to 'points'
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load hockey player statistics from CSV file.
        Expected columns: season, player_id, games_played, shots, shooting_pct, 
                         time_on_ice, goals, assists, points, team_1
        """
        df = pd.read_csv(csv_path)
        
        # Use existing points column or calculate if missing
        if 'points' not in df.columns:
            df['points'] = df['goals'] + df['assists']
        
        # Create total_points for internal consistency (same as points)
        df['total_points'] = df['points']
        
        # Ensure shooting percentage is in decimal form (0-1) not percentage (0-100)
        if df['shooting_pct'].max() > 1:
            df['shooting_pct'] = df['shooting_pct'] / 100
        
        # Sort by player and season for proper lag creation
        df = df.sort_values(['player_id', 'season'])
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lag_years: int = 3) -> pd.DataFrame:
        """
        Create lagged features for previous seasons' statistics.
        Creates lags for goals, assists, shots, games_played, shooting_pct, and time_on_ice.
        Current year goals/assists will be excluded from features for prediction.
        """
        # Create lags for all key variables
        lag_features = ['goals', 'assists', 'shots', 'games_played', 'shooting_pct', 'time_on_ice']
        
        # Create a copy of the dataframe
        df_with_lags = df.copy()
        
        # Create lag features for each player
        for player in df['player_id'].unique():
            player_data = df[df['player_id'] == player].copy()
            
            for feature in lag_features:
                for lag in range(1, lag_years + 1):
                    lag_col_name = f'{feature}_lag{lag}'
                    player_data[lag_col_name] = player_data[feature].shift(lag)
            
            # Update the main dataframe
            df_with_lags.loc[df_with_lags['player_id'] == player, 
                           [f'{f}_lag{l}' for f in lag_features for l in range(1, lag_years + 1)]] = \
                player_data[[f'{f}_lag{l}' for f in lag_features for l in range(1, lag_years + 1)]]
        
        return df_with_lags
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features.
        Uses current year variables except for goals/assists which use lag features.
        """
        df_eng = df.copy()
        
        # Current season features (using current year data)
        df_eng['shots_per_game'] = df_eng['shots'] / df_eng['games_played']
        df_eng['time_on_ice_per_game'] = df_eng['time_on_ice'] / df_eng['games_played']
        
        # Historical features from lag data
        lag_columns = [col for col in df_eng.columns if 'lag' in col]
        if lag_columns:
            # Create averages and trends for all lagged variables
            lag_base_features = ['goals', 'assists', 'shots', 'games_played', 'shooting_pct', 'time_on_ice']
            
            for base_feature in lag_base_features:
                lag_cols = [col for col in lag_columns if col.startswith(f'{base_feature}_lag')]
                
                if lag_cols:
                    # Average over previous seasons
                    df_eng[f'{base_feature}_avg_prev3'] = df_eng[lag_cols].mean(axis=1)
                    
                    # Trend (lag1 - lag2, positive means improvement)
                    if f'{base_feature}_lag1' in df_eng.columns and f'{base_feature}_lag2' in df_eng.columns:
                        df_eng[f'{base_feature}_trend'] = df_eng[f'{base_feature}_lag1'] - df_eng[f'{base_feature}_lag2']
            
            # Additional engineered features from lag data
            # Historical per-game rates
            for lag in range(1, 4):  # lag 1, 2, 3
                if f'goals_lag{lag}' in df_eng.columns and f'games_played_lag{lag}' in df_eng.columns:
                    df_eng[f'goals_per_game_lag{lag}'] = df_eng[f'goals_lag{lag}'] / df_eng[f'games_played_lag{lag}']
                    
                if f'assists_lag{lag}' in df_eng.columns and f'games_played_lag{lag}' in df_eng.columns:
                    df_eng[f'assists_per_game_lag{lag}'] = df_eng[f'assists_lag{lag}'] / df_eng[f'games_played_lag{lag}']
                    
                if f'shots_lag{lag}' in df_eng.columns and f'games_played_lag{lag}' in df_eng.columns:
                    df_eng[f'shots_per_game_lag{lag}'] = df_eng[f'shots_lag{lag}'] / df_eng[f'games_played_lag{lag}']
                    
                if f'time_on_ice_lag{lag}' in df_eng.columns and f'games_played_lag{lag}' in df_eng.columns:
                    df_eng[f'time_on_ice_per_game_lag{lag}'] = df_eng[f'time_on_ice_lag{lag}'] / df_eng[f'games_played_lag{lag}']
                
                # Historical points calculation
                if f'goals_lag{lag}' in df_eng.columns and f'assists_lag{lag}' in df_eng.columns:
                    df_eng[f'points_lag{lag}'] = df_eng[f'goals_lag{lag}'] + df_eng[f'assists_lag{lag}']
                    
                    if f'games_played_lag{lag}' in df_eng.columns:
                        df_eng[f'points_per_game_lag{lag}'] = df_eng[f'points_lag{lag}'] / df_eng[f'games_played_lag{lag}']
            
            # Average per-game stats from historical data
            per_game_lag_features = ['goals_per_game', 'assists_per_game', 'shots_per_game', 'time_on_ice_per_game', 'points_per_game']
            for base_feature in per_game_lag_features:
                lag_cols = [col for col in df_eng.columns if col.startswith(f'{base_feature}_lag')]
                if lag_cols:
                    df_eng[f'{base_feature}_avg_prev3'] = df_eng[lag_cols].mean(axis=1)
        
        # Replace infinite values with NaN
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        
        return df_eng
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target variable for modeling.
        Excludes current year goals and assists from features.
        """
        # Select feature columns (excluding target and current year goals/assists)
        exclude_cols = ['player_id', 'team_1', 'season', 'points', 'goals', 'assists']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train the machine learning model.
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select and train model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
        else:
            raise ValueError("model_type must be 'random_forest', 'gradient_boosting', or 'linear_regression'")
        
        # Train the model
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
        if model_type == 'linear_regression':
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        
        results = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'model_type': model_type
        }
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            # For linear regression, use absolute coefficients
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': np.abs(self.model.coef_)
            }).sort_values('importance', ascending=False)
            return importance_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.predict(X[self.feature_columns])
        else:
            X_scaled = self.scaler.transform(X[self.feature_columns])
            return self.model.predict(X_scaled)
    
    def plot_results(self, results: Dict[str, Any]):
        """
        Plot model results and diagnostics.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(results['y_test'], results['y_pred'], alpha=0.6)
        axes[0, 0].plot([results['y_test'].min(), results['y_test'].max()], 
                       [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Points')
        axes[0, 0].set_ylabel('Predicted Points')
        axes[0, 0].set_title(f'Actual vs Predicted Points\nR² = {results["r2"]:.3f}')
        
        # Residuals
        residuals = results['y_test'] - results['y_pred']
        axes[0, 1].scatter(results['y_pred'], residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Points')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        
        # Feature Importance (top 15)
        if hasattr(self, 'model') and self.model is not None:
            importance_df = self.get_feature_importance().head(15)
            axes[1, 0].barh(range(len(importance_df)), importance_df['importance'])
            axes[1, 0].set_yticks(range(len(importance_df)))
            axes[1, 0].set_yticklabels(importance_df['feature'])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top 15 Feature Importances')
        
        # Distribution of residuals
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Residuals')
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        print(f"\nModel Performance ({results['model_type']}):")
        print(f"Mean Absolute Error: {results['mae']:.2f}")
        print(f"Root Mean Square Error: {results['rmse']:.2f}")
        print(f"R² Score: {results['r2']:.3f}")
        print(f"Cross-validation R² (mean ± std): {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")

# Example usage
def main():
    """
    Example workflow for using the HockeyPointsPredictor
    """
    # Initialize predictor
    predictor = HockeyPointsPredictor()
    
    # Load data (replace with your CSV file path)
    df = predictor.load_data('/Users/juliandigiovanni/Library/CloudStorage/Dropbox/hockeyanalytics/nhl_output/skater_team_data.csv')
    
    # For demonstration, create sample data with correct column names
    np.random.seed(42)
    players = [f'Player_{i}' for i in range(50)]
    teams = ['Team_A', 'Team_B', 'Team_C', 'Team_D', 'Team_E']
    seasons = ['2020-21', '2021-22', '2022-23', '2023-24']
    
    data = []
    for player in players:
        base_skill = np.random.normal(0.12, 0.04)  # Base shooting percentage
        for season in seasons:
            games = np.random.randint(60, 83)
            shots = np.random.randint(100, 300)
            shooting_pct = max(0.05, min(0.25, base_skill + np.random.normal(0, 0.02)))
            goals = int(shots * shooting_pct)
            assists = int(goals * np.random.uniform(0.8, 1.5))
            points = goals + assists
            time_on_ice = np.random.randint(800, 1500)  # Total TOI in minutes
            
            data.append({
                'player_id': player,
                'team_1': np.random.choice(teams),
                'season': season,
                'games_played': games,
                'shots': shots,
                'shooting_pct': shooting_pct,
                'time_on_ice': time_on_ice,
                'goals': goals,
                'assists': assists,
                'points': points
            })
    
    df = pd.DataFrame(data)
    
    print("Sample of loaded data:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    
    # Create lag features
    print("\nCreating lag features for all variables (previous 3 years)...")
    df_with_lags = predictor.create_lag_features(df, lag_years=3)
    
    # Engineer additional features
    print("Engineering features...")
    df_engineered = predictor.engineer_features(df_with_lags)
    
    # Prepare features for modeling (remove rows without sufficient history)
    df_model = df_engineered.dropna()
    print(f"Data shape after removing rows without lag features: {df_model.shape}")
    
    if len(df_model) < 50:
        print("Warning: Limited data available for training after creating lag features")
        return
    
    # Prepare features and target
    X, y = predictor.prepare_features(df_model)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(predictor.feature_columns)}")
    
    # Train and evaluate different models
    models = ['random_forest', 'gradient_boosting', 'linear_regression']
    best_model = None
    best_score = -float('inf')
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.replace('_', ' ').title()} Model")
        print('='*50)
        
        results = predictor.train_model(X, y, model_type=model_type)
        
        if results['r2'] > best_score:
            best_score = results['r2']
            best_model = model_type
        
        # Plot results for the best model
        if model_type == 'random_forest':  # Show plots for one model
            predictor.plot_results(results)
    
    print(f"\nBest performing model: {best_model} (R² = {best_score:.3f})")
    
    # Retrain with best model
    if best_model != 'random_forest':
        predictor.train_model(X, y, model_type=best_model)
    
    # Show feature importance
    print(f"\nTop 10 Most Important Features:")
    importance_df = predictor.get_feature_importance().head(10)
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    main()