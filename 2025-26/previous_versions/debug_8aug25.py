# Date: August 8, 2025
# Program: ML model for points - DIAGNOSTIC VERSION
# Author: Julian di Giovanni w/Claude.AI
# Purpose: Diagnose exactly why we're losing all our data

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

def detailed_data_diagnosis(csv_path: str):
    """
    Comprehensive diagnosis of the data to understand why we're losing observations
    """
    print("=" * 80)
    print("COMPREHENSIVE DATA DIAGNOSIS")
    print("=" * 80)
    
    # Load raw data
    df = pd.read_csv(csv_path)
    print(f"\n1. RAW DATA LOADED")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Parse season
    df['season_str'] = df['season'].astype(str)
    df['year'] = df['season_str'].str[:4].astype(int)
    
    print(f"\n2. SEASON PARSING")
    print(f"   Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"   Unique years: {sorted(df['year'].unique())}")
    print(f"   Seasons per year:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"      {year}: {count} player-seasons")
    
    # Check key columns
    key_columns = ['player_id', 'year', 'goals', 'assists', 'points', 'games_played', 'shots']
    
    # Handle points column
    if 'points' not in df.columns:
        if 'goals' in df.columns and 'assists' in df.columns:
            df['points'] = df['goals'] + df['assists']
            print(f"\n   Created 'points' column from goals + assists")
        else:
            print(f"\n   ERROR: No 'points' column and can't create from goals/assists")
            return None
    
    print(f"\n3. KEY COLUMN ANALYSIS")
    for col in key_columns:
        if col in df.columns:
            missing = df[col].isnull().sum()
            print(f"   {col}: {missing} missing ({missing/len(df)*100:.1f}%)")
            if missing > 0:
                print(f"      Non-null range: {df[col].min()} to {df[col].max()}")
        else:
            print(f"   {col}: COLUMN MISSING!")
    
    # Player career analysis
    print(f"\n4. PLAYER CAREER ANALYSIS")
    player_careers = df.groupby('player_id').agg({
        'year': ['count', 'min', 'max'],
        'points': 'mean'
    }).round(1)
    
    career_lengths = player_careers[('year', 'count')]
    print(f"   Total players: {len(career_lengths)}")
    print(f"   Career length distribution:")
    for length in range(1, career_lengths.max() + 1):
        count = (career_lengths == length).sum()
        print(f"      {length} season{'s' if length > 1 else ''}: {count} players")
    
    # Players with enough history for lags
    print(f"\n5. LAG FEASIBILITY ANALYSIS")
    for lag_years in [1, 2, 3]:
        eligible_players = (career_lengths > lag_years).sum()
        total_eligible_seasons = df[df['player_id'].isin(
            career_lengths[career_lengths > lag_years].index
        )].shape[0]
        
        print(f"   Lag {lag_years}: {eligible_players} players eligible")
        print(f"           {total_eligible_seasons} total player-seasons from eligible players")
        
        # But we lose the first N years of each player's career
        estimated_usable = df[df['player_id'].isin(
            career_lengths[career_lengths > lag_years].index
        )].groupby('player_id')['year'].count().apply(lambda x: x - lag_years).sum()
        
        print(f"           ~{estimated_usable} seasons usable after lag creation")
    
    # Sample a few players to show career progression
    print(f"\n6. SAMPLE PLAYER CAREERS")
    sample_players = df.groupby('player_id')['year'].count().sort_values(ascending=False).head(5).index
    
    for player in sample_players:
        player_data = df[df['player_id'] == player].sort_values('year')
        years = player_data['year'].tolist()
        points = player_data['points'].tolist()
        print(f"   Player {player}: {len(years)} seasons")
        print(f"      Years: {years}")
        print(f"      Points: {points}")
    
    # Check for missing values in combinations
    print(f"\n7. MISSING VALUE COMBINATIONS")
    essential_cols = [col for col in ['goals', 'assists', 'points', 'games_played', 'shots'] if col in df.columns]
    
    print(f"   Essential columns: {essential_cols}")
    
    # Count rows with ANY missing essential data
    any_missing = df[essential_cols].isnull().any(axis=1).sum()
    print(f"   Rows with ANY missing essential data: {any_missing} ({any_missing/len(df)*100:.1f}%)")
    
    # Count rows with ALL essential data
    all_present = (~df[essential_cols].isnull().any(axis=1)).sum()
    print(f"   Rows with ALL essential data present: {all_present} ({all_present/len(df)*100:.1f}%)")
    
    return df

def test_lag_creation_step_by_step(df, lag_years=1):
    """
    Step by step lag creation to see exactly where we lose data
    """
    print(f"\n" + "=" * 80)
    print(f"STEP-BY-STEP LAG CREATION (LAG_YEARS = {lag_years})")
    print("=" * 80)
    
    df = df.sort_values(['player_id', 'year'])
    print(f"1. Starting data shape: {df.shape}")
    
    # Essential columns for lag creation
    lag_features = ['goals', 'assists', 'shots', 'games_played']
    existing_features = [f for f in lag_features if f in df.columns]
    print(f"2. Features to lag: {existing_features}")
    
    # Create lag columns
    df_with_lags = df.copy()
    lag_columns = []
    
    for feature in existing_features:
        for lag in range(1, lag_years + 1):
            lag_col = f'{feature}_lag{lag}'
            lag_columns.append(lag_col)
            df_with_lags[lag_col] = np.nan
    
    print(f"3. Created {len(lag_columns)} lag columns")
    
    # Process each player
    players_processed = 0
    observations_with_lags = 0
    
    for player in df['player_id'].unique():
        player_data = df[df['player_id'] == player].sort_values('year')
        
        if len(player_data) > lag_years:
            players_processed += 1
            
            for feature in existing_features:
                for lag in range(1, lag_years + 1):
                    lag_col = f'{feature}_lag{lag}'
                    lagged_values = player_data[feature].shift(lag)
                    df_with_lags.loc[df_with_lags['player_id'] == player, lag_col] = lagged_values
            
            # Count how many observations this player contributes
            player_complete = df_with_lags[df_with_lags['player_id'] == player][lag_columns].dropna()
            observations_with_lags += len(player_complete)
    
    print(f"4. Players with enough history: {players_processed}")
    
    # Check completeness
    complete_lag_data = ~df_with_lags[lag_columns].isnull().any(axis=1)
    complete_observations = complete_lag_data.sum()
    
    print(f"5. Observations with complete lag data: {complete_observations}")
    
    if complete_observations > 0:
        complete_years = df_with_lags[complete_lag_data]['year'].value_counts().sort_index()
        print(f"6. Year distribution of complete observations:")
        for year, count in complete_years.items():
            print(f"   {year}: {count} observations")
    else:
        print(f"6. NO COMPLETE OBSERVATIONS - investigating why...")
        
        # Debug: check a sample player
        sample_players = df.groupby('player_id').size().sort_values(ascending=False).head(3)
        print(f"   Checking sample players with most seasons:")
        
        for player_id, season_count in sample_players.items():
            print(f"\n   Player {player_id} ({season_count} seasons):")
            player_data = df[df['player_id'] == player_id].sort_values('year')
            player_lags = df_with_lags[df_with_lags['player_id'] == player_id].sort_values('year')
            
            print(f"   Years: {player_data['year'].tolist()}")
            print(f"   Goals: {player_data['goals'].tolist()}")
            
            for lag_col in lag_columns[:2]:  # Show first 2 lag columns
                values = player_lags[lag_col].tolist()
                print(f"   {lag_col}: {values}")
    
    return df_with_lags

def simple_model_test(df):
    """
    Try to build a very simple model with minimal requirements
    """
    print(f"\n" + "=" * 80)
    print(f"SIMPLE MODEL TEST (NO LAGS)")
    print("=" * 80)
    
    # Use current season features only - no lags
    feature_cols = []
    
    # Basic current season features
    if 'games_played' in df.columns:
        feature_cols.append('games_played')
    if 'shots' in df.columns:
        feature_cols.append('shots')
    if 'shots' in df.columns and 'games_played' in df.columns:
        df['shots_per_game'] = df['shots'] / df['games_played'].replace(0, np.nan)
        feature_cols.append('shots_per_game')
    
    # Handle shooting percentage
    shooting_cols = ['shooting_pct', 'shooting_percentage', 'sh_pct']
    shooting_col = None
    for col in shooting_cols:
        if col in df.columns:
            shooting_col = col
            break
    
    if shooting_col:
        feature_cols.append(shooting_col)
    elif 'goals' in df.columns and 'shots' in df.columns:
        df['shooting_pct'] = df['goals'] / df['shots'].replace(0, np.nan)
        feature_cols.append('shooting_pct')
    
    print(f"1. Using features: {feature_cols}")
    print(f"2. Target: points")
    
    if not feature_cols:
        print("ERROR: No usable features found!")
        return None
    
    # Prepare data
    X = df[feature_cols]
    y = df['points']
    
    # Remove missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"3. Clean data shape: {X_clean.shape}")
    
    if len(X_clean) < 20:
        print(f"ERROR: Only {len(X_clean)} clean observations")
        return None
    
    # Simple linear regression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    
    print(f"4. Simple model R²: {r2:.3f}")
    print(f"5. Feature coefficients:")
    for feature, coef in zip(feature_cols, model.coef_):
        print(f"   {feature}: {coef:.3f}")
    
    return model

def main():
    """
    Run comprehensive diagnosis
    """
    csv_path = '/Users/juliandigiovanni/Library/CloudStorage/Dropbox/hockeyanalytics/nhl_output/skater_team_data.csv'
    
    try:
        # Step 1: Detailed data diagnosis
        df = detailed_data_diagnosis(csv_path)
        
        if df is None:
            print("Could not load/process data")
            return
        
        # Step 2: Test lag creation for different lag lengths
        for lag_years in [1, 2, 3]:
            test_lag_creation_step_by_step(df, lag_years)
        
        # Step 3: Try a simple model without lags
        simple_model_test(df)
        
        print(f"\n" + "=" * 80)
        print("DIAGNOSIS COMPLETE")
        print("=" * 80)
        print("Check the output above to understand:")
        print("1. How many players have multiple seasons")
        print("2. Where exactly we lose data in lag creation")
        print("3. Whether a simple model (no lags) works")
        
    except Exception as e:
        print(f"Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()