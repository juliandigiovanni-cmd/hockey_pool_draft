# Date: August 22, 2025
# Program: open and merge player and team data files
# Output:
#   1. data_output/skater_team_data.csv
#   2. data_output/goalie_team_data.csv
# Author: Julian di Giovanni w/Claude

"""
NHL Data Merger - Python Script
===============================

DESCRIPTION:
This script merges NHL player statistics (skaters and goalies) with team statistics and information.
It processes the data downloaded from the NHL API (2008-2025) and creates comprehensive datasets
that combine player performance with their team's performance for analysis.

MAIN PROCESSING STEPS:
1. Load NHL data files from the input directory:
   - nhl_skater_stats.csv (individual skater statistics by season)
   - nhl_goalie_stats.csv (individual goalie statistics by season)  
   - nhl_team_stats.csv (team performance statistics by season)
   - nhl_teams.csv (team information and codes)

2. Clean and prepare team data:
   - Merge team statistics with team abbreviation codes
   - Remove duplicate team-season combinations

3. Process multi-team players:
   - For players who played on multiple teams in a season (indicated by comma-separated 
     team abbreviations like "BOS,NYR"), extract only the FIRST team listed
   - This simplifies analysis by assigning each player-season to a single team

4. Merge player data with team data:
   - Join player statistics with team statistics and information based on:
     * Team abbreviation (first team only for multi-team players)
     * Season year
   - This creates comprehensive datasets with both player and team performance metrics

5. Standardize team identifiers:
   - For observations with the same franchise_id, assign the same team_id
   - This handles cases where franchises may have had different team_ids over time
   - Ensures consistent team identification across seasons

6. Save merged datasets:
   - skater_team_data.csv: Skaters with their team information and statistics
   - goalie_team_data.csv: Goalies with their team information and statistics

INPUT REQUIREMENTS:
- Input folder: "nhl_data_complete_2008_2025" (relative to script location)
- Required files in input folder:
  * nhl_skater_stats.csv
  * nhl_goalie_stats.csv  
  * nhl_team_stats.csv
  * nhl_teams.csv

OUTPUT:
- Output folder: "data_output" (created if doesn't exist)
- Generated files:
  * skater_team_data.csv: Complete skater dataset with team information
  * goalie_team_data.csv: Complete goalie dataset with team information

USAGE:
Simply run this script from any directory. The script will automatically:
- Find the input data folder relative to the script location
- Create the output folder if it doesn't exist
- Process all data and save results

TECHNICAL NOTES:
- Handles missing data gracefully with left joins
- Uses pandas for efficient data processing
- Generalizes file paths to work across different operating systems
- Processes large datasets efficiently with appropriate data types

Author: Converted from Jupyter notebook
Date: August 2025
"""

import pandas as pd
import os
import sys

def setup_directories():
    """
    Set up input and output directories relative to script location.
    This ensures the script works regardless of where it's run from.
    
    Expected folder structure:
    parent_folder/
    ├── nhl_data_merger.py (this script)
    ├── nhl_data_complete_2008_2025/
    └── data_output/ (created if doesn't exist)
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output directories in the same folder as the script
    input_dir = os.path.join(script_dir, 'nhl_data_complete_2008_2025')
    output_dir = os.path.join(script_dir, 'data_output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        print(f"Please ensure the 'nhl_data_complete_2008_2025' folder exists in the same directory as this script.")
        print(f"Expected structure:")
        print(f"  parent_folder/")
        print(f"  ├── nhl_data_merger.py (this script)")
        print(f"  ├── nhl_data_complete_2008_2025/")
        print(f"  └── data_output/ (created automatically)")
        sys.exit(1)
    
    return input_dir, output_dir

def load_nhl_data(input_dir):
    """
    Load all required NHL data files.
    
    Args:
        input_dir (str): Path to directory containing NHL data files
        
    Returns:
        tuple: (df_skater, df_goalie, df_team, df_tcode) DataFrames
    """
    print("Loading NHL data files...")
    
    # Define required files
    files_to_load = {
        'skater': 'nhl_skater_stats.csv',
        'goalie': 'nhl_goalie_stats.csv', 
        'team': 'nhl_team_stats.csv',
        'tcode': 'nhl_teams.csv'
    }
    
    data_frames = {}
    
    # Load each file and check if it exists
    for key, filename in files_to_load.items():
        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            print(f"ERROR: Required file not found: {file_path}")
            sys.exit(1)
        
        try:
            data_frames[key] = pd.read_csv(file_path)
            print(f"✓ Loaded {filename}: {len(data_frames[key]):,} rows")
        except Exception as e:
            print(f"ERROR loading {filename}: {e}")
            sys.exit(1)
    
    return data_frames['skater'], data_frames['goalie'], data_frames['team'], data_frames['tcode']

def prepare_team_data(df_team, df_tcode):
    """
    Prepare team data by merging team stats with team codes and removing duplicates.
    
    Args:
        df_team (DataFrame): Team statistics data
        df_tcode (DataFrame): Team codes and information
        
    Returns:
        DataFrame: Prepared team data with abbreviations and franchise info
    """
    print("Preparing team data...")
    
    # Merge team stats with team abbreviation codes AND franchise_id
    df_team_prepared = df_team.merge(
        df_tcode[['team_id', 'team_abbrev', 'franchise_id']], 
        on='team_id', 
        how='left'
    )
    
    # Remove duplicate team-season combinations
    initial_rows = len(df_team_prepared)
    df_team_prepared = df_team_prepared.drop_duplicates(subset=['team_abbrev', 'season'])
    final_rows = len(df_team_prepared)
    
    if initial_rows != final_rows:
        print(f"  Removed {initial_rows - final_rows} duplicate team-season combinations")
    
    # Check for franchise transitions (like Arizona -> Utah)
    franchise_transitions = df_team_prepared.groupby('franchise_id').agg({
        'team_id': lambda x: list(x.unique()),
        'team_abbrev': lambda x: list(x.unique()),
        'season': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    franchise_transitions.columns = ['franchise_id', 'team_ids', 'team_abbrevs', 'first_season', 'last_season']
    
    # Show franchises with multiple team_ids or abbreviations
    multi_team_franchises = franchise_transitions[
        (franchise_transitions['team_ids'].apply(len) > 1) | 
        (franchise_transitions['team_abbrevs'].apply(len) > 1)
    ]
    
    if not multi_team_franchises.empty:
        print("  Franchises with multiple team_ids/abbreviations:")
        for _, row in multi_team_franchises.iterrows():
            print(f"    Franchise {row['franchise_id']}: team_ids {row['team_ids']}, abbrevs {row['team_abbrevs']}")
    
    print(f"✓ Team data prepared: {len(df_team_prepared):,} unique team-seasons")
    return df_team_prepared

def extract_first_team(df_player):
    """
    Extract only the first team from the team_abbrev field for multi-team players.
    
    Args:
        df_player (DataFrame): Player data with potentially comma-separated teams
        
    Returns:
        DataFrame: Player data with only first team in team_abbrev
    """
    print("Processing multi-team players (keeping first team only)...")
    
    # Count multi-team players before processing
    multi_team_count = df_player['team_abbrev'].str.contains(',', na=False).sum()
    print(f"  Found {multi_team_count:,} player-seasons with multiple teams")
    
    # Extract first team only (everything before the first comma, or the whole string if no comma)
    df_player = df_player.copy()
    df_player['team_abbrev'] = df_player['team_abbrev'].str.split(',').str[0].str.strip()
    
    print(f"✓ Processed multi-team players: now using first team only")
    return df_player

def merge_player_team_data(df_player, df_team, player_type):
    """
    Merge player data with team data.
    
    Args:
        df_player (DataFrame): Player statistics data
        df_team (DataFrame): Team data with abbreviations
        player_type (str): Type of player ("skater" or "goalie")
        
    Returns:
        DataFrame: Merged player-team data
    """
    print(f"Merging {player_type} data with team data...")
    
    initial_rows = len(df_player)
    
    # Debug: Check for specific franchise before merge (like Utah/Arizona)
    utah_franchise_teams = df_team[df_team['franchise_id'] == 40.0] if 'franchise_id' in df_team.columns else pd.DataFrame()
    if not utah_franchise_teams.empty:
        print(f"  Utah franchise (40) team data available:")
        utah_summary = utah_franchise_teams.groupby(['team_abbrev', 'team_id']).agg({
            'season': ['min', 'max', 'count']
        }).reset_index()
        utah_summary.columns = ['team_abbrev', 'team_id', 'first_season', 'last_season', 'season_count']
        for _, row in utah_summary.iterrows():
            print(f"    {row['team_abbrev']} (team_id {row['team_id']}): {row['first_season']}-{row['last_season']} ({row['season_count']} seasons)")
    
    # Check player data for Utah franchise before merge
    utah_players_before = df_player[df_player['team_abbrev'].isin(['ARI', 'UTA'])] if 'team_abbrev' in df_player.columns else pd.DataFrame()
    if not utah_players_before.empty:
        utah_player_summary = utah_players_before.groupby('team_abbrev')['season'].agg(['min', 'max', 'count']).reset_index()
        print(f"  Utah/Arizona player data before merge:")
        for _, row in utah_player_summary.iterrows():
            print(f"    {row['team_abbrev']} players: {row['min']}-{row['max']} ({row['count']} player-seasons)")
    
    # Merge player data with team data
    df_merged = df_player.merge(
        df_team,
        left_on=['team_abbrev', 'season'],
        right_on=['team_abbrev', 'season'],
        how='left',
        suffixes=('_player', '_team')
    )
    
    final_rows = len(df_merged)
    
    # Check Utah data after merge
    utah_players_after = df_merged[df_merged['team_abbrev'].isin(['ARI', 'UTA'])] if 'team_abbrev' in df_merged.columns else pd.DataFrame()
    if not utah_players_after.empty:
        utah_after_summary = utah_players_after.groupby('team_abbrev')['season'].agg(['min', 'max', 'count']).reset_index()
        print(f"  Utah/Arizona player data after merge:")
        for _, row in utah_after_summary.iterrows():
            print(f"    {row['team_abbrev']} players: {row['min']}-{row['max']} ({row['count']} player-seasons)")
        
        # Check if any Utah players didn't get team_id
        utah_missing_team_id = utah_players_after['team_id'].isna().sum()
        if utah_missing_team_id > 0:
            print(f"    WARNING: {utah_missing_team_id} Utah/Arizona players missing team_id after merge")
    
    # Check for any data loss
    if final_rows != initial_rows:
        print(f"  WARNING: Row count changed from {initial_rows:,} to {final_rows:,}")
    
    # Check merge success rate
    successful_merges = df_merged['team_id'].notna().sum()
    merge_rate = (successful_merges / final_rows) * 100
    print(f"  Merge success rate: {merge_rate:.1f}% ({successful_merges:,}/{final_rows:,})")
    
    if merge_rate < 90:
        print(f"  WARNING: Low merge rate. Check team abbreviations and seasons.")
        
        # Show some failed merges for debugging
        failed_merges = df_merged[df_merged['team_id'].isna()]
        if not failed_merges.empty:
            print("  Sample failed merges:")
            sample_fails = failed_merges[['team_abbrev', 'season']].drop_duplicates().head(5)
            for _, row in sample_fails.iterrows():
                print(f"    {row['team_abbrev']} - {row['season']}")
    
    return df_merged

def standardize_team_ids_after_merge(df_merged, df_tcode):
    """
    After merging, reassign team_ids so that all observations with the same franchise_id 
    get the same team_id. This handles cases like Utah (franchise_id 40) which had 
    team_ids 59 and 68 in different seasons.
    
    Args:
        df_merged (DataFrame): Merged player-team data
        df_tcode (DataFrame): Team codes with franchise information from nhl_teams.csv
        
    Returns:
        DataFrame: Data with standardized team_ids based on franchise_id
    """
    print("Standardizing team_ids after merge based on franchise_id...")
    
    # First, merge with nhl_teams.csv to get franchise_id information
    df_with_franchise = df_merged.merge(
        df_tcode[['team_id', 'franchise_id']],
        on='team_id',
        how='left',
        suffixes=('', '_tcode')
    )
    
    # Check merge success
    franchise_missing = df_with_franchise['franchise_id'].isna().sum()
    if franchise_missing > 0:
        print(f"  WARNING: {franchise_missing:,} records missing franchise_id information")
    
    # Create franchise to team_id mapping from nhl_teams.csv
    # For each franchise, use the most common team_id (or the latest one)
    franchise_team_mapping = df_tcode.groupby('franchise_id')['team_id'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[-1]  # Use last one if tie
    ).to_dict()
    
    # Show the mapping for debugging
    print("  Franchise to team_id mapping:")
    for franchise_id, team_id in sorted(franchise_team_mapping.items()):
        # Show if this franchise has multiple team_ids
        team_ids_for_franchise = df_tcode[df_tcode['franchise_id'] == franchise_id]['team_id'].unique()
        if len(team_ids_for_franchise) > 1:
            print(f"    Franchise {franchise_id}: {list(team_ids_for_franchise)} → {team_id} (STANDARDIZED)")
    
    # Apply the standardization
    df_standardized = df_with_franchise.copy()
    df_standardized['team_id_original'] = df_standardized['team_id']  # Keep original for reference
    
    # Update team_id based on franchise_id mapping
    df_standardized['team_id'] = df_standardized['franchise_id'].map(franchise_team_mapping).fillna(df_standardized['team_id'])
    
    # Count changes made
    changes_made = (df_standardized['team_id_original'] != df_standardized['team_id']).sum()
    
    print(f"  Standardized {changes_made:,} team_id values based on franchise_id")
    print(f"✓ Team ID standardization complete")
    
    return df_standardized

def create_synthetic_team_ids(df_merged):
    """
    Create synthetic team_ids for franchises that have multiple team_ids.
    Replace original team_ids with synthetic ones for consistency.
    
    Args:
        df_merged (DataFrame): Merged player-team data with franchise_id
        
    Returns:
        DataFrame: Data with synthetic team_ids
    """
    print("Creating synthetic team_ids for multi-team franchises...")
    
    # Check if franchise_id column exists
    if 'franchise_id' not in df_merged.columns:
        print("  WARNING: No franchise_id column found. Skipping synthetic team_id creation.")
        return df_merged
    
    # Identify franchises with multiple team_ids
    franchise_teams = df_merged.groupby('franchise_id')['team_id'].nunique().reset_index()
    multi_team_franchises = franchise_teams[franchise_teams['team_id'] > 1]['franchise_id'].tolist()
    
    if not multi_team_franchises:
        print("  No multi-team franchises found. No synthetic team_ids needed.")
        return df_merged
    
    print(f"  Found {len(multi_team_franchises)} franchises with multiple team_ids:")
    
    # Create synthetic team_id mapping
    df_result = df_merged.copy()
    df_result['team_id_original'] = df_result['team_id']  # Keep original
    
    # Create synthetic team_ids starting from a high number to avoid conflicts
    synthetic_id_counter = 1000
    
    for franchise_id in multi_team_franchises:
        # Skip if franchise_id is NaN
        if pd.isna(franchise_id):
            continue
            
        # Get all team_ids for this franchise
        franchise_data = df_merged[df_merged['franchise_id'] == franchise_id]
        original_team_ids = franchise_data['team_id'].unique()
        
        # Create synthetic team_id for this franchise
        synthetic_team_id = synthetic_id_counter
        synthetic_id_counter += 1
        
        # Show the mapping
        print(f"    Franchise {franchise_id}: {list(original_team_ids)} → {synthetic_team_id}")
        
        # Apply the synthetic team_id
        mask = df_result['franchise_id'] == franchise_id
        df_result.loc[mask, 'team_id'] = synthetic_team_id
    
    # Count changes made
    changes_made = (df_result['team_id'] != df_result['team_id_original']).sum()
    print(f"  Created synthetic team_ids for {changes_made:,} observations")
    print("✓ Synthetic team_id creation complete")
    
    return df_result

def save_merged_data(df_skater_merged, df_goalie_merged, output_dir):
    """
    Save the merged datasets to CSV files.
    
    Args:
        df_skater_merged (DataFrame): Merged skater-team data
        df_goalie_merged (DataFrame): Merged goalie-team data
        output_dir (str): Output directory path
    """
    print("Saving merged datasets...")
    
    # Define output files
    output_files = {
        'skater_team_data.csv': df_skater_merged,
        'goalie_team_data.csv': df_goalie_merged
    }
    
    # Save each dataset
    for filename, df in output_files.items():
        output_path = os.path.join(output_dir, filename)
        try:
            df.to_csv(output_path, index=False)
            print(f"✓ Saved {filename}: {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"ERROR saving {filename}: {e}")
            sys.exit(1)
    
    print(f"All files saved to: {output_dir}")

def main():
    """
    Main function to orchestrate the entire data merging process.
    """
    print("=" * 60)
    print("NHL DATA MERGER - Starting Process")
    print("=" * 60)
    
    try:
        # 1. Setup directories
        input_dir, output_dir = setup_directories()
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print()
        
        # 2. Load data
        df_skater, df_goalie, df_team, df_tcode = load_nhl_data(input_dir)
        print()
        
        # 3. Prepare team data
        df_team_prepared = prepare_team_data(df_team, df_tcode)
        print()
        
        # 4. Process skater data
        print("PROCESSING SKATER DATA:")
        df_skater_processed = extract_first_team(df_skater)
        df_skater_merged = merge_player_team_data(df_skater_processed, df_team_prepared, "skater")
        df_skater_final = create_synthetic_team_ids(df_skater_merged)
        print()
        
        # 5. Process goalie data  
        print("PROCESSING GOALIE DATA:")
        df_goalie_processed = extract_first_team(df_goalie)
        df_goalie_merged = merge_player_team_data(df_goalie_processed, df_team_prepared, "goalie")
        df_goalie_final = create_synthetic_team_ids(df_goalie_merged)
        print()
        
        # 6. Save results
        save_merged_data(df_skater_final, df_goalie_final, output_dir)
        print()
        
        # 7. Summary
        print("=" * 60)
        print("MERGE PROCESS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Final datasets:")
        print(f"  • Skaters: {len(df_skater_final):,} player-seasons")
        print(f"  • Goalies: {len(df_goalie_final):,} player-seasons")
        print(f"  • Columns in skater data: {len(df_skater_final.columns)}")
        print(f"  • Columns in goalie data: {len(df_goalie_final.columns)}")
        print()
        print(f"Output files saved to: {output_dir}")
        print(f"  • skater_team_data.csv")
        print(f"  • goalie_team_data.csv")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print("Process terminated.")
        sys.exit(1)

if __name__ == "__main__":
    main()