import requests
import pandas as pd
import json
import time
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import os

class NHLAPIClient:
    def __init__(self, base_delay: float = 0.5):
        """
        Initialize NHL API client
        
        Args:
            base_delay: Delay between requests in seconds
        """
        # New NHL API endpoints
        self.new_api_base = "https://api-web.nhle.com"
        self.stats_api_base = "https://api.nhle.com/stats/rest"
        
        # Legacy NHL API (still works for some endpoints)
        self.legacy_api_base = "https://statsapi.web.nhl.com/api/v1"
        
        self.session = requests.Session()
        self.base_delay = base_delay
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'NHL-Stats-Collector/1.0',
            'Accept': 'application/json'
        })
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make API request with error handling and rate limiting
        """
        try:
            time.sleep(self.base_delay)
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error for {url}: {e}")
            return None
    
    def get_teams(self) -> pd.DataFrame:
        """
        Get all NHL teams
        """
        url = f"{self.new_api_base}/v1/teams"
        data = self._make_request(url)
        
        if not data:
            return pd.DataFrame()
        
        teams = []
        for team in data:
            teams.append({
                'team_id': team.get('id'),
                'team_name': team.get('name'),
                'team_abbrev': team.get('abbrev'),
                'team_common_name': team.get('commonName'),
                'team_location': team.get('locationName'),
                'division': team.get('division'),
                'conference': team.get('conference')
            })
        
        df = pd.DataFrame(teams)
        self.logger.info(f"Retrieved {len(df)} teams")
        return df
    
    def get_team_stats(self, season: str) -> pd.DataFrame:
        """
        Get team statistics for a season
        
        Args:
            season: Season in format "20232024" (2023-24 season)
        """
        url = f"{self.stats_api_base}/en/team/summary"
        params = {
            'season': season,
            'gameType': 2  # Regular season (2), Playoffs (3)
        }
        
        data = self._make_request(url, params)
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        teams_data = []
        for team in data['data']:
            teams_data.append({
                'season': season,
                'team_id': team.get('teamId'),
                'team_name': team.get('teamName'),
                'games_played': team.get('gamesPlayed'),
                'wins': team.get('wins'),
                'losses': team.get('losses'),
                'ot_losses': team.get('otLosses'),
                'points': team.get('points'),
                'goals_for': team.get('goalsFor'),
                'goals_against': team.get('goalsAgainst'),
                'goal_diff': team.get('goalDifferential'),
                'points_pct': team.get('pointPct'),
                'pp_pct': team.get('ppPct'),
                'pk_pct': team.get('pkPct'),
                'shots_for_per_game': team.get('shotsForPerGame'),
                'shots_against_per_game': team.get('shotsAgainstPerGame'),
                'face_off_win_pct': team.get('faceoffWinPct')
            })
        
        df = pd.DataFrame(teams_data)
        self.logger.info(f"Retrieved team stats for {len(df)} teams in season {season}")
        return df
    
    def get_player_stats(self, season: str, player_type: str = 'skaters') -> pd.DataFrame:
        """
        Get player statistics for a season
        
        Args:
            season: Season in format "20232024"
            player_type: 'skaters' or 'goalies'
        """
        if player_type == 'skaters':
            url = f"{self.stats_api_base}/en/skater/summary"
        else:
            url = f"{self.stats_api_base}/en/goalie/summary"
        
        params = {
            'season': season,
            'gameType': 2,  # Regular season
            'limit': 1000  # Get up to 1000 players
        }
        
        data = self._make_request(url, params)
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        players_data = []
        for player in data['data']:
            if player_type == 'skaters':
                players_data.append({
                    'season': season,
                    'player_id': player.get('playerId'),
                    'player_name': player.get('skaterFullName'),
                    'team_abbrev': player.get('teamAbbrevs'),
                    'position': player.get('positionCode'),
                    'games_played': player.get('gamesPlayed'),
                    'goals': player.get('goals'),
                    'assists': player.get('assists'),
                    'points': player.get('points'),
                    'plus_minus': player.get('plusMinus'),
                    'penalty_minutes': player.get('penaltyMinutes'),
                    'shots': player.get('shots'),
                    'shooting_pct': player.get('shootingPct'),
                    'time_on_ice_per_game': player.get('timeOnIcePerGame'),
                    'face_off_win_pct': player.get('faceoffWinPct'),
                    'pp_goals': player.get('ppGoals'),
                    'pp_points': player.get('ppPoints'),
                    'sh_goals': player.get('shGoals'),
                    'sh_points': player.get('shPoints'),
                    'gw_goals': player.get('gameWinningGoals'),
                    'hits': player.get('hits'),
                    'blocked_shots': player.get('blockedShots')
                })
            else:  # goalies
                players_data.append({
                    'season': season,
                    'player_id': player.get('playerId'),
                    'player_name': player.get('goalieFullName'),
                    'team_abbrev': player.get('teamAbbrevs'),
                    'games_played': player.get('gamesPlayed'),
                    'games_started': player.get('gamesStarted'),
                    'wins': player.get('wins'),
                    'losses': player.get('losses'),
                    'ot_losses': player.get('otLosses'),
                    'saves': player.get('saves'),
                    'shots_against': player.get('shotsAgainst'),
                    'save_pct': player.get('savePct'),
                    'goals_against': player.get('goalsAgainst'),
                    'goals_against_avg': player.get('goalsAgainstAverage'),
                    'time_on_ice': player.get('timeOnIce'),
                    'shutouts': player.get('shutouts'),
                    'goals': player.get('goals'),
                    'assists': player.get('assists'),
                    'points': player.get('points'),
                    'penalty_minutes': player.get('penaltyMinutes')
                })
        
        df = pd.DataFrame(players_data)
        self.logger.info(f"Retrieved {player_type} stats for {len(df)} players in season {season}")
        return df
    
    def get_game_results(self, season: str, team_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get game results for a season
        
        Args:
            season: Season in format "20232024"
            team_id: Optional team ID to filter by specific team
        """
        # Get schedule first
        url = f"{self.new_api_base}/v1/schedule/{season}"
        
        data = self._make_request(url)
        if not data or 'weeks' not in data:
            return pd.DataFrame()
        
        games_data = []
        for week in data['weeks']:
            for game in week.get('games', []):
                if team_id and game.get('homeTeam', {}).get('id') != team_id and game.get('awayTeam', {}).get('id') != team_id:
                    continue
                
                games_data.append({
                    'season': season,
                    'game_id': game.get('id'),
                    'game_date': game.get('gameDate'),
                    'game_type': game.get('gameType'),
                    'game_state': game.get('gameState'),
                    'home_team_id': game.get('homeTeam', {}).get('id'),
                    'home_team_abbrev': game.get('homeTeam', {}).get('abbrev'),
                    'away_team_id': game.get('awayTeam', {}).get('id'),
                    'away_team_abbrev': game.get('awayTeam', {}).get('abbrev'),
                    'home_score': game.get('homeTeam', {}).get('score'),
                    'away_score': game.get('awayTeam', {}).get('score'),
                    'venue': game.get('venue', {}).get('default')
                })
        
        df = pd.DataFrame(games_data)
        self.logger.info(f"Retrieved {len(df)} games for season {season}")
        return df
    
    def get_standings(self, season: str) -> pd.DataFrame:
        """
        Get standings for a season
        
        Args:
            season: Season in format "20232024"
        """
        url = f"{self.new_api_base}/v1/standings/{season}"
        
        data = self._make_request(url)
        if not data or 'standings' not in data:
            return pd.DataFrame()
        
        standings_data = []
        for standing in data['standings']:
            standings_data.append({
                'season': season,
                'team_id': standing.get('teamId'),
                'team_name': standing.get('teamName'),
                'team_abbrev': standing.get('teamAbbrev'),
                'conference_abbrev': standing.get('conferenceAbbrev'),
                'division_abbrev': standing.get('divisionAbbrev'),
                'games_played': standing.get('gamesPlayed'),
                'wins': standing.get('wins'),
                'losses': standing.get('losses'),
                'ot_losses': standing.get('otLosses'),
                'points': standing.get('points'),
                'point_pct': standing.get('pointPct'),
                'goals_for': standing.get('goalFor'),
                'goals_against': standing.get('goalAgainst'),
                'goal_diff': standing.get('goalDifferential'),
                'home_wins': standing.get('homeWins'),
                'home_losses': standing.get('homeLosses'),
                'home_ot_losses': standing.get('homeOtLosses'),
                'road_wins': standing.get('roadWins'),
                'road_losses': standing.get('roadLosses'),
                'road_ot_losses': standing.get('roadOtLosses'),
                'l10_wins': standing.get('l10Wins'),
                'l10_losses': standing.get('l10Losses'),
                'l10_ot_losses': standing.get('l10OtLosses'),
                'streak_code': standing.get('streakCode'),
                'streak_count': standing.get('streakCount')
            })
        
        df = pd.DataFrame(standings_data)
        self.logger.info(f"Retrieved standings for {len(df)} teams in season {season}")
        return df
    
    def get_multiple_seasons_data(self, seasons: List[str], include_games: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive data for multiple seasons
        
        Args:
            seasons: List of seasons in format ["20222023", "20232024"]
            include_games: Whether to include game-by-game data (can be large)
        
        Returns:
            Dictionary with DataFrames for different data types
        """
        results = {
            'teams': pd.DataFrame(),
            'team_stats': [],
            'skater_stats': [],
            'goalie_stats': [],
            'standings': []
        }
        
        if include_games:
            results['games'] = []
        
        # Get teams once (they don't change much)
        results['teams'] = self.get_teams()
        
        for season in seasons:
            self.logger.info(f"Downloading data for season {season}")
            
            # Team stats
            team_stats = self.get_team_stats(season)
            if not team_stats.empty:
                results['team_stats'].append(team_stats)
            
            # Player stats
            skater_stats = self.get_player_stats(season, 'skaters')
            if not skater_stats.empty:
                results['skater_stats'].append(skater_stats)
            
            goalie_stats = self.get_player_stats(season, 'goalies')
            if not goalie_stats.empty:
                results['goalie_stats'].append(goalie_stats)
            
            # Standings
            standings = self.get_standings(season)
            if not standings.empty:
                results['standings'].append(standings)
            
            # Games (optional)
            if include_games:
                games = self.get_game_results(season)
                if not games.empty:
                    results['games'].append(games)
        
        # Combine seasons
        final_results = {}
        for key, data_list in results.items():
            if key == 'teams':
                final_results[key] = data_list
            elif isinstance(data_list, list) and data_list:
                final_results[key] = pd.concat(data_list, ignore_index=True)
            else:
                final_results[key] = pd.DataFrame()
        
        return final_results
    
    def save_data_to_files(self, data: Dict[str, pd.DataFrame], output_dir: str = "nhl_data"):
        """
        Save data to CSV files
        
        Args:
            data: Dictionary of DataFrames
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for data_type, df in data.items():
            if not df.empty:
                filename = os.path.join(output_dir, f"nhl_{data_type}.csv")
                df.to_csv(filename, index=False)
                self.logger.info(f"Saved {len(df)} records to {filename}")
            else:
                self.logger.warning(f"No data to save for {data_type}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize client
    client = NHLAPIClient(base_delay=0.5)
    
    # Define seasons to download (format: start_year + end_year)
    seasons = ["20222023", "20232024"]  # 2022-23 and 2023-24 seasons
    
    print("=== NHL API Data Downloader ===")
    print(f"Downloading data for seasons: {seasons}")
    
    # Download comprehensive data
    data = client.get_multiple_seasons_data(
        seasons=seasons,
        include_games=True  # Set to False if you don't need game-by-game data
    )
    
    # Print summary
    print("\n=== Data Summary ===")
    for data_type, df in data.items():
        if not df.empty:
            print(f"{data_type}: {len(df)} records")
            if hasattr(df, 'columns'):
                print(f"  Columns: {list(df.columns)[:10]}...")  # First 10 columns
        else:
            print(f"{data_type}: No data")
    
    # Save to files
    client.save_data_to_files(data)
    
    print("\n=== Sample Data ===")
    # Show sample of each dataset
    for data_type, df in data.items():
        if not df.empty:
            print(f"\n{data_type.upper()} (first 3 rows):")
            print(df.head(3))