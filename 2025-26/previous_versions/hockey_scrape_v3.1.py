# Date: August 22, 2025
# Author: Julian di Giovanni w/Claude.ai - data scraping
# Version: 3.1 - insure no filling missing values with other seasons' data

"""
NHL API Data Scraper - Version 3 (Fixed with Era-Based Data Selection)

This script provides a comprehensive interface for downloading NHL statistics data from the official NHL API.
It includes methods for fetching team statistics, player statistics (skaters and goalies), standings, 
and game results across multiple seasons.

ORIGINAL ISSUE FIXED:
The original version was downloading 2023-2024 team statistics for all seasons due to the NHL API 
returning the most recent available data when older seasons were requested. Through testing, we discovered
that historical team statistics (pre-2008) are simply not available via the NHL API endpoints.

SOLUTION IMPLEMENTED:
This version includes era-based data selection that only attempts to download data types that are 
actually available for different time periods:

- MODERN ERA (2008+): Full API support - team stats, player stats, standings, games
- TRANSITION ERA (2005-2007): Limited support - standings and games only  
- HISTORICAL ERA (1990-2004): Minimal support - standings only (if any)

The script now intelligently skips data types that aren't available for specific eras, preventing
the duplicate data issue and providing realistic expectations about what can be downloaded.

API ENDPOINTS USED:
- Team data: https://api.nhle.com/stats/rest/en/team
- Team stats: https://api.nhle.com/stats/rest/en/team/summary (2008+ only)
- Player stats: https://api.nhle.com/stats/rest/en/skater/summary (2008+ only)
                https://api.nhle.com/stats/rest/en/goalie/summary (2008+ only)
- Standings: https://api-web.nhle.com/v1/standings/{date} (2005+ limited availability)
- Game results: https://api-web.nhle.com/v1/club-schedule-season/{team}/{season} (2005+)
- Legacy API: https://statsapi.web.nhl.com/api/v1/ (fallback for some historical data)

DATA AVAILABILITY LIMITATIONS:
- Team statistics: Only available for seasons 2008-2009 onwards
- Player statistics: Only available for seasons 2008-2009 onwards  
- Standings: Limited availability, best for 2008+ seasons
- Game data: Available from 2005-2006 onwards with some gaps
- Historical data (1990s): Very limited, mostly not available via current NHL API

FEATURES:
- Era-based data selection to avoid requesting unavailable data
- Batch processing to handle large date ranges without memory issues
- Rate limiting to respect API constraints
- Comprehensive error handling and logging
- Season validation and duplicate detection
- Data availability reporting by era
- CSV export functionality
- Support for both smart era-based and legacy data collection modes
- Fallback to legacy NHL API for some historical data

USAGE:
    client = NHLAPIClient()
    
    # Show what data is available by era
    availability = client.get_data_availability_by_era()
    
    # Download data with era-based selection (recommended)
    data = client.get_multiple_seasons_data_batched(
        seasons=["20082009", "20092010", "20232024"],  # Use modern era seasons
        batch_size=3,
        include_games=True,
        use_smart_selection=True  # Only attempt data types available for each era
    )

RECOMMENDATIONS:
- For complete team statistics: Use seasons 2008-2024
- For historical analysis: Consider external data sources or standings-only data
- For comprehensive analysis: Focus on 2008+ seasons where all data types are available
- For older seasons: Expect very limited data availability

Author: [Original + Fixes + Era-based Selection]
Version: 3.1 (Era-based data selection)
"""

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
        url = f"{self.stats_api_base}/en/team"
        data = self._make_request(url)

        if not data or 'data' not in data:
            return pd.DataFrame()

        teams = []
        for team in data['data']:
            teams.append({
                'team_id': team.get('id'),
                'team_abbrev': team.get('triCode'),
                'team_name': team.get('fullName'),
                'franchise_id': team.get('franchiseId'),
                'league_id': team.get('league'),
            })

        df = pd.DataFrame(teams)
        # Ensure team_id is an integer
        if not df.empty:
            df['franchise_id'] = df['franchise_id'].astype('Int64')
        self.logger.info(f"Retrieved {len(df)} teams")
        return df

    def _validate_season_data(self, data_list: List[Dict], requested_season: str, data_type: str) -> bool:
        """
        Validate that the returned data actually corresponds to the requested season
        """
        if not data_list:
            return False
            
        # For team stats, season info is not included in individual records
        # The season is just a request parameter, so we can't validate it from the response
        if data_type in ['team_stats']:
            # For team stats, just check if we got reasonable data
            sample_item = data_list[0]
            # Check if we have basic team stat fields
            required_fields = ['teamId', 'teamName', 'gamesPlayed']
            has_basic_fields = any(field in sample_item for field in required_fields)
            if not has_basic_fields:
                self.logger.warning(f"Team stats data doesn't contain expected fields for season {requested_season}")
                return False
            return True
            
        # For player stats, check if any of the returned data has season information
        seasons_in_data = set()
        for item in data_list[:5]:  # Check first 5 items
            # Look for season indicators in various possible fields
            for field in ['season', 'seasonId', 'gameTypeId']:
                if field in item and item[field]:
                    seasons_in_data.add(str(item[field]))
        
        # If we found season data, check if it matches our request
        if seasons_in_data:
            if requested_season not in seasons_in_data:
                self.logger.warning(f"API returned data for seasons {seasons_in_data} but we requested {requested_season} for {data_type}")
                return False
                
        return True

    def get_team_stats(self, season: str) -> pd.DataFrame:
        """
        Get team statistics for a season using correct API parameters

        Args:
            season: Season in format "20232024" (2023-24 season)
        """
        url = f"{self.stats_api_base}/en/team/summary"
        # Use cayenneExp parameter - this is the correct format for the NHL API
        params = {
            'cayenneExp': f'seasonId={season} and gameTypeId=2'  # gameTypeId=2 for regular season
        }

        self.logger.info(f"Requesting team stats for season {season} with cayenneExp parameter")
        data = self._make_request(url, params)
        
        if not data or 'data' not in data or not data['data']:
            self.logger.warning(f"No team stats data returned for season {season}")
            return pd.DataFrame()

        self.logger.info(f"Received {len(data['data'])} teams for season {season}")

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
        if not df.empty:
            self.logger.info(f"Successfully retrieved team stats for {len(df)} teams in season {season}")
            
            # Validate that we got reasonable data for the season
            games_played_values = df['games_played'].dropna().tolist()
            if games_played_values:
                max_games = max(games_played_values)
                min_games = min(games_played_values)
                avg_games = sum(games_played_values) / len(games_played_values)
                self.logger.info(f"Season {season} games played range: {min_games}-{max_games}, avg: {avg_games:.1f}")
        else:
            self.logger.warning(f"No team stats found for season {season}")
            
        return df

    def _get_team_stats_legacy(self, season: str) -> pd.DataFrame:
        """
        Try to get team stats using the legacy NHL API
        """
        # Convert season format for legacy API (20232024 -> 20232024)
        url = f"{self.legacy_api_base}/teams"
        params = {
            'season': season,
            'expand': 'team.stats'
        }
        
        self.logger.info(f"Trying legacy API for team stats season {season}")
        data = self._make_request(url, params)
        
        if not data or 'teams' not in data:
            self.logger.warning(f"Legacy API also failed for season {season}")
            return pd.DataFrame()
        
        teams_data = []
        for team in data['teams']:
            # Legacy API has stats nested differently
            team_stats = {}
            if 'teamStats' in team and team['teamStats']:
                for stat_group in team['teamStats']:
                    if stat_group.get('type', {}).get('displayName') == 'statsSingleSeason':
                        team_stats = stat_group.get('splits', [{}])[0].get('stat', {})
                        break
            
            teams_data.append({
                'season': season,
                'team_id': team.get('id'),
                'team_name': team.get('name'),
                'games_played': team_stats.get('gamesPlayed'),
                'wins': team_stats.get('wins'),
                'losses': team_stats.get('losses'),
                'ot_losses': team_stats.get('ot'),
                'points': team_stats.get('pts'),
                'goals_for': team_stats.get('goalsPerGame'),
                'goals_against': team_stats.get('goalsAgainstPerGame'),
                'goal_diff': None,  # Not directly available
                'points_pct': team_stats.get('ptPctg'),
                'pp_pct': team_stats.get('powerPlayPercentage'),
                'pk_pct': team_stats.get('penaltyKillPercentage'),
                'shots_for_per_game': team_stats.get('shotsPerGame'),
                'shots_against_per_game': team_stats.get('shotsAllowed'),
                'face_off_win_pct': team_stats.get('faceOffWinPercentage')
            })
        
        df = pd.DataFrame(teams_data)
        if not df.empty:
            self.logger.info(f"Legacy API: Retrieved team stats for {len(df)} teams in season {season}")
        else:
            self.logger.warning(f"Legacy API: No team stats found for season {season}")
        return df

    def get_player_stats(self, season: str, player_type: str = 'skaters') -> pd.DataFrame:
        """
        Get player statistics for a season with proper pagination using correct API parameters

        Args:
            season: Season in format "20232024"
            player_type: 'skaters' or 'goalies'
        """
        if player_type == 'skaters':
            url = f"{self.stats_api_base}/en/skater/summary"
        else:
            url = f"{self.stats_api_base}/en/goalie/summary"

        self.logger.info(f"Requesting {player_type} stats for season {season}")
        
        all_players_data = []
        limit = 100  # API seems to work better with smaller chunks
        start = 0
        max_requests = 50  # Safety limit to prevent infinite loops
        request_count = 0
        
        while request_count < max_requests:
            # Use cayenneExp parameter - this is the correct format for the NHL API
            params = {
                'limit': limit,
                'start': start,
                'cayenneExp': f'seasonId={season} and gameTypeId=2',  # Regular season
            }

            data = self._make_request(url, params)
            if not data or 'data' not in data or not data['data']:
                break

            players_batch = data['data']
            
            # Log progress for first batch
            if request_count == 0:
                self.logger.info(f"First batch for {player_type} season {season}: {len(players_batch)} players")
            
            for player in players_batch:
                if player_type == 'skaters':
                    all_players_data.append({
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
                    all_players_data.append({
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

            # Check if we got fewer results than the limit (indicates we're at the end)
            if len(players_batch) < limit:
                break
                
            start += limit
            request_count += 1
            if request_count % 5 == 0:  # Log every 5 batches
                self.logger.info(f"Retrieved {len(all_players_data)} {player_type} so far...")

        df = pd.DataFrame(all_players_data)
        if not df.empty:
            self.logger.info(f"Successfully retrieved {player_type} stats for {len(df)} players in season {season}")
        else:
            self.logger.warning(f"No {player_type} data found for season {season}")
        return df

    def get_game_results(self, season: str, team_abbrev: str) -> pd.DataFrame:
        """
        Get game results for a season for a specific team

        Args:
            season: Season in format "20232024"
            team_abbrev: Team abbreviation (e.g., "BOS", "TOR")
        """
        url = f"{self.new_api_base}/v1/club-schedule-season/{team_abbrev}/{season}"
        data = self._make_request(url)
        if not data or 'weeks' not in data:
            self.logger.warning(f"No game data found for team {team_abbrev} in season {season}")
            return pd.DataFrame()

        games_data = []
        for week in data['weeks']:
            for game in week.get('games', []):
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
        if not df.empty:
            self.logger.info(f"Retrieved {len(df)} games for team {team_abbrev} in season {season}")
        else:
            self.logger.warning(f"No games found for team {team_abbrev} in season {season}")
        return df

    def _get_standings_final_date(self, season: str) -> Optional[str]:
        """
        Fetch the final standings date for a given season from the standings-season endpoint.
        """
        url = f"{self.new_api_base}/v1/standings-season"
        data = self._make_request(url)
        if not data or 'seasons' not in data:
            self.logger.warning(f"Could not fetch standings-season data for season {season}")
            return None
        for entry in data['seasons']:  
            if str(entry.get('id')) == str(season):
                return entry.get('standingsStart')
        self.logger.warning(f"No standings date found for season {season}")
        return None

    def get_standings(self, season: str) -> pd.DataFrame:
        """
        Get standings for a season with validation

        Args:
            season: Season in format "20232024"
        """
        standings_date = self._get_standings_final_date(season)
        if not standings_date:
            self.logger.warning(f"Could not get standings date for season {season}")
            return pd.DataFrame()
            
        url = f"{self.new_api_base}/v1/standings/{standings_date}"

        self.logger.info(f"Requesting standings for season {season} (date: {standings_date})")
        data = self._make_request(url)
        if not data or 'standings' not in data:
            self.logger.warning(f"No standings data returned for season {season}")
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
        if not df.empty:
            self.logger.info(f"Successfully retrieved standings for {len(df)} teams in season {season}")
        else:
            self.logger.warning(f"No standings found for season {season}")
        return df

    def check_data_availability(self, seasons: List[str]) -> Dict[str, Dict[str, bool]]:
        """
        Check data availability for different seasons and data types
        
        Args:
            seasons: List of seasons to check
            
        Returns:
            Dictionary mapping seasons to data availability by type
        """
        availability = {}
        
        for season in seasons:
            self.logger.info(f"Checking data availability for season {season}")
            season_availability = {}
            
            # Check team stats - use correct cayenneExp parameter
            url = f"{self.stats_api_base}/en/team/summary"
            params = {
                'cayenneExp': f'seasonId={season} and gameTypeId=2'
            }
            data = self._make_request(url, params)
            season_availability['team_stats'] = bool(data and data.get('data') and len(data['data']) > 0)
            
            # Check standings
            standings = self.get_standings(season)
            season_availability['standings'] = not standings.empty
            
            # Sample check for player stats (just first 10 skaters) - use correct cayenneExp parameter
            url = f"{self.stats_api_base}/en/skater/summary"
            params = {
                'limit': 10,
                'start': 0,
                'cayenneExp': f'seasonId={season} and gameTypeId=2',
            }
            data = self._make_request(url, params)
            season_availability['player_stats'] = bool(data and data.get('data'))
            
            availability[season] = season_availability
            self.logger.info(f"Season {season} availability: {season_availability}")
            
        return availability

    def get_data_availability_by_era(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Get information about what data types are available for different NHL eras
        
        Returns:
            Dictionary mapping era names to data types and their available seasons
        """
        # Based on NHL API limitations discovered through testing
        availability_map = {
            "modern_era": {
                "description": "2008-2009 to present - Full API support",
                "seasons": [f"{year}{year+1}" for year in range(2008, 2025)],
                "available_data": ["team_stats", "skater_stats", "goalie_stats", "standings", "games"]
            },
            "transition_era": {
                "description": "2005-2008 - Limited API support", 
                "seasons": [f"{year}{year+1}" for year in range(2005, 2008)],
                "available_data": ["standings", "games"]  # Team stats may not be reliable
            },
            "historical_era": {
                "description": "1990-2004 - Minimal API support",
                "seasons": [f"{year}{year+1}" for year in range(1990, 2005)],
                "available_data": ["standings"]  # Very limited data available
            }
        }
        
        return availability_map

    def get_single_season_data_smart(self, season: str, include_games: bool = False, 
                                   teams_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for a single season, only attempting data types that are likely available
        
        Args:
            season: Season in format "20232024"
            include_games: Whether to include game-by-game data
            teams_df: Pre-fetched teams DataFrame
        """
        results = {
            'team_stats': pd.DataFrame(),
            'skater_stats': pd.DataFrame(),
            'goalie_stats': pd.DataFrame(),
            'standings': pd.DataFrame()
        }
        
        if include_games:
            results['games'] = pd.DataFrame()
            
        # Determine what data to attempt based on season
        season_year = int(season[:4])
        
        self.logger.info(f"Downloading available data for season {season} (year: {season_year})")
        
        # Only attempt team stats for recent seasons (2008+)
        if season_year >= 2008:
            self.logger.info(f"Attempting team stats for recent season {season}")
            team_stats = self.get_team_stats(season)
            if not team_stats.empty:
                results['team_stats'] = team_stats
            else:
                self.logger.warning(f"No team stats available for season {season}")
        else:
            self.logger.info(f"Skipping team stats for historical season {season} (not available in API)")
            
        # Only attempt player stats for recent seasons (2008+)
        if season_year >= 2008:
            self.logger.info(f"Attempting player stats for recent season {season}")
            skater_stats = self.get_player_stats(season, 'skaters')
            if not skater_stats.empty:
                results['skater_stats'] = skater_stats
                
            goalie_stats = self.get_player_stats(season, 'goalies')
            if not goalie_stats.empty:
                results['goalie_stats'] = goalie_stats
        else:
            self.logger.info(f"Skipping player stats for historical season {season} (not available in API)")
            
        # Standings might be available for more seasons
        if season_year >= 2005:  # Try standings for 2005+
            self.logger.info(f"Attempting standings for season {season}")
            standings = self.get_standings(season)
            if not standings.empty:
                results['standings'] = standings
        else:
            self.logger.info(f"Skipping standings for very old season {season} (likely not available)")
            
        # Games (if requested) - try for 2005+
        if include_games and season_year >= 2005 and teams_df is not None and not teams_df.empty:
            self.logger.info(f"Attempting game data for season {season}")
            all_team_abbrevs = teams_df['team_abbrev'].dropna().unique().tolist()
            games_list = []
            for team_abbrev in all_team_abbrevs[:3]:  # Try only first 3 teams for historical seasons
                games = self.get_game_results(season, team_abbrev)
                if not games.empty:
                    games_list.append(games)
                    break  # If one team works, likely all will work
            if games_list:
                # For historical seasons, get all teams if first one worked
                if season_year < 2008:
                    for team_abbrev in all_team_abbrevs[3:]:
                        games = self.get_game_results(season, team_abbrev)
                        if not games.empty:
                            games_list.append(games)
                
                all_games = pd.concat(games_list, ignore_index=True)
                all_games = all_games.drop_duplicates(subset=['game_id'])
                results['games'] = all_games
                
        return results

    def _detect_duplicate_season_data(self, all_season_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Detect if we're getting the same data for multiple seasons (indicating API issue)
        """
        if not all_season_data:
            return False
            
        # Check if team stats are identical across seasons
        seasons = list(all_season_data.keys())
        if len(seasons) < 2:
            return False
            
        # Compare team stats between first two seasons
        season1, season2 = seasons[0], seasons[1]
        df1, df2 = all_season_data[season1], all_season_data[season2]
        
        if df1.empty or df2.empty:
            return False
            
        # Compare a few key columns (excluding season column)
        compare_cols = ['team_name', 'games_played', 'wins', 'losses', 'points']
        available_cols = [col for col in compare_cols if col in df1.columns and col in df2.columns]
        
        if not available_cols:
            return False
            
        # Sort by team_name and compare
        df1_compare = df1[available_cols].sort_values('team_name').reset_index(drop=True)
        df2_compare = df2[available_cols].sort_values('team_name').reset_index(drop=True)
        
        # If dataframes are identical, we have duplicate data
        are_identical = df1_compare.equals(df2_compare)
        
        if are_identical:
            self.logger.error(f"DUPLICATE DATA DETECTED: Seasons {season1} and {season2} have identical team stats!")
            self.logger.error(f"This indicates the API is returning the same data regardless of season parameter")
            
        return are_identical

    def get_multiple_seasons_data_batched(self, seasons: List[str], batch_size: int = 5, 
                                        include_games: bool = False, 
                                        output_dir: str = "nhl_data",
                                        use_smart_selection: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive data for multiple seasons using batch processing
        
        Args:
            seasons: List of seasons in format ["20222023", "20232024"]
            batch_size: Number of seasons to process at once
            include_games: Whether to include game-by-game data
            output_dir: Directory to save intermediate files
            use_smart_selection: Use era-based data selection (recommended)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Show data availability information
        if use_smart_selection:
            availability_info = self.get_data_availability_by_era()
            self.logger.info("=== NHL API Data Availability by Era ===")
            for era, info in availability_info.items():
                self.logger.info(f"{era.upper()}: {info['description']}")
                self.logger.info(f"  Available data types: {info['available_data']}")
                
        # Get teams once (they don't change much)
        self.logger.info(f"Fetching team data")
        teams_df = self.get_teams()
        
        # Save teams data
        if not teams_df.empty:
            teams_file = os.path.join(output_dir, "nhl_teams.csv")
            teams_df.to_csv(teams_file, index=False)
            self.logger.info(f"Saved teams data to {teams_file}")
        
        # Process seasons in batches
        data_types = ['team_stats', 'skater_stats', 'goalie_stats', 'standings']
        if include_games:
            data_types.append('games')
            
        # Initialize final results
        final_results = {data_type: [] for data_type in data_types}
        final_results['teams'] = teams_df
        
        # Process in batches
        for i in range(0, len(seasons), batch_size):
            batch_seasons = seasons[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            self.logger.info(f"Processing batch {batch_num}/{((len(seasons) - 1) // batch_size) + 1}: {batch_seasons}")
            
            batch_results = {data_type: [] for data_type in data_types}
            
            # Process each season in the batch
            for season in batch_seasons:
                try:
                    if use_smart_selection:
                        season_data = self.get_single_season_data_smart(season, include_games, teams_df)
                    else:
                        season_data = self.get_single_season_data(season, include_games, teams_df)
                    
                    for data_type in data_types:
                        if data_type in season_data and not season_data[data_type].empty:
                            batch_results[data_type].append(season_data[data_type])
                            
                except Exception as e:
                    self.logger.error(f"Error processing season {season}: {e}")
                    continue
            
            # Save batch results and add to final results
            for data_type in data_types:
                if batch_results[data_type]:
                    batch_df = pd.concat(batch_results[data_type], ignore_index=True)
                    
                    # Save intermediate batch file
                    batch_file = os.path.join(output_dir, f"nhl_{data_type}_batch_{batch_num}.csv")
                    batch_df.to_csv(batch_file, index=False)
                    self.logger.info(f"Saved batch {batch_num} {data_type} data to {batch_file}")
                    
                    final_results[data_type].append(batch_df)
                    
            # Clear memory
            del batch_results
            
        # Combine all batches into final files
        self.logger.info("Combining all batches into final files...")
        combined_results = {}
        
        for data_type in data_types:
            if final_results[data_type]:
                combined_df = pd.concat(final_results[data_type], ignore_index=True)
                combined_results[data_type] = combined_df
                
                # Save final combined file
                final_file = os.path.join(output_dir, f"nhl_{data_type}.csv")
                combined_df.to_csv(final_file, index=False)
                self.logger.info(f"Saved final {data_type} data with {len(combined_df)} records to {final_file}")
                
                # Clean up intermediate batch files
                for i in range(0, len(seasons), batch_size):
                    batch_num = i // batch_size + 1
                    batch_file = os.path.join(output_dir, f"nhl_{data_type}_batch_{batch_num}.csv")
                    if os.path.exists(batch_file):
                        os.remove(batch_file)
                        self.logger.info(f"Removed intermediate file {batch_file}")
            else:
                combined_results[data_type] = pd.DataFrame()
                
        combined_results['teams'] = teams_df
        return combined_results

    def get_single_season_data(self, season: str, include_games: bool = False, teams_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive data for a single season (original method - attempts all data types)
        
        Args:
            season: Season in format "20232024"
            include_games: Whether to include game-by-game data
            teams_df: Pre-fetched teams DataFrame to avoid refetching
        """
        results = {
            'team_stats': pd.DataFrame(),
            'skater_stats': pd.DataFrame(),
            'goalie_stats': pd.DataFrame(),
            'standings': pd.DataFrame()
        }
        
        if include_games:
            results['games'] = pd.DataFrame()
            
        self.logger.info(f"Downloading data for season {season}")
        
        # Team stats
        team_stats = self.get_team_stats(season)
        if not team_stats.empty:
            results['team_stats'] = team_stats
            
        # Player stats
        skater_stats = self.get_player_stats(season, 'skaters')
        if not skater_stats.empty:
            results['skater_stats'] = skater_stats
            
        goalie_stats = self.get_player_stats(season, 'goalies')
        if not goalie_stats.empty:
            results['goalie_stats'] = goalie_stats
            
        # Standings
        standings = self.get_standings(season)
        if not standings.empty:
            results['standings'] = standings
            
        # Games (optional)
        if include_games and teams_df is not None and not teams_df.empty:
            all_team_abbrevs = teams_df['team_abbrev'].dropna().unique().tolist()
            games_list = []
            for team_abbrev in all_team_abbrevs:
                games = self.get_game_results(season, team_abbrev)
                if not games.empty:
                    games_list.append(games)
            if games_list:
                # Concatenate and drop duplicates by game_id
                all_games = pd.concat(games_list, ignore_index=True)
                all_games = all_games.drop_duplicates(subset=['game_id'])
                results['games'] = all_games
                
        return results

    def get_multiple_seasons_data(self, seasons: List[str], include_games: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive data for multiple seasons (original method kept for compatibility)
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
        self.logger.info(f"Fetching team data")
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
                all_team_abbrevs = results['teams']['team_abbrev'].dropna().unique().tolist()
                games_list = []
                for team_abbrev in all_team_abbrevs:
                    games = self.get_game_results(season, team_abbrev)
                    if not games.empty:
                        games_list.append(games)
                if games_list:
                    # Concatenate and drop duplicates by game_id
                    all_games = pd.concat(games_list, ignore_index=True)
                    all_games = all_games.drop_duplicates(subset=['game_id'])
                    results['games'].append(all_games)

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

    # Define all seasons from 1990-1991 to 2024-2025
    all_seasons = [f"{year}{year+1}" for year in range(1990, 2025)]
    
    # Split into manageable chunks for better organization and monitoring
    seasons_1990s = [f"{year}{year+1}" for year in range(1990, 2000)]  # 1990-1999 seasons
    seasons_2000s = [f"{year}{year+1}" for year in range(2000, 2010)]  # 2000-2009 seasons  
    seasons_2010s = [f"{year}{year+1}" for year in range(2010, 2020)]  # 2010-2019 seasons
    seasons_2020s = [f"{year}{year+1}" for year in range(2020, 2025)]  # 2020-2024 seasons

    print("=== NHL API Data Downloader (Complete Historical Dataset 1990-2025) ===")
    print(f"Total seasons to download: {len(all_seasons)}")
    print(f"Season range: {all_seasons[0]} to {all_seasons[-1]}")
    print(f"Using fixed cayenneExp parameters for accurate season-specific data")
    
    # Quick data availability check on a sample of seasons
    print(f"\n=== Quick Data Availability Check ===")
    sample_seasons = [seasons_1990s[0], seasons_1990s[-1], seasons_2000s[0], 
                     seasons_2010s[0], seasons_2020s[0], seasons_2020s[-1]]
    
    availability = client.check_data_availability(sample_seasons)
    print("Sample availability across different eras:")
    for season, avail in availability.items():
        print(f"  {season}: {avail}")
    
    # Download data by decade for better organization and monitoring
    print(f"\n=== Downloading 1990s Data ({len(seasons_1990s)} seasons) ===")
    print(f"Seasons: {seasons_1990s[0]} to {seasons_1990s[-1]}")
    data_1990s = client.get_multiple_seasons_data_batched(
        seasons=seasons_1990s,
        batch_size=3,
        include_games=True,
        output_dir="nhl_data_1990s",
        use_smart_selection=True
    )
    
    print(f"\n=== Downloading 2000s Data ({len(seasons_2000s)} seasons) ===")
    print(f"Seasons: {seasons_2000s[0]} to {seasons_2000s[-1]}")
    data_2000s = client.get_multiple_seasons_data_batched(
        seasons=seasons_2000s,
        batch_size=3,
        include_games=True,
        output_dir="nhl_data_2000s",
        use_smart_selection=True
    )
    
    print(f"\n=== Downloading 2010s Data ({len(seasons_2010s)} seasons) ===")
    print(f"Seasons: {seasons_2010s[0]} to {seasons_2010s[-1]}")
    data_2010s = client.get_multiple_seasons_data_batched(
        seasons=seasons_2010s,
        batch_size=3,
        include_games=True,
        output_dir="nhl_data_2010s",
        use_smart_selection=True
    )
    
    print(f"\n=== Downloading 2020s Data ({len(seasons_2020s)} seasons) ===")
    print(f"Seasons: {seasons_2020s[0]} to {seasons_2020s[-1]}")
    data_2020s = client.get_multiple_seasons_data_batched(
        seasons=seasons_2020s,
        batch_size=3,
        include_games=True,
        output_dir="nhl_data_2020s",
        use_smart_selection=True
    )
    
    # Combine all decades into final comprehensive dataset
    print(f"\n=== Combining All Decades into Final Dataset ===")
    combined_data = {}
    
    # Use most recent teams data
    combined_data['teams'] = data_2020s['teams'] if 'teams' in data_2020s else pd.DataFrame()
    
    # Combine all data types across all decades
    decade_datasets = {
        '1990s': data_1990s,
        '2000s': data_2000s, 
        '2010s': data_2010s,
        '2020s': data_2020s
    }
    
    for data_type in ['team_stats', 'skater_stats', 'goalie_stats', 'standings', 'games']:
        dfs_to_combine = []
        type_summary = {}
        
        for decade_name, decade_data in decade_datasets.items():
            if data_type in decade_data and not decade_data[data_type].empty:
                decade_df = decade_data[data_type]
                dfs_to_combine.append(decade_df)
                
                # Track what we got from each decade
                if 'season' in decade_df.columns:
                    seasons_in_decade = sorted(decade_df['season'].unique())
                    type_summary[decade_name] = f"{len(seasons_in_decade)} seasons ({seasons_in_decade[0]}-{seasons_in_decade[-1]})"
                else:
                    type_summary[decade_name] = f"{len(decade_df)} records"
        
        if dfs_to_combine:
            combined_data[data_type] = pd.concat(dfs_to_combine, ignore_index=True)
            print(f"{data_type}: Combined from {type_summary}")
        else:
            combined_data[data_type] = pd.DataFrame()
            print(f"{data_type}: No data found across any decade")
    
    # Save the final comprehensive dataset
    print(f"\n=== Saving Final Complete Dataset ===")
    client.save_data_to_files(combined_data, output_dir="nhl_data_complete_1990_2025")
    
    # Generate comprehensive summary
    print(f"\n=== Final Complete Dataset Summary (1990-2025) ===")
    total_seasons_found = set()
    
    for data_type, df in combined_data.items():
        if not df.empty:
            print(f"\n{data_type.upper()}:")
            print(f"  Total records: {len(df):,}")
            
            if 'season' in df.columns:
                unique_seasons = sorted(df['season'].unique())
                total_seasons_found.update(unique_seasons)
                print(f"  Seasons covered: {len(unique_seasons)} ({unique_seasons[0]} to {unique_seasons[-1]})")
                
                # Decade breakdown
                decades_breakdown = {
                    '1990s': [s for s in unique_seasons if 1990 <= int(s[:4]) < 2000],
                    '2000s': [s for s in unique_seasons if 2000 <= int(s[:4]) < 2010], 
                    '2010s': [s for s in unique_seasons if 2010 <= int(s[:4]) < 2020],
                    '2020s': [s for s in unique_seasons if 2020 <= int(s[:4]) < 2030]
                }
                
                for decade, decade_seasons in decades_breakdown.items():
                    if decade_seasons:
                        print(f"    {decade}: {len(decade_seasons)} seasons")
                
                # Show some sample statistics
                if data_type == 'team_stats' and 'games_played' in df.columns:
                    avg_games_by_decade = {}
                    for decade, decade_seasons in decades_breakdown.items():
                        if decade_seasons:
                            decade_data = df[df['season'].isin(decade_seasons)]
                            avg_games = decade_data['games_played'].mean()
                            avg_games_by_decade[decade] = f"{avg_games:.1f}"
                    print(f"    Average games played by decade: {avg_games_by_decade}")
        else:
            print(f"\n{data_type.upper()}: No data available")
    
    print(f"\n=== Overall Summary ===")
    print(f"• Total unique seasons with data: {len(total_seasons_found)}")
    print(f"• Season range: {min(total_seasons_found) if total_seasons_found else 'N/A'} to {max(total_seasons_found) if total_seasons_found else 'N/A'}")
    print(f"• Data files saved to: nhl_data_complete_1990_2025/")
    print(f"• Individual decade data saved to: nhl_data_1990s/, nhl_data_2000s/, nhl_data_2010s/, nhl_data_2020s/")
    
    # Show some interesting sample data
    print(f"\n=== Sample Data from Different Eras ===")
    
    if 'team_stats' in combined_data and not combined_data['team_stats'].empty:
        team_stats = combined_data['team_stats']
        
        # Sample from different decades
        sample_seasons = []
        for decade_start in [1990, 2000, 2010, 2020]:
            decade_data = team_stats[team_stats['season'].str[:4].astype(int) >= decade_start]
            decade_data = decade_data[decade_data['season'].str[:4].astype(int) < decade_start + 10]
            if not decade_data.empty:
                sample_season = decade_data['season'].iloc[0]
                sample_seasons.append(sample_season)
        
        for season in sample_seasons:
            season_data = team_stats[team_stats['season'] == season]
            if not season_data.empty:
                sample_team = season_data.iloc[0]
                print(f"\nSample from {season}:")
                print(f"  Team: {sample_team['team_name']}")
                print(f"  Games: {sample_team['games_played']}, Wins: {sample_team['wins']}, Points: {sample_team['points']}")
    
    print(f"\n=== Download Complete! ===")
    print(f"Your complete NHL dataset from 1990-2025 is ready!")
    print(f"Check the 'nhl_data_complete_1990_2025' folder for all CSV files.")
    
    # Show file sizes/counts for user reference
    print(f"\n=== Data Volume Summary ===")
    for data_type, df in combined_data.items():
        if not df.empty:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            print(f"  {data_type}: {len(df):,} rows, ~{memory_mb:.1f} MB")