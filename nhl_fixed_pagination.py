def get_player_stats(self, season: str, player_type: str = 'skaters') -> pd.DataFrame:
    """
    Get player statistics for a season with proper pagination
    
    Args:
        season: Season in format "20232024"
        player_type: 'skaters' or 'goalies'
    """
    if player_type == 'skaters':
        url = f"{self.stats_api_base}/en/skater/summary"
    else:
        url = f"{self.stats_api_base}/en/goalie/summary"

    all_players_data = []
    start = 0
    limit = 100  # Use smaller, more reliable limit
    
    self.logger.info(f"Starting to fetch {player_type} data for season {season}")
    
    while True:
        # Build the cayenneExp parameter properly
        cayenne_exp = f"seasonId={season} and gameTypeId=2"
        
        params = {
            'limit': limit,
            'start': start,
            'cayenneExp': cayenne_exp,
            'sort': 'points' if player_type == 'skaters' else 'wins'
        }

        self.logger.info(f"Fetching {player_type} batch: start={start}, limit={limit}")
        data = self._make_request(url, params)
        
        if not data or 'data' not in data:
            self.logger.warning(f"No data returned for {player_type} at start={start}")
            break
            
        current_batch = data['data']
        if not current_batch:
            self.logger.info(f"Empty batch returned for {player_type} at start={start} - end of data")
            break
            
        batch_size = len(current_batch)
        self.logger.info(f"Retrieved {batch_size} {player_type} records starting from position {start}")
        
        # Process the current batch
        for player in current_batch:
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
        
        # Check if we got fewer results than requested (indicates end of data)
        if batch_size < limit:
            self.logger.info(f"Received {batch_size} < {limit} records - reached end of data")
            break
            
        start += limit  # Move to next page
        
        # Safety check to prevent infinite loops
        if len(all_players_data) > 10000:  # Reasonable upper bound
            self.logger.warning(f"Retrieved {len(all_players_data)} records - stopping to prevent runaway")
            break

    df = pd.DataFrame(all_players_data)
    self.logger.info(f"Final result: Retrieved {player_type} stats for {len(df)} total players in season {season}")
    return df

def get_all_player_stats_comprehensive(self, season: str) -> Dict[str, pd.DataFrame]:
    """
    Get comprehensive player stats including regular season and playoffs
    """
    results = {}
    
    # Regular season skaters
    self.logger.info("Fetching regular season skater stats...")
    reg_skaters = self.get_player_stats(season, 'skaters')
    if not reg_skaters.empty:
        reg_skaters['game_type'] = 'regular'
        results['skaters_regular'] = reg_skaters
    
    # Regular season goalies  
    self.logger.info("Fetching regular season goalie stats...")
    reg_goalies = self.get_player_stats(season, 'goalies')
    if not reg_goalies.empty:
        reg_goalies['game_type'] = 'regular'
        results['goalies_regular'] = reg_goalies
    
    # Playoff stats (modify the method to handle different game types)
    playoff_skaters = self.get_player_stats_playoffs(season, 'skaters')
    if not playoff_skaters.empty:
        playoff_skaters['game_type'] = 'playoffs'
        results['skaters_playoffs'] = playoff_skaters
        
    playoff_goalies = self.get_player_stats_playoffs(season, 'goalies')
    if not playoff_goalies.empty:
        playoff_goalies['game_type'] = 'playoffs'
        results['goalies_playoffs'] = playoff_goalies
    
    return results

def get_player_stats_playoffs(self, season: str, player_type: str = 'skaters') -> pd.DataFrame:
    """
    Get playoff player statistics - same as regular but with gameTypeId=3
    """
    if player_type == 'skaters':
        url = f"{self.stats_api_base}/en/skater/summary"
    else:
        url = f"{self.stats_api_base}/en/goalie/summary"

    all_players_data = []
    start = 0
    limit = 100
    
    self.logger.info(f"Starting to fetch playoff {player_type} data for season {season}")
    
    while True:
        # Use gameTypeId=3 for playoffs
        cayenne_exp = f"seasonId={season} and gameTypeId=3"
        
        params = {
            'limit': limit,
            'start': start,
            'cayenneExp': cayenne_exp,
            'sort': 'points' if player_type == 'skaters' else 'wins'
        }

        data = self._make_request(url, params)
        
        if not data or 'data' not in data:
            break
            
        current_batch = data['data']
        if not current_batch:
            break
            
        batch_size = len(current_batch)
        self.logger.info(f"Retrieved {batch_size} playoff {player_type} records starting from position {start}")
        
        # Same processing logic as regular season
        for player in current_batch:
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
        
        if batch_size < limit:
            break
            
        start += limit

    df = pd.DataFrame(all_players_data)
    self.logger.info(f"Final result: Retrieved playoff {player_type} stats for {len(df)} total players in season {season}")
    return df