import fastf1 as ff1
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import os
import time
import gc
import copy # Add copy for deep copying standings
# Removed specific FastF1 exception imports

# --- Configuration ---
# Get the directory where the script is located
script_dir = Path(__file__).resolve().parent

# Define cache paths relative to the script's directory
FASTF1_CACHE_PATH = os.getenv('FASTF1_CACHE_PATH', script_dir / 'cache')
DATA_CACHE_PATH = script_dir / "data_cache"

if not os.path.exists(FASTF1_CACHE_PATH):
    os.makedirs(FASTF1_CACHE_PATH)
ff1.Cache.enable_cache(FASTF1_CACHE_PATH)

# --- Helper Functions ---

def get_team_color_name(team_name: str | None) -> str:
    """Gets a simplified team color name."""
    if not team_name: return 'gray'
    simple_name = team_name.lower().replace(" ", "").replace("-", "")
    if 'mclaren' in simple_name: return 'mclaren'
    if 'mercedes' in simple_name: return 'mercedes'
    if 'redbull' in simple_name: return 'redbull'
    if 'ferrari' in simple_name: return 'ferrari'
    if 'alpine' in simple_name: return 'alpine'
    if 'astonmartin' in simple_name: return 'astonmartin'
    if 'williams' in simple_name: return 'williams';
    if 'haas' in simple_name: return 'haas'
    if 'sauber' in simple_name: return 'alfaromeo' # Covers Kick Sauber too
    if 'racingbulls' in simple_name or 'alphatauri' in simple_name: return 'alphatauri'
    return 'gray'

def save_json(data, file_path: Path):
    """Saves data to a JSON file, handling potential numpy types."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to list of dicts first
                records = data.replace({np.nan: None}).to_dict(orient='records')
                # Add pd.Timedelta handling to default serializer
                json.dump(records, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x) if isinstance(x, pd.Timestamp) else x.total_seconds() if isinstance(x, pd.Timedelta) else None if pd.isna(x) else x)
            elif isinstance(data, (list, dict)):
                 # Handle potential numpy/pandas types within list/dict structures if necessary
                 # Add pd.Timedelta handling to default serializer
                 json.dump(data, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x) if isinstance(x, pd.Timestamp) else x.total_seconds() if isinstance(x, pd.Timedelta) else None if pd.isna(x) else x)
            else:
                 print(f" -> Unsupported data type for JSON saving: {type(data)}")
                 return
        # print(f" -> Data successfully saved to {file_path}") # Reduce verbosity
    except Exception as e:
        print(f" -> Error saving JSON to {file_path}: {e}")
        import traceback
        traceback.print_exc()

def is_sprint_weekend(event_format: str) -> bool:
    """Check if event uses a sprint weekend format based on FastF1 EventFormat."""
    # Check if the format string contains 'sprint', case-insensitive
    return isinstance(event_format, str) and 'sprint' in event_format.lower()


def format_lap_time(lap_time_delta):
    """Formats a Pandas Timedelta lap time into MM:SS.ms string."""
    if pd.isna(lap_time_delta) or lap_time_delta is None:
        return None
    # Ensure it's a Timedelta object
    if not isinstance(lap_time_delta, pd.Timedelta):
        try:
            # Attempt conversion if it's a compatible type (like string)
            lap_time_delta = pd.to_timedelta(lap_time_delta)
        except (ValueError, TypeError):
            return None # Cannot convert

    total_seconds = lap_time_delta.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000)) # Round milliseconds
    # Corrected format string for minutes
    return f"{minutes:d}:{seconds:02d}.{milliseconds:03d}"

def get_session_details(event):
    """Extract all sessions present in an event based on schedule data."""
    sessions = []
    # Check all possible session slots (Session1Type to Session5Type)
    for i in range(1, 6):
        session_type_key = f'Session{i}Type'
        session_date_key = f'Session{i}Date'
        # Safely get values using .get() on the Series (event)
        session_type = event.get(session_type_key)
        session_date = event.get(session_date_key)

        if pd.notna(session_type) and pd.notna(session_date):
            sessions.append({
                'type': session_type,
                'date': session_date # Keep as Timestamp for now
            })
    # Add Race session separately if EventDate exists
    race_date = event.get('EventDate')
    if pd.notna(race_date):
         sessions.append({
             'type': 'R',
             'date': race_date
         })
    # Sort sessions chronologically by date
    sessions.sort(key=lambda x: x['date'])
    return sessions

def process_qualifying_segment(laps: pd.DataFrame, session_id: str) -> list[dict]:
    """Process qualifying segment results from laps data."""
    print(f"    -> Processing qualifying segment: {session_id}")
    if laps is None or laps.empty:
        print(f"    !! No lap data provided for segment {session_id}")
        return []

    # Filter for the specific segment (Q1, Q2, Q3, SQ1, etc.)
    # FastF1 uses 'Segment' column in recent versions. Need to check its existence.
    # This logic needs refinement based on actual FastF1 data structure for segments.
    # Assuming 'Segment' column exists for now.
    if 'Segment' not in laps.columns:
        print(f"    !! 'Segment' column not found in lap data for {session_id}. Cannot process segment.")
        # Fallback: If no segment column, assume all laps belong to the requested segment (e.g., 'Q')
        if session_id in ['Q', 'SQ']:
             segment_laps = laps.copy()
             print(f"    -> WARNING: 'Segment' column missing, using all laps for {session_id}")
        else:
             return []
    else:
        segment_laps = laps[laps['Segment'] == session_id].copy()

    if segment_laps.empty:
        print(f"    !! No laps found for segment {session_id}")
        return []

    accurate_laps = segment_laps.pick_accurate()
    processed = []

    # Get all drivers who participated in the segment
    all_segment_drivers_info = segment_laps[['Driver', 'Team', 'FullName']].drop_duplicates()

    if accurate_laps.empty:
        print(f"    !! No accurate laps found for segment {session_id}")
        # Return drivers who participated but set no time
        for _, driver_row in all_segment_drivers_info.iterrows():
             processed.append({
                 'position': None, 'driverCode': str(driver_row.get('Driver')),
                 'fullName': str(driver_row.get('FullName', 'N/A')),
                 'team': str(driver_row.get('Team', 'N/A')),
                 'teamColor': get_team_color_name(driver_row.get('Team')),
                 'status': 'No Time Set', 'fastestLapTime': None,
                 'lapsCompleted': len(segment_laps[segment_laps['Driver'] == driver_row.get('Driver')])
             })
        processed.sort(key=lambda x: x['fullName']) # Sort alphabetically if no times
        return processed

    # Get best lap for each driver in this segment
    best_laps = accurate_laps.loc[accurate_laps.groupby('Driver')['LapTime'].idxmin()]

    # Create results for drivers with accurate times
    drivers_with_times = set()
    for _, row in best_laps.iterrows():
        driver_code = str(row.get('Driver'))
        drivers_with_times.add(driver_code)
        processed.append({
            'position': None, # Will be assigned after sorting
            'driverCode': driver_code,
            'fullName': str(row.get('FullName', 'N/A')),
            'team': str(row.get('Team', 'N/A')),
            'teamColor': get_team_color_name(row.get('Team')),
            'status': 'Time Set',
            'fastestLapTime': format_lap_time(row.get('LapTime')),
            'lapsCompleted': len(segment_laps[segment_laps['Driver'] == driver_code])
        })

    # Add drivers who participated but didn't set an accurate time
    for _, driver_row in all_segment_drivers_info.iterrows():
        driver_code = str(driver_row.get('Driver'))
        if driver_code not in drivers_with_times:
             processed.append({
                 'position': None, 'driverCode': driver_code,
                 'fullName': str(driver_row.get('FullName', 'N/A')),
                 'team': str(driver_row.get('Team', 'N/A')),
                 'teamColor': get_team_color_name(driver_row.get('Team')),
                 'status': 'No Accurate Time', 'fastestLapTime': None,
                 'lapsCompleted': len(segment_laps[segment_laps['Driver'] == driver_code])
             })

    # Sort by lap time (None times go last), then assign position
    processed.sort(key=lambda x: (x['fastestLapTime'] is None, x['fastestLapTime']))
    for i, res in enumerate(processed):
        res['position'] = i + 1

    print(f"    -> Processed {len(processed)} drivers for segment {session_id}")
    return processed


def process_session_results(year: int, event: str, session_identifier: str, save_file: bool = True):
    """
    Process results for a specific session (FP1, Q1, R, Sprint, etc.)
    and optionally save relevant columns to JSON. Always returns processed data.
    """
    print(f"  -> Processing results for: {year} {event} {session_identifier} (Save: {save_file})")
    processed_results = []
    # Ensure consistent lowercase slug for results files (using event name passed in)
    event_slug = event.lower().replace(' ', '_')
    results_file = DATA_CACHE_PATH / str(year) / "races" / f"{event_slug}_{session_identifier}.json"

    try:
        # Determine parent session for segments
        parent_session_id = None
        is_segment = False
        if session_identifier in ['Q1', 'Q2', 'Q3']:
            parent_session_id = 'Q'
            is_segment = True
        elif session_identifier in ['SQ1', 'SQ2', 'SQ3']:
            parent_session_id = 'SQ'
            is_segment = True

        # Load session data
        session_to_load = parent_session_id if is_segment else session_identifier
        session_obj = ff1.get_session(year, event, session_to_load)
        # Determine if laps *should* be loaded based on session type
        should_load_laps = session_identifier.startswith(('FP', 'Q', 'SQ')) or is_segment
        # Force load laps for R and Sprint as well, as they are needed for fastest lap calc
        if session_identifier == 'R' or session_identifier == 'Sprint':
            should_load_laps = True
        print(f"    -> Loading laps for {session_identifier}: {should_load_laps}")
        session_obj.load(laps=should_load_laps, telemetry=False, weather=False, messages=False)

        results = session_obj.results
        # Assign laps_data *after* loading, based on should_load_laps
        laps_data = session_obj.laps if should_load_laps else None

        if results is None: results = pd.DataFrame() # Ensure results is DataFrame

        # --- Handle Qualifying/Sprint Qualifying Segments ---
        if is_segment:
            if laps_data is not None and not laps_data.empty:
                processed_results = process_qualifying_segment(laps_data, session_identifier)
            else:
                print(f"    !! No lap data found for parent session {parent_session_id} to process segment {session_identifier}")

        # --- Handle FP, R, Sprint, or Fallback Q/SQ (if results exist but laps don't for segments) ---
        else:
            if results.empty and session_identifier.startswith('FP') and laps_data is not None and not laps_data.empty:
                 print(f"    -> No structured results for {session_identifier}, using lap data...")
                 drivers_in_laps = laps_data[['Driver', 'Team', 'FullName']].drop_duplicates(subset=['Driver'])
                 for _, driver_row in drivers_in_laps.iterrows():
                     driver_code = str(driver_row.get('Driver'))
                     if not driver_code: continue
                     driver_laps = laps_data.pick_driver(driver_code)
                     fastest = driver_laps.pick_fastest() if not driver_laps.empty else None
                     processed_results.append({
                         'position': None, 'driverCode': driver_code,
                         'fullName': str(driver_row.get('FullName', 'N/A')),
                         'team': str(driver_row.get('Team', 'N/A')),
                         'teamColor': get_team_color_name(driver_row.get('Team')),
                         'status': 'N/A', 'points': 0.0,
                         'fastestLapTime': format_lap_time(fastest['LapTime']) if fastest is not None else None,
                         'lapsCompleted': len(driver_laps) if not driver_laps.empty else 0
                     })
                 # Sort FP results by fastest lap time
                 processed_results.sort(key=lambda x: (x['fastestLapTime'] is None, x['fastestLapTime']))
                 for i, res in enumerate(processed_results):
                     res['position'] = i + 1

            elif not results.empty:
                # Process standard results DataFrame
                print(f"    -> Processing standard results for {session_identifier}")
                for index, row in results.iterrows():
                    driver_code = str(row.get('Abbreviation')) if pd.notna(row.get('Abbreviation')) else None
                    if not driver_code: continue

                    # --- Add Debug Print Here (Raw Row) ---
                    # print(f"      -> Raw Row for {driver_code}: Pos={row.get('Position')}, Grid={row.get('GridPosition')}, Q1={row.get('Q1')}, Q2={row.get('Q2')}, Q3={row.get('Q3')}")
                    # --- End Debug Print ---

                    result = {
                        'position': int(row['Position']) if pd.notna(row['Position']) else None,
                        'driverCode': driver_code,
                        'fullName': str(row.get('FullName', 'N/A')),
                        'team': str(row.get('TeamName', 'N/A')),
                        'teamColor': get_team_color_name(row.get('TeamName')),
                        'status': str(row.get('Status', 'N/A')),
                    }

                    if session_identifier == 'R' or session_identifier == 'Sprint':
                        result['points'] = float(row.get('Points', 0.0)) if pd.notna(row.get('Points')) else 0.0
                        result['gridPosition'] = int(row.get('GridPosition')) if pd.notna(row.get('GridPosition')) else None
                         # --- Add Debug Print Here (Processed GridPos) ---
                        # print(f"      -> Processed Result for {driver_code}: gridPosition={result.get('gridPosition')}")
                         # --- End Debug Print ---
                        # isFastestLap might need separate handling
                    elif session_identifier.startswith('FP'):
                         result['points'] = 0.0
                         if laps_data is not None and not laps_data.empty:
                             driver_laps = laps_data.pick_driver(driver_code)
                             fastest = driver_laps.pick_fastest() if not driver_laps.empty else None
                             result['fastestLapTime'] = format_lap_time(fastest['LapTime']) if fastest is not None else None
                             result['lapsCompleted'] = len(driver_laps) if not driver_laps.empty else 0
                         else:
                             result['fastestLapTime'] = None; result['lapsCompleted'] = 0
                    # Add Q/SQ times from results if processing the parent session ('Q' or 'SQ')
                    elif session_identifier == 'Q' or session_identifier == 'SQ':
                         result['q1Time'] = format_lap_time(row.get('Q1'))
                         result['q2Time'] = format_lap_time(row.get('Q2'))
                         result['q3Time'] = format_lap_time(row.get('Q3'))

                    processed_results.append(result)
                # Sort standard results by position
                processed_results.sort(key=lambda x: (x['position'] is None, x['position']))

                # --- DEBUG: Print Qualifying Times before returning ---
                if session_identifier.startswith('Q'):
                    print(f"    -> DEBUG: Qualifying data being processed: {processed_results}") # Added this print

                # --- Add Fastest Lap Flag (specifically for Race 'R' or 'Sprint') ---
                if (session_identifier == 'R' or session_identifier == 'Sprint') and laps_data is not None and not laps_data.empty:
                    try:
                        fastest_lap = laps_data.pick_fastest()
                        if fastest_lap is not None and pd.notna(fastest_lap['Driver']):
                            fastest_driver_code = str(fastest_lap['Driver'])
                            fastest_time_formatted = format_lap_time(fastest_lap['LapTime'])
                            print(f"    -> Fastest lap found: Driver {fastest_driver_code}, Time {fastest_time_formatted}")
                            for res in processed_results:
                                is_fastest = (res.get('driverCode') == fastest_driver_code)
                                res['isFastestLap'] = is_fastest
                                if is_fastest:
                                    res['fastestLapTimeValue'] = fastest_time_formatted # Store the formatted time
                        else:
                            print("    -> Could not determine fastest lap holder.")
                            for res in processed_results:
                                res['isFastestLap'] = False # Ensure flag is present and false if none found
                    except Exception as fl_err:
                        print(f"    !! Error determining fastest lap: {fl_err}")
                        for res in processed_results:
                             res['isFastestLap'] = False # Default to false on error
                elif (session_identifier == 'R' or session_identifier == 'Sprint'):
                     print("    !! No lap data available to determine fastest lap.")
                     for res in processed_results:
                          res['isFastestLap'] = False # Ensure flag is present and false if no laps
                          res.pop('fastestLapTimeValue', None) # Remove potential value if laps aren't available

                # --- Pole Lap Time logic removed - will be handled in process_season_data ---

            else:
                 print(f"    !! No results or lap data found for {year} {event} {session_identifier}")

        # Save the processed results (even if empty), only if requested
        if save_file:
            save_json(processed_results, results_file)
            print(f"    -> Saved results ({len(processed_results)} drivers) for {session_identifier} to {results_file.name}")
        else:
            print(f"    -> Processed results ({len(processed_results)} drivers) for {session_identifier}, skipping save.")
        return processed_results # Always return the processed data

    except Exception as e:
        print(f"    !! Error processing session results for {session_identifier}: {e}")
        # Attempt to save empty file on error
        try:
            if not results_file.exists():
                 # Save empty file only if saving was requested
                 if save_file:
                     save_json([], results_file)
                     print(f"    -> Saved empty results file for {session_identifier} due to error.")
                 else:
                      print(f"    -> Returning empty results for {session_identifier} due to error (save skipped).")
        except Exception as save_err:
             print(f"    !! Failed to save empty results file after error: {save_err}")
        return [] # Return empty list on error


def _process_session_points(results: pd.DataFrame | None,
                           driver_standings: defaultdict,
                           team_standings: defaultdict,
                           is_sprint_session: bool):
    """Universal points processing for any session type (Sprint or Race)."""
    if results is None or results.empty or not isinstance(results, pd.DataFrame):
        print(f"    !! No results data to process for points (Sprint={is_sprint_session}).")
        return

    print(f"    -> Processing points from {'Sprint' if is_sprint_session else 'Race'} session...")
    points_added_driver = 0
    points_added_team = 0
    # Ensure required columns exist
    required_cols = ['Abbreviation', 'TeamName', 'Points', 'FullName']
    if not all(col in results.columns for col in required_cols):
        print(f"    !! Missing required columns for points processing in {'Sprint' if is_sprint_session else 'Race'} results: { [col for col in required_cols if col not in results.columns] }")
        return

    for _, res in results.iterrows():
        try:
            driver_code = str(res.get('Abbreviation')) if pd.notna(res.get('Abbreviation')) else None
            team_name = str(res.get('TeamName')) if pd.notna(res.get('TeamName')) else None
            points = float(res.get('Points', 0.0)) if pd.notna(res.get('Points')) else 0.0

            if not driver_code or not team_name: continue

            # Initialize standings dicts if driver/team is new
            driver_standings[driver_code].setdefault('points', 0.0)
            driver_standings[driver_code].setdefault('sprint_points', 0.0)
            driver_standings[driver_code].setdefault('wins', 0)
            driver_standings[driver_code].setdefault('podiums', 0)
            team_standings[team_name].setdefault('points', 0.0)
            team_standings[team_name].setdefault('sprint_points', 0.0)
            team_standings[team_name].setdefault('wins', 0)
            team_standings[team_name].setdefault('podiums', 0)

            # Update total points
            driver_standings[driver_code]['points'] += points
            team_standings[team_name]['points'] += points
            points_added_driver += points
            points_added_team += points

            # Update driver/team info (can be updated by either sprint or race)
            # Ensure 'FullName' exists before accessing
            full_name = res.get('FullName', 'N/A')
            driver_standings[driver_code]['team'] = team_name
            driver_standings[driver_code]['name'] = full_name if pd.notna(full_name) else 'N/A'
            driver_standings[driver_code]['code'] = driver_code
            team_standings[team_name]['team'] = team_name
            team_color = get_team_color_name(team_name) # Get color once
            team_standings[team_name]['color'] = team_color

            # Track sprint points separately
            if is_sprint_session:
                driver_standings[driver_code]['sprint_points'] += points
                team_standings[team_name]['sprint_points'] += points

            # --- IMPORTANT: Win/Podium counting moved to after Race processing ---

        except KeyError as ke: print(f"    !! Missing key processing session points row: {ke}")
        except Exception as row_err: print(f"    !! Error processing session points row: {row_err}")
    print(f"    -> Added {points_added_driver:.1f} driver points and {points_added_team:.1f} team points from {'Sprint' if is_sprint_session else 'Race'}.")


# --- Main Data Processing Function ---

def process_season_data(year: int):
    """
    Processes all completed race weekends (including Sprints) for a given year,
    calculates standings, and saves results, standings, and chart data to JSON files.
    """
    print(f"--- Starting data processing for {year} ---")
    year_cache_path = DATA_CACHE_PATH / str(year)
    year_cache_path.mkdir(parents=True, exist_ok=True)
    races_path = year_cache_path / "races"
    races_path.mkdir(parents=True, exist_ok=True)
    standings_path = year_cache_path / "standings"
    standings_path.mkdir(parents=True, exist_ok=True)
    charts_path = year_cache_path / "charts"
    charts_path.mkdir(parents=True, exist_ok=True)

    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        # Ensure EventDate is timezone-aware for comparison
        if schedule['EventDate'].dt.tz is None:
             schedule['EventDate'] = schedule['EventDate'].dt.tz_localize('UTC')
        completed_events = schedule[schedule['EventDate'] < datetime.now(timezone.utc)]


        if completed_events.empty:
            print(f"No completed events found for {year} yet.")
            if not (standings_path / "standings.json").exists():
                 save_json({"drivers": [], "teams": []}, standings_path / "standings.json")
            return

        print(f"Found {len(completed_events)} completed events for {year}.")
        is_ongoing = len(completed_events) < len(schedule)
        print(f"Season ongoing: {is_ongoing}")

        # Cumulative standings dictionaries
        driver_standings = defaultdict(lambda: {'points': 0.0, 'sprint_points': 0.0, 'wins': 0, 'podiums': 0, 'team': '', 'name': '', 'code': ''})
        team_standings = defaultdict(lambda: {'points': 0.0, 'sprint_points': 0.0, 'wins': 0, 'podiums': 0, 'team': '', 'color': 'gray'})
        previous_driver_standings = {}
        previous_team_standings = {}
        all_race_results_summary = []

        num_completed = len(completed_events)
        for i, (index, event) in enumerate(completed_events.iterrows()):
            event_name = event['EventName']
            round_number = event['RoundNumber']
            event_format = event['EventFormat']
            # Ensure event_slug is lowercase for consistent file naming
            event_slug = event_name.lower().replace(' ', '_')

            print(f"\nProcessing event: {event_name} (Round {round_number}, Format: {event_format})...")
            # Chart file paths (remain the same, based on event slug)
            # Use the lowercase slug for chart file paths as well
            lap_times_file = charts_path / f"{event_slug}_laptimes.json"
            tire_strategy_file = charts_path / f"{event_slug}_tirestrategy.json"
            session_drivers_file = charts_path / f"{event_slug}_drivers.json"
            positions_file = charts_path / f"{event_slug}_positions.json"

            # Snapshot standings before processing the *last* completed event if season is ongoing
            if is_ongoing and i == num_completed - 1:
                print(f"  -> Snapshotting standings before processing final completed event: {event_name}")
                previous_driver_standings = copy.deepcopy(driver_standings)
                previous_team_standings = copy.deepcopy(team_standings)

            # --- Determine and Process All Sessions/Segments ---
            session_details = get_session_details(event)
            sessions_to_process = []
            q_session_laps = None # Store loaded Q laps to avoid reloading

            # Expand Q/SQ into segments
            for session_info in session_details:
                session_type = session_info['type']
                if session_type == 'Qualifying':
                    try:
                        qual_session = ff1.get_session(year, round_number, 'Q')
                        qual_session.load(laps=True, telemetry=False, weather=False, messages=False)
                        q_session_laps = qual_session.laps # Store laps
                        has_q1, has_q2, has_q3 = False, False, False
                        if q_session_laps is not None and not q_session_laps.empty:
                             # Check based on results columns first (more reliable if present)
                             if qual_session.results is not None and not qual_session.results.empty:
                                 has_q1 = 'Q1' in qual_session.results.columns and pd.notna(qual_session.results['Q1']).any()
                                 has_q2 = 'Q2' in qual_session.results.columns and pd.notna(qual_session.results['Q2']).any()
                                 has_q3 = 'Q3' in qual_session.results.columns and pd.notna(qual_session.results['Q3']).any()
                             # Fallback to laps if results columns are missing/empty
                             else:
                                 print("    -> Qualifying results columns missing, checking laps for segments (simplified check).")
                                 # Check if laps exist for each segment (needs refinement based on actual data structure)
                                 # Assuming 'Segment' column exists or similar logic
                                 if 'Segment' in q_session_laps.columns:
                                     segments_present = q_session_laps['Segment'].unique()
                                     has_q1 = 'Q1' in segments_present
                                     has_q2 = 'Q2' in segments_present
                                     has_q3 = 'Q3' in segments_present
                                 else: # Crude fallback if no segment column
                                     has_q1 = True # Assume Q1 always happens if Q exists

                        if has_q3: sessions_to_process.extend(['Q1', 'Q2', 'Q3'])
                        elif has_q2: sessions_to_process.extend(['Q1', 'Q2'])
                        elif has_q1: sessions_to_process.append('Q1')
                        else: sessions_to_process.append('Q') # Fallback if no segments detected
                        del qual_session # Clean up session object
                    except Exception as q_err:
                        print(f"    !! Error checking qualifying segments, falling back to 'Q': {q_err}")
                        sessions_to_process.append('Q')
                elif session_type == 'Sprint Qualifying':
                     # Similar logic for SQ segments - assume SQ1/2/3 for now
                     print("    -> Assuming SQ1, SQ2, SQ3 for Sprint Qualifying.")
                     sessions_to_process.extend(['SQ1', 'SQ2', 'SQ3'])
                else:
                    sessions_to_process.append(session_type) # Add FP, Sprint, R

            # Remove duplicates and maintain rough order
            final_session_list = sorted(list(set(sessions_to_process)), key=lambda x: ['FP1','FP2','FP3','SQ1','SQ2','SQ3','Q1','Q2','Q3','Sprint','R'].index(x) if x in ['FP1','FP2','FP3','SQ1','SQ2','SQ3','Q1','Q2','Q3','Sprint','R'] else 99)
            print(f"  -> Sessions/Segments to process: {final_session_list}")

            # --- Process Non-Race/Sprint Sessions First (FP, Q, SQ) ---
            print(f"  -> Processing non-points sessions (FP, Q, SQ)...")
            qualifying_times_by_driver = {} # Store best Q time per driver
            for session_id in final_session_list:
                if session_id not in ['Sprint', 'R']:
                    # Pass loaded Q laps to avoid reloading if processing Q1/Q2/Q3
                    # laps_to_pass = q_session_laps if session_id in ['Q1', 'Q2', 'Q3'] else None # TODO: Refine if needed
                    processed_data = process_session_results(year, event_name, session_id, save_file=True)
                    # Store qualifying times if processing Q/Q1/Q2/Q3
                    if session_id.startswith('Q') and processed_data:
                        for res in processed_data:
                            driver_code = res.get('driverCode')
                            # Use fastestLapTime for segments, or q3/q2/q1 for parent Q
                            q_time = res.get('fastestLapTime') or res.get('q3Time') or res.get('q2Time') or res.get('q1Time')
                            if driver_code and q_time:
                                # Store the best time found so far for this driver
                                if driver_code not in qualifying_times_by_driver or q_time < qualifying_times_by_driver[driver_code]:
                                     qualifying_times_by_driver[driver_code] = q_time
            # --- DEBUG: Print collected qualifying times ---
            print(f"    -> Collected qualifying times: {qualifying_times_by_driver}")
            # --- Process Sprint Session (if applicable) ---
            is_sprint = is_sprint_weekend(event_format)
            sprint_session_results_data = None # Store processed data
            if is_sprint and 'Sprint' in final_session_list:
                print(f"  -> Processing Sprint session points and results...")
                # Process results and save the file
                sprint_session_results_data = process_session_results(year, event_name, 'Sprint', save_file=True)
                if sprint_session_results_data:
                    # Process points using the returned data (convert to DataFrame first)
                    _process_session_points(pd.DataFrame(sprint_session_results_data), driver_standings, team_standings, is_sprint_session=True)
                else:
                    print(f"    !! No results data returned for Sprint session {event_name}")

            # --- Process Race Session ---
            print(f"  -> Processing Race session points, results, and charts...")
            race_session_results_data = None # Store processed data
            race_session_laps_data = None # Store laps data
            if 'R' in final_session_list:
                try:
                    # Load Race session WITH LAPS
                    race_session = ff1.get_session(year, round_number, 'R')
                    print(f"    -> Loading Race session with laps...")
                    race_session.load(laps=True, telemetry=False, weather=False, messages=False)
                    race_session_raw_results = race_session.results # Raw results for points/wins
                    race_session_laps_data = race_session.laps # Laps for charts/fastest lap

                    if race_session_raw_results is not None and not race_session_raw_results.empty:
                        # Process points from Race using raw results
                        _process_session_points(race_session_raw_results, driver_standings, team_standings, is_sprint_session=False)

                        # --- Count Wins and Podiums (ONLY from Race results) ---
                        print("    -> Counting Wins and Podiums from Race results...")
                        for _, res in race_session_raw_results.iterrows():
                            driver_code = str(res.get('Abbreviation')) if pd.notna(res.get('Abbreviation')) else None
                            team_name = str(res.get('TeamName')) if pd.notna(res.get('TeamName')) else None
                            position = res.get('Position')

                            if not driver_code or not team_name or pd.isna(position): continue
                            position = int(position)

                            if position == 1:
                                driver_standings[driver_code]['wins'] += 1
                                team_standings[team_name]['wins'] += 1
                            if position <= 3:
                                driver_standings[driver_code]['podiums'] += 1
                                team_standings[team_name]['podiums'] += 1

                        # Process Race results format (including fastest lap), but DON'T save yet
                        race_session_results_data = process_session_results(year, event_name, 'R', save_file=False)

                        # Add pole time to the processed race results data
                        if race_session_results_data:
                            pole_sitter_entry = next((res for res in race_session_results_data if res.get('gridPosition') == 1), None)
                            if pole_sitter_entry:
                                pole_driver_code = pole_sitter_entry.get('driverCode')
                                # --- Add Debug Print Here (Pole Sitter ID) ---
                                print(f"    -> Identified pole sitter (GridPos=1) as: {pole_driver_code}")
                                # --- End Debug Print ---
                                if pole_driver_code in qualifying_times_by_driver:
                                    pole_time = qualifying_times_by_driver[pole_driver_code]
                                    pole_sitter_entry['poleLapTimeValue'] = pole_time
                                    print(f"    -> Added pole time for {pole_driver_code}: {pole_time}")
                                else:
                                    print(f"    !! Pole time not found in collected qualifying data for {pole_driver_code}.")
                            else:
                                print("    !! Could not identify pole sitter (GridPosition=1) in processed Race results.")

                            # Now save the final Race results with pole time included
                            race_results_file = races_path / f"{event_slug}_R.json"
                            save_json(race_session_results_data, race_results_file)
                            print(f"    -> Saved final Race results (with pole time) to {race_results_file.name}")
                        else:
                            print("    !! Processing Race results returned no data.")

                    else:
                        print(f"    !! No raw results found for Race session {event_name}")

                    del race_session # Clean up session object
                except Exception as race_err:
                    print(f"    !! Error processing race session for {event_name}: {race_err}")
                    # Attempt to save empty file if race results processing failed entirely
                    race_results_file = races_path / f"{event_slug}_R.json"
                    if not race_results_file.exists():
                        try:
                            save_json([], race_results_file)
                            print(f"    -> Saved empty Race results file due to error.")
                        except Exception as save_err:
                            print(f"    !! Failed to save empty Race results file after error: {save_err}")

            # --- Process Chart Data (using Race session laps if loaded) ---
            if race_session_laps_data is not None and not race_session_laps_data.empty:
                print("    -> Processing chart data for Race...")
                all_drivers_laps = race_session_laps_data['Driver'].unique()
                # Lap Times
                if not lap_times_file.exists():
                    laps_filtered = race_session_laps_data.pick_drivers(all_drivers_laps).pick_accurate().copy()
                    if not laps_filtered.empty:
                        laps_filtered.loc[:, 'LapTimeSeconds'] = laps_filtered['LapTime'].dt.total_seconds()
                        laps_pivot = laps_filtered.pivot_table(index='LapNumber', columns='Driver', values='LapTimeSeconds')
                        laps_pivot = laps_pivot.reset_index()
                        save_json(laps_pivot, lap_times_file)
                # Position Changes
                if not positions_file.exists():
                    pos_data = race_session_laps_data[['LapNumber', 'Driver', 'Position']].dropna(subset=['Position'])
                    if not pos_data.empty:
                        pos_data['Position'] = pos_data['Position'].astype(int)
                        pos_pivot = pos_data.pivot_table(index='LapNumber', columns='Driver', values='Position')
                        pos_pivot = pos_pivot.reset_index()
                        save_json(pos_pivot, positions_file)
                # Tire Strategy
                if not tire_strategy_file.exists():
                    strategy_list = []
                    for drv_code in all_drivers_laps:
                        drv_laps = race_session_laps_data.pick_driver(drv_code)
                        if drv_laps.empty: continue
                        stints_grouped = drv_laps.groupby("Stint")
                        stint_data = []
                        for stint_num, stint_laps in stints_grouped:
                            if stint_laps.empty: continue
                            compound = stint_laps["Compound"].iloc[0]
                            start_lap = stint_laps["LapNumber"].min()
                            end_lap = stint_laps["LapNumber"].max()
                            lap_count = len(stint_laps)
                            stint_data.append({"compound": compound, "startLap": int(start_lap), "endLap": int(end_lap), "lapCount": int(lap_count)})
                        if stint_data:
                            stint_data.sort(key=lambda x: x['startLap'])
                            strategy_list.append({"driver": drv_code, "stints": stint_data})
                    save_json(strategy_list, tire_strategy_file)
                # Session Drivers (using results data from the main race session results)
                if not session_drivers_file.exists() and race_session_results_data: # Use stored R results
                    driver_list = [{"code": str(res['driverCode']), "name": res['fullName'], "team": res['team']}
                                   for res in race_session_results_data if res.get('driverCode')]
                    driver_list.sort(key=lambda x: x['code'])
                    save_json(driver_list, session_drivers_file)
                print(f"    -> Saved chart data for {event_name}.")
            elif 'R' in final_session_list: # Only print warning if Race was supposed to be processed
                 print(f"    !! No lap data found for {event_name} Race, skipping chart data generation.")
            # else: # Don't print if Race wasn't in the list (e.g., error loading schedule)
                 # print(f"    -> Race session not processed, skipping chart data.")


            # Add winner to summary list (only from Race session results)
            if race_session_results_data: # Use the formatted results
                winner = next((res for res in race_session_results_data if res.get('position') == 1), None)
                if winner:
                    all_race_results_summary.append({
                        "year": year, "event": event_name, "round": round_number,
                        "driver": winner.get('fullName', 'N/A'), # Use fullName now
                        "team": winner.get('team', 'N/A'),
                        "teamColor": winner.get('teamColor', 'gray')
                    })

            # --- Cleanup ---
            gc.collect()
            time.sleep(0.2) # Slightly shorter sleep

        # --- Final Standings Calculation & Saving ---
        print("\n--- Calculating Final Standings ---")
        driver_standings_list = []
        for code, driver in driver_standings.items():
            race_points = driver['points'] - driver.get('sprint_points', 0.0)
            driver_data = {
                'code': code,
                'name': driver['name'],
                'team': driver['team'],
                'total_points': driver['points'],
                'race_points': race_points,
                'sprint_points': driver.get('sprint_points', 0.0),
                'wins': driver['wins'],
                'podiums': driver['podiums'],
                'teamColor': get_team_color_name(driver['team'])
            }
            driver_standings_list.append(driver_data)

        team_standings_list = []
        for name, team in team_standings.items():
            race_points = team['points'] - team.get('sprint_points', 0.0)
            team_data = {
                'team': name,
                'total_points': team['points'],
                'race_points': race_points,
                'sprint_points': team.get('sprint_points', 0.0),
                'wins': team['wins'],
                'podiums': team['podiums'],
                'color': team['color'],
                'shortName': name[:3].upper() if name else 'N/A'
            }
            team_standings_list.append(team_data)

        # Calculate points change if season is ongoing
        if is_ongoing:
            print("Calculating points change for ongoing season...")
            for driver_data in driver_standings_list:
                prev_total_points = previous_driver_standings.get(driver_data['code'], {}).get('points', 0.0)
                driver_data['points_change'] = driver_data['total_points'] - prev_total_points
            for team_data in team_standings_list:
                prev_total_points = previous_team_standings.get(team_data['team'], {}).get('points', 0.0)
                team_data['points_change'] = team_data['total_points'] - prev_total_points

        # Sort standings
        driver_standings_list.sort(key=lambda x: (x['total_points'], x['wins'], x['podiums'], x['race_points']), reverse=True)
        for i, driver in enumerate(driver_standings_list):
            driver['rank'] = i + 1

        team_standings_list.sort(key=lambda x: (x['total_points'], x['wins'], x['podiums'], x['race_points']), reverse=True)
        for i, team in enumerate(team_standings_list):
            team['rank'] = i + 1

        # Prepare final standings structure for JSON
        final_driver_standings = [
            {
                'rank': d['rank'], 'code': d['code'], 'name': d['name'], 'team': d['team'],
                'points': d['total_points'], 'wins': d['wins'], 'podiums': d['podiums'],
                'points_change': d.get('points_change'), 'teamColor': d['teamColor']
            } for d in driver_standings_list
        ]
        final_team_standings = [
            {
                'rank': t['rank'], 'team': t['team'], 'points': t['total_points'],
                'wins': t['wins'], 'podiums': t['podiums'],
                'points_change': t.get('points_change'), 'teamColor': t['color'],
                'shortName': t['shortName']
            } for t in team_standings_list
        ]
        final_standings = {"drivers": final_driver_standings, "teams": final_team_standings}

        # Save final standings
        save_json(final_standings, standings_path / "standings.json")
        print(f"Saved final standings for {year}.")

        # Save race results summary (only main races)
        all_race_results_summary.sort(key=lambda x: x['round'])
        save_json(all_race_results_summary, year_cache_path / "race_results.json")
        print(f"Saved race results summary for {year}.")

    except Exception as e:
        print(f"An critical error occurred during {year} processing: {e}")
        import traceback
        traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    target_years = [2023, 2024, 2025] # Add more years if needed
    current_year = datetime.now().year
    if current_year not in target_years: target_years.append(current_year)

    # Process years in reverse order (most recent first)
    for year in sorted(list(set(target_years)), reverse=True):
        process_season_data(year) # This function has internal error handling per year

    print("--- Data processing complete ---") # Ensure this is the final message
