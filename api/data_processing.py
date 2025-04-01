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
# Removed specific FastF1 exception imports
# from fastf1.exceptions import SessionNotAvailableError, DataNotLoadedError, FastF1Error

# --- Configuration ---
FASTF1_CACHE_PATH = os.getenv('FASTF1_CACHE_PATH', './cache')
DATA_CACHE_PATH = Path("./data_cache") # Keep for processor part

if not os.path.exists(FASTF1_CACHE_PATH):
    os.makedirs(FASTF1_CACHE_PATH)
ff1.Cache.enable_cache(FASTF1_CACHE_PATH)

# Helper to get team color mapping
def get_team_color_name(team_name: str | None) -> str:
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
    if 'sauber' in simple_name: return 'alfaromeo'
    if 'racingbulls' in simple_name or 'alphatauri' in simple_name: return 'alphatauri'
    return 'gray'

def save_json(data, file_path: Path):
    """Saves data to a JSON file."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            if isinstance(data, pd.DataFrame):
                data = data.replace({np.nan: None})
                json.dump(data.to_dict(orient='records'), f, indent=2, default=str)
            elif isinstance(data, (list, dict)):
                 json.dump(data, f, indent=2, default=str)
            else:
                 print(f" -> Unsupported data type for JSON saving: {type(data)}")
                 return
        print(f" -> Data successfully saved to {file_path}")
    except Exception as e:
        print(f" -> Error saving JSON to {file_path}: {e}")
        import traceback
        traceback.print_exc()

# --- Data Processing Functions (for Live API Endpoints) ---

def fetch_session_drivers(year: int, event: str, session_type: str) -> list[dict] | None:
    """ Fetches the list of drivers who participated in a session. """
    print(f"Fetching drivers for {year} {event} {session_type}")
    try:
        session = ff1.get_session(year, event, session_type)
        # Removed results=True from load(), results are accessed directly via session.results
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        print("Session loaded for driver list.")

        driver_list = []
        if session.results is not None and not session.results.empty:
            driver_list = [{"code": row['Abbreviation'], "name": row['LastName'], "team": row['TeamName']}
                           for _, row in session.results.iterrows() if pd.notna(row['Abbreviation'])]
        elif session.laps is not None and not session.laps.empty:
             drivers_info = session.laps[['Driver', 'Team', 'FullName']].drop_duplicates(subset=['Driver'])
             if not drivers_info.empty:
                 driver_list = [{"code": row['Driver'], "name": row['FullName'], "team": row['Team']}
                                for _, row in drivers_info.iterrows() if pd.notna(row['Driver'])]

        if not driver_list:
             print("No drivers found in session data.")
             return None

        driver_list.sort(key=lambda x: x['code'])
        print(f"Found {len(driver_list)} drivers.")
        return driver_list
    except Exception as e: # Catch general Exception
        print(f"Error fetching session drivers: {e}")
        raise e # Re-raise for the API endpoint to handle


def fetch_and_process_laptimes_multi(year: int, event: str, session_type: str, driver_codes: list[str]) -> pd.DataFrame | None:
    """ Fetches and processes lap times for multiple drivers (2 or 3). """
    print(f"Processing lap times for {driver_codes} - {year} {event} {session_type}")
    if not (2 <= len(driver_codes) <= 3): raise ValueError("Supports 2 or 3 drivers.")
    try:
        session = ff1.get_session(year, event, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps.pick_drivers(driver_codes).pick_accurate()
        if laps.empty: return None
        laps_copy = laps.copy()
        laps_copy['LapTimeSeconds'] = laps_copy['LapTime'].dt.total_seconds()
        laps_filtered = laps_copy[['LapNumber', 'Driver', 'LapTimeSeconds']]
        lap_comparison_df = laps_filtered.pivot_table(index='LapNumber', columns='Driver', values='LapTimeSeconds').reset_index()
        for code in driver_codes:
            if code not in lap_comparison_df.columns: lap_comparison_df[code] = pd.NA
        final_df = lap_comparison_df[['LapNumber'] + driver_codes]
        final_df = final_df.replace({np.nan: None})
        print(f"Successfully processed {len(final_df)} laps for comparison.")
        return final_df
    except Exception as e: print(f"Error processing multi-lap times: {e}"); raise e


def fetch_and_process_speed_trace(year: int, event: str, session_type: str, driver_code: str, lap_identifier: str) -> pd.DataFrame | None:
    """ Fetches telemetry for a specific lap and processes speed data. """
    print(f"Processing speed trace for {driver_code} lap {lap_identifier} - {year} {event} {session_type}")
    try:
        session = ff1.get_session(year, event, session_type)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = session.laps.pick_driver(driver_code)
        if laps.empty: return None
        target_lap = None
        if lap_identifier.lower() == 'fastest':
            target_lap = laps.pick_fastest()
            if target_lap is None or pd.isna(target_lap['LapTime']): target_lap = laps.iloc[-1] if not laps.empty else None
        else:
            try: lap_num = int(lap_identifier); lap_row = laps[laps['LapNumber'] == lap_num]; target_lap = lap_row.iloc[0] if not lap_row.empty else None
            except (ValueError, IndexError): return None
        if target_lap is None: return None
        telemetry = target_lap.get_car_data(pad=1)
        if telemetry.empty or 'Speed' not in telemetry.columns: return None
        if 'Distance' not in telemetry.columns: telemetry = telemetry.add_distance()
        speed_df = telemetry[['Distance', 'Speed']].copy().dropna(subset=['Distance', 'Speed']).replace({np.nan: None})
        print(f"Successfully processed speed trace for lap {getattr(target_lap, 'LapNumber', 'N/A')}. Records: {len(speed_df)}")
        return speed_df
    except Exception as e: print(f"Error processing speed trace: {e}"); raise e


def fetch_and_process_gear_map(year: int, event: str, session_type: str, driver_code: str, lap_identifier: str) -> pd.DataFrame | None:
    """ Fetches telemetry for a specific lap and processes X, Y, Gear data. """
    print(f"Processing gear map for {driver_code} lap {lap_identifier} - {year} {event} {session_type}")
    try:
        session = ff1.get_session(year, event, session_type)
        session.load(laps=True, telemetry=True, weather=False, messages=False)
        laps = session.laps.pick_driver(driver_code)
        if laps.empty: return None
        target_lap = None
        if lap_identifier.lower() == 'fastest':
            target_lap = laps.pick_fastest()
            if target_lap is None or pd.isna(target_lap['LapTime']): target_lap = laps.iloc[-1] if not laps.empty else None
        else:
            try: lap_num = int(lap_identifier); lap_row = laps[laps['LapNumber'] == lap_num]; target_lap = lap_row.iloc[0] if not lap_row.empty else None
            except (ValueError, IndexError): return None
        if target_lap is None: return None
        telemetry = target_lap.get_telemetry()
        required_cols = ['X', 'Y', 'nGear']
        if telemetry.empty or not all(col in telemetry.columns for col in required_cols):
             print(f"Required telemetry data (X, Y, nGear) not found for lap {getattr(target_lap, 'LapNumber', 'N/A')}.")
             return None
        gear_df = telemetry[required_cols].copy().dropna().reset_index(drop=True)
        gear_df['nGear'] = gear_df['nGear'].astype(int)
        gear_df = gear_df.replace({np.nan: None})
        print(f"Successfully processed gear map data for lap {getattr(target_lap, 'LapNumber', 'N/A')}. Records: {len(gear_df)}")
        return gear_df
    except Exception as e: print(f"Error processing gear map: {e}"); raise e


def fetch_and_process_tire_strategy(year: int, event: str, session_type: str) -> list[dict] | None:
    """ Fetches lap data and extracts tire stint information. """
    print(f"Processing tire strategy - {year} {event} {session_type}")
    try:
        session = ff1.get_session(year, event, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps.empty: return None
        drivers = laps['Driver'].unique()
        if len(drivers) == 0: return None
        strategy_list = []
        for drv_code in drivers:
            try:
                drv_laps = laps.pick_driver(drv_code)
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
            except Exception as inner_e: print(f"Error processing stints for driver {drv_code}: {inner_e}")
        print(f"Successfully processed tire strategy for {len(strategy_list)} drivers.")
        return strategy_list
    except Exception as e: print(f"Error processing tire strategy: {e}"); raise e


# --- Standings & Results Processing Functions (for processor.py) ---
# Keep the more specific error handling in calculate_standings and fetch_race_results
# as they are run offline by processor.py, where specific errors are more useful.

def calculate_standings(year: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """ Calculates driver and team standings for a given year by processing race results. """
    # ... (Keep existing implementation with specific error handling) ...
    print(f"Calculating standings for {year}...")
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        completed_races = schedule[schedule['EventDate'].dt.tz_localize('UTC') < datetime.now(timezone.utc)]
        if completed_races.empty: return None, None
        driver_points = defaultdict(lambda: {'points': 0.0, 'wins': 0, 'podiums': 0, 'team': '', 'name': '', 'code': ''})
        team_points = defaultdict(lambda: {'points': 0.0, 'wins': 0, 'podiums': 0, 'team': '', 'color': 'gray'})
        print(f"Processing {len(completed_races)} completed events for {year}...")
        for index, event in completed_races.iterrows():
            print(f" Loading results for {event['EventName']}...")
            try:
                session = ff1.get_session(year, event['RoundNumber'], 'R')
                session.load(laps=False, telemetry=False, weather=False, messages=False) # Removed results=True
                if session.results is None or session.results.empty: continue
                for _, res in session.results.iterrows():
                    try:
                        driver_code = res.get('Abbreviation'); team_name = res.get('TeamName'); points = res.get('Points', 0.0)
                        position_val = res.get('Position'); status = str(res.get('Status', 'Unknown'))
                        last_name = res.get('LastName', 'N/A'); first_name = res.get('FirstName', '')
                        full_name = f"{first_name} {last_name}".strip()
                        driver_code = str(driver_code) if pd.notna(driver_code) else None
                        team_name = str(team_name) if pd.notna(team_name) else None
                        points = float(points) if pd.notna(points) else 0.0
                        try: position = int(float(position_val)) if pd.notna(position_val) and str(position_val).replace('.','',1).isdigit() else None
                        except (ValueError, TypeError): position = None
                        if not driver_code or not team_name: continue
                        driver_points[driver_code]['points'] += points; team_points[team_name]['points'] += points
                        driver_points[driver_code]['team'] = team_name; driver_points[driver_code]['name'] = full_name if full_name else last_name
                        driver_points[driver_code]['code'] = driver_code; team_points[team_name]['team'] = team_name
                        team_points[team_name]['color'] = get_team_color_name(team_name)
                        if status.startswith(('Finished', '+')) or status == 'Running':
                            if position == 1: driver_points[driver_code]['wins'] += 1; team_points[team_name]['wins'] += 1
                            if position is not None and position <= 3: driver_points[driver_code]['podiums'] += 1; team_points[team_name]['podiums'] += 1
                    except Exception as row_err: print(f"    !! Error processing row: {row_err}")
                del session; gc.collect(); time.sleep(0.1) # Shorter delay
            except Exception as race_err: print(f"  -> Error processing session {event['EventName']}: {race_err}") # Catch general exception here
        driver_df = pd.DataFrame(list(driver_points.values())) if driver_points else None
        teams_df = pd.DataFrame(list(team_points.values())) if team_points else None
        if driver_df is not None and not driver_df.empty:
            driver_df = driver_df.sort_values(by=['points', 'wins', 'podiums', 'code'], ascending=[False, False, False, True]).reset_index(drop=True)
            driver_df['rank'] = driver_df.index + 1
        if teams_df is not None and not teams_df.empty:
            teams_df = teams_df.sort_values(by=['points', 'wins', 'podiums', 'team'], ascending=[False, False, False, True]).reset_index(drop=True)
            teams_df['rank'] = teams_df.index + 1
            teams_df['shortName'] = teams_df['team'].apply(lambda x: x[:3].upper() if x else 'N/A')
        print(f"Finished calculating standings for {year}.")
        return driver_df, teams_df
    except Exception as e: print(f"Error calculating standings for {year}: {e}"); raise e


def fetch_race_results(year: int) -> list[dict] | None:
    """ Fetches basic race results (winner) for a given year. """
    print(f"Fetching race results summary for {year}...")
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        completed_races = schedule[schedule['EventDate'].dt.tz_localize('UTC') < datetime.now(timezone.utc)]
        if completed_races.empty: return []
        results_list = []
        print(f"Processing {len(completed_races)} completed events for {year}...")
        for index, event in completed_races.iterrows():
            try:
                session = ff1.get_session(year, event['RoundNumber'], 'R')
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                if session.results is None or session.results.empty: continue
                winner = session.results[session.results['Position'] == 1].iloc[0] if not session.results[session.results['Position'] == 1].empty else None
                if winner is not None:
                     results_list.append({"year": year, "event": event['EventName'], "round": event['RoundNumber'],
                                          "driver": winner.get('LastName', 'N/A'), "team": winner.get('TeamName'),
                                          "teamColor": get_team_color_name(winner.get('TeamName'))})
                del session; gc.collect(); time.sleep(0.1)
            except Exception as race_err: print(f"  -> Error fetching winner for {event['EventName']}: {race_err}") # Catch general exception here
        results_list.sort(key=lambda x: x['round'])
        print(f"Finished fetching race results summary for {year}. Found {len(results_list)} results.")
        return results_list
    except Exception as e: print(f"Error fetching race results summary for {year}: {e}"); raise e

# --- Main Execution (for processor.py) ---
# (Keep __main__ block as is)
# ...
