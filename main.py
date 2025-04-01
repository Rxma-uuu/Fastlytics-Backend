import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import numpy as np # Keep numpy for NaN replacement if needed by chance
from dotenv import load_dotenv
import fastf1 as ff1
import pandas as pd # Import fastf1 for schedule
# Re-import data_processing as it's needed again
from api import data_processing
import time # Import time for logging

# --- Configuration ---
load_dotenv()
script_dir = Path(__file__).resolve().parent # Get script directory
DATA_CACHE_PATH = script_dir / "data_cache" # Root for our processed JSON
STATIC_DATA_PATH = script_dir / "static_data" # Root for static JSON files
FASTF1_CACHE_PATH = os.getenv('FASTF1_CACHE_PATH', script_dir / 'cache') # Needed for ff1 schedule
API_KEY_NAME = "X-API-Key" # Standard header name for API keys
API_KEY = os.getenv("FASTLYTICS_API_KEY") # Read API key from environment variable
if not API_KEY:
    print("WARNING: FASTLYTICS_API_KEY environment variable not set. API will be unsecured.")
    # Consider raising an error here if security is mandatory:
    # raise ValueError("FASTLYTICS_API_KEY environment variable is required for security.")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Ensure FastF1 cache is enabled if not already by processor
if not Path(FASTF1_CACHE_PATH).exists():
    os.makedirs(FASTF1_CACHE_PATH)
try:
    ff1.Cache.enable_cache(FASTF1_CACHE_PATH)
except Exception as e:
    print(f"Warning: Could not enable FastF1 cache in main.py: {e}")


app = FastAPI(
    title="Fastlytics Backend API",
    description="API to serve pre-processed and live-processed Formula 1 data.",
    version="0.4.0", # Bump version for new features
)

# --- CORS Configuration ---
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:5173')
origins = [ FRONTEND_URL ]
print(f"Allowing CORS origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security Dependency ---
async def get_api_key(key: str = Depends(api_key_header)):
    """Dependency to validate the API key."""
    if not API_KEY: # If no key is set in the environment, allow access (for local dev maybe)
        print("Allowing request because no API key is configured.")
        return key # Or return None, depending on desired behavior

    if key == API_KEY:
        return key
    else:
        print(f"Unauthorized access attempt with key: {key[:5]}...") # Log attempt without full key
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

# --- Helper Function to Read JSON Cache ---
def read_json_cache(file_path: Path):
    """Reads JSON data from a cache file."""
    if not file_path.is_file():
        print(f"Cache file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            # Handle potential NaN values during load if processor missed them
            return json.load(f, parse_constant=lambda x: None if x == 'NaN' else x)
    except Exception as e:
        print(f"Error reading cache file {file_path}: {e}")
        # Let the endpoint handler raise HTTPException
        return None

# --- API Routes ---
# Apply the security dependency to all routes that need protection
# You can apply it globally using app.include_router(..., dependencies=[Depends(get_api_key)])
# Or apply it per-route as shown below. Applying per-route offers more flexibility.

@app.get("/")
async def read_root():
    # This root endpoint might not need protection
    return {"message": "Welcome to the Fastlytics Backend API"}

# --- Schedule Endpoint ---
@app.get("/api/schedule/{year}", dependencies=[Depends(get_api_key)])
async def get_schedule(year: int):
    """ Retrieves the event schedule for a given year using FastF1. """
    print(f"Received request for schedule: {year}")
    try:
        # Fetch schedule directly using FastF1 (caching handled by FastF1)
        schedule_df = ff1.get_event_schedule(year, include_testing=False)

        # Define date columns to convert
        date_columns = ['EventDate', 'Session1Date', 'Session2Date', 'Session3Date', 'Session4Date', 'Session5Date']

        # Convert Timestamp columns to ISO format strings safely
        for col in date_columns:
            if col in schedule_df.columns:
                # Apply conversion only to valid Timestamps, keep NaT/None as None
                schedule_df[col] = schedule_df[col].apply(lambda x: x.isoformat() if pd.notna(x) and isinstance(x, pd.Timestamp) else None)

        # Convert the DataFrame to a list of dictionaries
        # Note: NaT/None values handled by the apply function above should become null in JSON
        schedule_dict = schedule_df.to_dict(orient='records')
        print(f"Returning schedule for {year} with {len(schedule_dict)} events.")
        return schedule_dict
    except Exception as e:
        print(f"Error fetching schedule for {year}: {e}")
        # Use built-in Exception as per custom instructions
        raise HTTPException(status_code=500, detail=f"Failed to fetch schedule: {e}")


# --- Live Processing Endpoints ---

@app.get("/api/session/drivers", dependencies=[Depends(get_api_key)])
async def get_session_drivers(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(default="R", regex="^[RQSF][P123]?$", description="Session type (R, Q, S, FP1, FP2, FP3)")
):
    """ Retrieves a list of drivers (code, name, team) who participated in a session. """
    print(f"Received request for session drivers: {year}, {event}, {session}")
    try:
        # Cache key needs to handle event name vs round number if needed
        event_slug = event.replace(' ', '_') if isinstance(event, str) and not event.isdigit() else f"Round_{event}"
        cache_file = DATA_CACHE_PATH / str(year) / "charts" / f"{event_slug}_drivers.json" # Assuming processor saves this
        cached_drivers = read_json_cache(cache_file)
        if cached_drivers is not None:
            print(f"Returning cached session drivers for {year} {event} {session}.")
            return cached_drivers

        # If not cached, raise error (rely on processor script)
        print(f"Cache miss for session drivers: {cache_file}")
        raise HTTPException(status_code=404, detail=f"Session drivers data not available for {year} {event} {session}. Run processor script.")
    except Exception as e:
        print(f"Error fetching session drivers: {e}")
        # Consider more specific error codes based on exception type if needed
        raise HTTPException(status_code=500, detail=f"Failed to fetch session drivers: {e}")


@app.get("/api/laptimes", dependencies=[Depends(get_api_key)])
async def get_lap_times(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(default="R", regex="^[RQSF][P123]?$", description="Session type"),
    drivers: list[str] = Query(..., min_length=1, max_length=5, description="List of 1 to 5 driver codes") # Increased max_length to 5
):
    """ Retrieves and compares lap times for one to five drivers. """ # Updated docstring
    print(f"Received request for laptimes: {year}, {event}, {session}, drivers={drivers}")
    if not (1 <= len(drivers) <= 5): # Updated validation check
         raise HTTPException(status_code=400, detail="Please provide 1 to 5 driver codes.")
    try:
        # Cache key needs update if processor saves per-driver data
        event_slug = event.replace(' ', '_') if isinstance(event, str) and not event.isdigit() else f"Round_{event}"
        cache_file = DATA_CACHE_PATH / str(year) / "charts" / f"{event_slug}_laptimes.json"
        cached_laptimes = read_json_cache(cache_file)
        if cached_laptimes is not None:
             # Filter cached data for requested drivers
             filtered_data = [
                 {
                     'LapNumber': lap['LapNumber'],
                     **{drv: lap.get(drv) for drv in drivers if drv in lap}
                 } for lap in cached_laptimes
             ]
             # Check if all requested drivers were found in cache for at least one lap
             if any(all(drv in lap for drv in drivers) for lap in filtered_data):
                  print(f"Returning cached lap times for {drivers} in {year} {event} {session}.")
                  return filtered_data
             # If cache hit but drivers seem missing, still return the filtered data or raise error
             # For simplicity, let's return what we found or raise if nothing matched
             if not filtered_data or not any(any(drv in lap for drv in drivers) for lap in filtered_data):
                  print(f"Cache hit, but requested drivers {drivers} not found in {cache_file}")
                  # Raise 404 as the data for *these specific drivers* isn't in the expected format/file
                  raise HTTPException(status_code=404, detail=f"Lap time data for requested drivers not found in cache for {year} {event} {session}.")
             else:
                  # Return the data filtered from cache even if not all drivers present in every lap dict key
                  print(f"Returning potentially partially filtered cached lap times for {drivers} in {year} {event} {session}.")
                  return filtered_data


        # If cache file itself was not found
        print(f"Cache miss for lap times: {cache_file}")
        raise HTTPException(status_code=404, detail=f"Lap time data not available for {year} {event} {session}. Run processor script.")
    except Exception as e:
        print(f"Error fetching lap times: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch lap times: {e}")


@app.get("/api/telemetry/speed", dependencies=[Depends(get_api_key)])
async def get_telemetry_speed(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(default="R", regex="^[RQSF][P123]?$", description="Session type"),
    driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code"),
    lap: str = Query(default="fastest", description="Lap number (integer) or 'fastest'") # Updated description
):
    """ Retrieves speed telemetry data for a specific driver lap. """
    print(f"Received request for speed telemetry: {year}, {event}, {session}, {driver}, lap={lap}")
    # NOTE: Telemetry is usually NOT pre-processed due to size. Fetch live.
    try:
        # data_processing needs update to handle integer lap numbers
        speed_data_df = data_processing.fetch_and_process_speed_trace(year, event, session, driver, lap)
        if speed_data_df is None or speed_data_df.empty:
             raise HTTPException(status_code=404, detail="Speed telemetry data not found.")
        result_json = speed_data_df.to_dict(orient='records')
        return result_json
    except ValueError as ve: # Catch potential ValueError from int conversion
        print(f"Invalid lap parameter: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid lap parameter: {lap}. Must be 'fastest' or an integer.")
    except Exception as e:
        print(f"Error fetching speed telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch speed telemetry: {e}")


@app.get("/api/telemetry/gear", dependencies=[Depends(get_api_key)])
async def get_telemetry_gear(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(default="R", regex="^[RQSF][P123]?$", description="Session type"),
    driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code"),
    lap: str = Query(default="fastest", description="Lap number (integer) or 'fastest'") # Updated description
):
    """ Retrieves gear telemetry data (X, Y, nGear) for a specific driver lap. """
    print(f"Received request for gear telemetry: {year}, {event}, {session}, {driver}, lap={lap}")
    # NOTE: Telemetry is usually NOT pre-processed due to size. Fetch live.
    try:
        # data_processing needs update to handle integer lap numbers
        gear_data_df = data_processing.fetch_and_process_gear_map(year, event, session, driver, lap)
        if gear_data_df is None or gear_data_df.empty:
             raise HTTPException(status_code=404, detail="Gear telemetry data not found.")
        result_json = gear_data_df.to_dict(orient='records')
        return result_json
    except ValueError as ve: # Catch potential ValueError from int conversion
        print(f"Invalid lap parameter: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid lap parameter: {lap}. Must be 'fastest' or an integer.")
    except Exception as e:
        print(f"Error fetching gear telemetry: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch gear telemetry: {e}")


@app.get("/api/strategy/tires", dependencies=[Depends(get_api_key)])
async def get_tire_strategy(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(default="R", regex="^[RQSF][P123]?$", description="Session type")
):
    """ Retrieves tire stint data for all drivers in a session. """
    print(f"Received request for tire strategy: {year}, {event}, {session}")
    try:
        # Cache key needs update
        event_slug = event.replace(' ', '_') if isinstance(event, str) and not event.isdigit() else f"Round_{event}"
        cache_file = DATA_CACHE_PATH / str(year) / "charts" / f"{event_slug}_tirestrategy.json"
        cached_strategy = read_json_cache(cache_file)
        if cached_strategy is not None:
            print(f"Returning cached tire strategy for {year} {event} {session}.")
            return cached_strategy

        # If not cached, raise error (rely on processor script)
        print(f"Cache miss for tire strategy: {cache_file}")
        raise HTTPException(status_code=404, detail=f"Tire strategy data not available for {year} {event} {session}. Run processor script.")
    except Exception as e:
        print(f"Error fetching tire strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch tire strategy: {e}")


@app.get("/api/lapdata/positions", dependencies=[Depends(get_api_key)])
async def get_lap_positions(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(default="R", regex="^[RS]$", description="Session type (R or S)") # Allow Sprint
):
    """ Retrieves lap-by-lap position data for all drivers in a race or sprint session from cache. """
    print(f"Received request for cached lap positions: {year}, {event}, {session}")
    if session not in ['R', 'S']:
        raise HTTPException(status_code=400, detail="Position data is only available for Race (R) or Sprint (S) sessions.")
    try:
        event_slug = event.replace(' ', '_') if isinstance(event, str) and not event.isdigit() else f"Round_{event}"
        # Cache file name might need session identifier if processor saves separately
        positions_file = DATA_CACHE_PATH / str(year) / "charts" / f"{event_slug}_positions.json" # Assuming R for now
        positions_data = read_json_cache(positions_file)
        if positions_data is None:
            # TODO: Potentially fetch live if cache miss? Or rely on processor.
            raise HTTPException(status_code=404, detail=f"Lap position data not available for {year} {event} {session}. Run processor script.")
        print(f"Returning cached lap positions for {year} {event} {session}.")
        # Data is already pivoted: [{'LapNumber': 1, 'VER': 1, 'LEC': 2, ...}, ...]
        return positions_data
    except HTTPException as http_exc: raise http_exc
    except Exception as e: print(f"Unexpected error: {e}"); raise HTTPException(status_code=500)


# --- Standings & Results Endpoints (Read from Cache) ---

@app.get("/api/standings/drivers", dependencies=[Depends(get_api_key)])
async def get_driver_standings_api( year: int = Query(...) ):
    """ Retrieves pre-calculated driver standings for a given year from cache. """
    start_time = time.time()
    print(f"{start_time:.2f} - REQ START: Driver Standings {year}")
    try:
        standings_file = DATA_CACHE_PATH / str(year) / "standings" / "standings.json"
        standings_data = read_json_cache(standings_file)
        if standings_data is None or 'drivers' not in standings_data:
            print(f"{time.time():.2f} - REQ ERROR: Driver Standings {year} - Not Found")
            raise HTTPException(status_code=404, detail=f"Driver standings not available for {year}. Run processor script.")
        result = standings_data['drivers']
        print(f"{time.time():.2f} - REQ END: Driver Standings {year} ({time.time() - start_time:.3f}s)")
        return result
    except HTTPException as http_exc:
        print(f"{time.time():.2f} - REQ HTTP EXC: Driver Standings {year} ({http_exc.status_code})")
        raise http_exc
    except Exception as e:
        print(f"{time.time():.2f} - REQ UNEXP EXC: Driver Standings {year} - {e}")
        raise HTTPException(status_code=500)

@app.get("/api/standings/teams", dependencies=[Depends(get_api_key)])
async def get_team_standings_api( year: int = Query(...) ):
    """ Retrieves pre-calculated constructor standings for a given year from cache. """
    start_time = time.time()
    print(f"{start_time:.2f} - REQ START: Team Standings {year}")
    try:
        standings_file = DATA_CACHE_PATH / str(year) / "standings" / "standings.json"
        standings_data = read_json_cache(standings_file)
        if standings_data is None or 'teams' not in standings_data:
            print(f"{time.time():.2f} - REQ ERROR: Team Standings {year} - Not Found")
            raise HTTPException(status_code=404, detail=f"Team standings not available for {year}. Run processor script.")
        result = standings_data['teams']
        print(f"{time.time():.2f} - REQ END: Team Standings {year} ({time.time() - start_time:.3f}s)")
        return result
    except HTTPException as http_exc:
        print(f"{time.time():.2f} - REQ HTTP EXC: Team Standings {year} ({http_exc.status_code})")
        raise http_exc
    except Exception as e:
        print(f"{time.time():.2f} - REQ UNEXP EXC: Team Standings {year} - {e}")
        raise HTTPException(status_code=500)


@app.get("/api/results/races", dependencies=[Depends(get_api_key)])
async def get_race_results_summary_api( year: int = Query(...) ):
    """ Retrieves summary of race results (winners) for a given year from cache. """
    start_time = time.time()
    print(f"{start_time:.2f} - REQ START: Race Results Summary {year}")
    try:
        results_file = DATA_CACHE_PATH / str(year) / "race_results.json"
        results_data = read_json_cache(results_file)
        if results_data is None:
            print(f"{time.time():.2f} - REQ ERROR: Race Results Summary {year} - Not Found")
            raise HTTPException(status_code=404, detail=f"Race results summary not available for {year}. Run processor script.")
        print(f"{time.time():.2f} - REQ END: Race Results Summary {year} ({time.time() - start_time:.3f}s)")
        return results_data
    except HTTPException as http_exc:
        print(f"{time.time():.2f} - REQ HTTP EXC: Race Results Summary {year} ({http_exc.status_code})")
        raise http_exc
    except Exception as e:
        print(f"{time.time():.2f} - REQ UNEXP EXC: Race Results Summary {year} - {e}")
        raise HTTPException(status_code=500)

@app.get("/api/sessions", dependencies=[Depends(get_api_key)])
async def get_available_sessions(
    year: int = Query(..., description="Year of the season"),
    event: str = Query(..., description="Event name or Round Number")
):
    """ Retrieves available processed session files (including segments) for a given event. """
    print(f"Received request for available sessions: {year}, {event}")
    try:
        # --- Determine event_slug (consistent with processor) ---
        event_slug = None
        try:
            # Try converting event to int first (for round number)
            round_num = int(event)
            schedule = ff1.get_event_schedule(year, include_testing=False)
            event_row = schedule[schedule['RoundNumber'] == round_num]
            if not event_row.empty:
                event_slug = event_row['EventName'].iloc[0].replace(' ', '_')
            else:
                 raise HTTPException(status_code=404, detail=f"Event round {event} not found for {year}")
        except ValueError:
            # If not an integer, assume it's an event name
            event_slug = event.replace(' ', '_')
            # Optional: Verify event name exists in schedule if needed
            # schedule = ff1.get_event_schedule(year, include_testing=False)
            # if not any(schedule['EventName'] == event):
            #     raise HTTPException(status_code=404, detail=f"Event name '{event}' not found for {year}")

        if not event_slug: # Should not happen if logic above is correct
             raise HTTPException(status_code=400, detail="Could not determine event slug.")

        # --- Find available session files ---
        event_results_path = DATA_CACHE_PATH / str(year) / "races"
        if not event_results_path.exists():
            print(f"No results directory found for {year}")
            return []

        available_sessions = []
        # Define the order and mapping for all possible sessions/segments
        session_order_map = {
            'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3',
            'SQ1': 'Sprint Quali 1', 'SQ2': 'Sprint Quali 2', 'SQ3': 'Sprint Quali 3',
            'Q1': 'Qualifying 1', 'Q2': 'Qualifying 2', 'Q3': 'Qualifying 3',
            'Sprint': 'Sprint Race', 'R': 'Race',
            # Add fallbacks for parent sessions if segments aren't processed
            'SQ': 'Sprint Quali', 'Q': 'Qualifying'
        }
        # Create ordered list based on the map keys
        session_order = list(session_order_map.keys())

        processed_files = set(f.name for f in event_results_path.glob(f"{event_slug}_*.json"))

        for session_id in session_order:
            filename = f"{event_slug}_{session_id}.json"
            if filename in processed_files:
                # Check if we should skip parent Q/SQ if segments exist
                if session_id == 'Q' and any(f"{event_slug}_Q{i}.json" in processed_files for i in [1, 2, 3]):
                    continue # Skip parent Q if Q1/Q2/Q3 exist
                if session_id == 'SQ' and any(f"{event_slug}_SQ{i}.json" in processed_files for i in [1, 2, 3]):
                    continue # Skip parent SQ if SQ1/SQ2/SQ3 exist

                available_sessions.append({
                    "name": session_order_map.get(session_id, session_id), # Use mapping
                    "type": session_id, # Use the specific identifier (FP1, Q1, R etc.)
                })

        print(f"Found available processed sessions for {year} {event_slug}: {[s['type'] for s in available_sessions]}")
        # Sort based on the predefined order
        available_sessions.sort(key=lambda x: session_order.index(x['type']) if x['type'] in session_order else 99)
        return available_sessions
    except HTTPException as http_exc:
        raise http_exc # Re-raise known HTTP exceptions
    except Exception as e:
        print(f"Error fetching available sessions for {year} {event}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Failed to fetch available sessions: {e}")

@app.get("/api/results/race/{year}/{event_slug}", dependencies=[Depends(get_api_key)])
async def get_specific_race_result_api(
    year: int,
    event_slug: str,
    session: str = Query(default="R", regex="^[RQSF][P123]?$")
):
    """ Retrieves detailed results for a specific session from cache. """
    print(f"Received request for specific session results: {year} / {event_slug} / {session}")
    try:
        # Corrected session-specific cache file naming
        session_suffix = f"_{session}" # Always include the session suffix
        race_filename = f"{event_slug.replace('-', '_')}{session_suffix}.json"
        results_file = DATA_CACHE_PATH / str(year) / "races" / race_filename
        print(f"Attempting to read cache file: {results_file}") # Add log for debugging
        results_data = read_json_cache(results_file)
        if results_data is None:
            # TODO: Fetch live if cache miss?
            raise HTTPException(status_code=404, detail=f"Detailed results not found for {year} {event_slug.replace('_', ' ')} session {session}. Run processor script.") # Updated error message
        print(f"Returning detailed results for {year} {event_slug} session {session}")
        return results_data
    except HTTPException as http_exc: raise http_exc
    except Exception as e: print(f"Unexpected error: {e}"); raise HTTPException(status_code=500)


# --- Track Evolution Analysis Endpoint ---
@app.get("/api/analysis/track-evolution", dependencies=[Depends(get_api_key)])
async def get_track_evolution_analysis(
    year: int = Query(..., description="Year of the season"),
    event: str = Query(..., description="Event name or Round Number"),
    session: str = Query(..., regex="^[RQSF][P123]?$", description="Session type (R, Q, S, FP1-3)")
):
    """ Analyzes track evolution based on lap times and track temperature. """
    print(f"Received request for track evolution: {year}, {event}, {session}")
    start_time = time.time()
    try:
        # --- Load Session Data ---
        try:
            ff1_session = ff1.get_session(year, event, session)
            # Load laps with telemetry for potential future enhancements, weather needed for temp
            ff1_session.load(laps=True, weather=True, telemetry=False, messages=False) # Telemetry=False for now to speed up
            laps = ff1_session.laps
            weather_data = ff1_session.weather_data
        except Exception as load_error:
            print(f"FastF1 Error loading session {year} {event} {session}: {load_error}")
            # Use built-in Exception as per custom instructions
            raise HTTPException(status_code=404, detail=f"Session data not available or failed to load: {load_error}")

        if laps is None or laps.empty:
             raise HTTPException(status_code=404, detail="No lap data found for this session.")
        if weather_data is None or weather_data.empty:
             print(f"Warning: No weather data found for {year} {event} {session}. Track temp will be unavailable.")
             # Proceed without weather if necessary, or raise error if temp is critical
             # raise HTTPException(status_code=404, detail="Weather data (for track temp) not found.")


        # --- Process Lap Data ---
        laps_proc = laps.pick_accurate().pick_quicklaps().copy() # Work on a copy
        if laps_proc.empty:
             raise HTTPException(status_code=404, detail="No accurate quick laps found for analysis.")

        laps_proc['LapTimeSeconds'] = laps_proc['LapTime'].dt.total_seconds()

        # Filter outliers (using quantiles as in the example)
        lower_bound = laps_proc['LapTimeSeconds'].quantile(0.05)
        upper_bound = laps_proc['LapTimeSeconds'].quantile(0.95)
        laps_proc = laps_proc[(laps_proc['LapTimeSeconds'] > lower_bound) &
                              (laps_proc['LapTimeSeconds'] < upper_bound)]

        if laps_proc.empty:
             raise HTTPException(status_code=404, detail="No laps remaining after outlier filtering.")

        # --- Calculate Rolling Averages ---
        driver_evolution_data = []
        all_lap_numbers = sorted(laps_proc['LapNumber'].unique())

        for driver in laps_proc['Driver'].unique():
            driver_laps = laps_proc[laps_proc['Driver'] == driver].sort_values(by='LapNumber')
            if len(driver_laps) < 5: # Need enough laps for rolling average
                continue

            # Calculate rolling average (5-lap window, centered)
            rolling_avg = driver_laps['LapTimeSeconds'].rolling(window=5, center=True, min_periods=1).mean()

            # Create data points for the chart { lap: number, time: number | null }
            driver_rolling_laps = []
            # Create a mapping from lap number to the original index for this driver
            lap_to_index_map = pd.Series(driver_laps.index, index=driver_laps['LapNumber'])

            for lap_num in range(int(driver_laps['LapNumber'].min()), int(driver_laps['LapNumber'].max()) + 1):
                # Get the original index for this lap number for this driver, if it exists
                original_index = lap_to_index_map.get(lap_num)
                lap_time_avg = None
                if original_index is not None:
                    # Use the original index to safely get the rolling average value
                    lap_time_avg = rolling_avg.get(original_index)

                driver_rolling_laps.append({
                    "lap": lap_num,
                    "time": round(lap_time_avg, 3) if pd.notna(lap_time_avg) else None
                })


            # Get driver color (handle potential errors if function not available/updated)
            try:
                color = data_processing.get_driver_color_map().get(driver, '#808080') # Use helper if exists
            except AttributeError:
                 # Fallback if helper doesn't exist or fails
                 color_map = {'VER': '#1E41FF', 'PER': '#1E41FF', 'HAM': '#6CD3BF', 'RUS': '#6CD3BF', 'LEC': '#FF2800', 'SAI': '#FF2800', 'NOR': '#FF8700', 'PIA': '#FF8700', 'ALO': '#2F9B90', 'STR': '#2F9B90', 'GAS': '#0090FF', 'OCO': '#0090FF', 'ALB': '#00A0DE', 'SAR': '#00A0DE', 'TSU': '#00287D', 'RIC': '#00287D', 'BOT': '#900000', 'ZHO': '#900000', 'MAG': '#B6BABD', 'HUL': '#B6BABD'}
                 color = color_map.get(driver, '#808080') # Default gray

            driver_evolution_data.append({
                "code": driver,
                "color": color,
                "rollingAverageLaps": driver_rolling_laps
            })

        # --- Process Track Temperature ---
        track_temp_data = []
        if weather_data is not None and not weather_data.empty and 'TrackTemp' in weather_data.columns and 'Time' in weather_data.columns:
            try:
                # Set 'Time' as index BEFORE resampling
                weather_data_indexed = weather_data.set_index('Time')
                # Resample weather data to 1-second intervals and forward fill missing values (use '1s')
                weather_resampled = weather_data_indexed.ffill().resample('1s').ffill()

                # Align temperature with lap start times (approximate)
                laps_with_start = laps_proc[['LapNumber', 'LapStartDate']].dropna().sort_values('LapStartDate')

                for _, lap_row in laps_with_start.iterrows():
                    lap_num = int(lap_row['LapNumber'])
                    start_time_lap = lap_row['LapStartDate']

                    # Find the closest weather timestamp (within a tolerance, e.g., 5 seconds)
                    temp_at_lap_start = weather_resampled['TrackTemp'].asof(start_time_lap, tolerance=pd.Timedelta('5s'))

                    if pd.notna(temp_at_lap_start):
                        track_temp_data.append({
                            "lap": lap_num,
                            "temp": round(temp_at_lap_start, 1)
                        })
                # Ensure unique laps and sort
                track_temp_data = pd.DataFrame(track_temp_data).drop_duplicates(subset=['lap']).sort_values('lap').to_dict('records')

            except Exception as weather_err:
                print(f"Warning: Error processing weather data: {weather_err}. Temp data might be incomplete.")
                track_temp_data = [] # Fallback to empty list

        # --- Prepare Response ---
        response_data = {
            "drivers": driver_evolution_data,
            "trackTemperature": track_temp_data
            # TODO: Add stint comparison and compound evolution results here if implemented
        }

        print(f"Track evolution analysis complete for {year} {event} {session} ({time.time() - start_time:.3f}s)")
        return response_data

    except HTTPException as http_exc:
        print(f"HTTP Exception during track evolution: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        print(f"Error during track evolution analysis for {year} {event} {session}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # Use built-in Exception as per custom instructions
        raise HTTPException(status_code=500, detail=f"Failed to analyze track evolution: {e}")


# --- Optional: Run directly with uvicorn for simple testing ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
