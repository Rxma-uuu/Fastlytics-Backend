import os
import json
from pathlib import Path  # Keep pathlib.Path for filesystem operations
import fastapi  # Import the full fastapi module
from fastapi import FastAPI, HTTPException, Query, Depends, status  # Remove Path alias import
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import numpy as np  # Keep numpy for NaN replacement if needed by chance
from dotenv import load_dotenv
import fastf1 as ff1
import pandas as pd  # Import fastf1 for schedule
# Re-import data_processing as it's needed again
from api import data_processing
import time  # Import time for logging

# --- Configuration ---
load_dotenv()
script_dir = Path(__file__).resolve().parent  # Get script directory
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
    # Temporarily remove regex to test validation
    session: str = Query(..., description="Session type (e.g., R, Q, S, FP1, FP2, FP3)")
    # session: str = Query(default="R", regex="^[RQSF][P123]?$", description="Session type (R, Q, S, FP1, FP2, FP3)")
):
    """ Retrieves a list of drivers (code, name, team) who participated in a session. """
    print(f"Received request for session drivers: {year}, {event}, {session}")
    try:
        # Handle event name vs round number AND replace hyphens/spaces consistently
        if isinstance(event, str) and not event.isdigit():
            # Replace spaces AND hyphens with underscores for event names
            event_slug_corrected = event.replace(' ', '_').replace('-', '_').lower()
        else:
            # Assume it's a round number, format consistently (lowercase)
            event_slug_corrected = f"round_{event}".lower() # Processor uses lowercase round_
        # Assuming processor saves driver lists in charts dir now
        cache_file = DATA_CACHE_PATH / str(year) / "charts" / f"{event_slug_corrected}_drivers.json"
        print(f"Attempting to read cache file: {cache_file}") # Add log for debugging path
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


@app.get("/api/laps/driver", dependencies=[Depends(get_api_key)])
async def get_driver_lap_numbers(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(..., description="Session type"),
    driver: str = Query(..., min_length=3, max_length=3, description="3-letter driver code")
):
    """ Retrieves a list of valid lap numbers for a specific driver in a session. """
    print(f"Received request for lap numbers: {year}, {event}, {session}, driver={driver}")
    try:
        lap_numbers = data_processing.fetch_driver_lap_numbers(year, event, session, driver)
        if lap_numbers is None: # Should return [] if no laps, None might indicate error upstream
            raise HTTPException(status_code=404, detail="Could not retrieve lap numbers.")
        return {"laps": lap_numbers} # Return as a JSON object with a 'laps' key
    except Exception as e:
        print(f"Error fetching lap numbers: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch lap numbers: {e}")


@app.get("/api/laptimes", dependencies=[Depends(get_api_key)])
async def get_lap_times(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(..., description="Session type"),
    drivers: list[str] = Query(..., min_length=1, max_length=5, description="List of 1 to 5 driver codes") # Increased max_length to 5
):
    """ Retrieves and compares lap times for one to five drivers. """ # Updated docstring
    print(f"Received request for laptimes: {year}, {event}, {session}, drivers={drivers}")
    if not (1 <= len(drivers) <= 5): # Updated validation check
         raise HTTPException(status_code=400, detail="Please provide 1 to 5 driver codes.")
    try:
        # Handle event name vs round number AND replace hyphens/spaces consistently
        if isinstance(event, str) and not event.isdigit():
            # Replace spaces AND hyphens with underscores for event names, then lowercase
            event_slug_corrected = event.replace(' ', '_').replace('-', '_').lower()
        else:
            # Assume it's a round number, format consistently (lowercase)
            event_slug_corrected = f"round_{event}".lower() # Processor uses lowercase round_
        cache_file = DATA_CACHE_PATH / str(year) / "charts" / f"{event_slug_corrected}_laptimes.json"
        print(f"Attempting to read cache file: {cache_file}") # Add log for debugging path
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
             # For Simplicity, let's return what we found or raise if nothing matched
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
    session: str = Query(..., description="Session type"),
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
    session: str = Query(..., description="Session type"),
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


@app.get("/api/comparison/sectors", dependencies=[Depends(get_api_key)])
async def get_sector_comparison(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(..., description="Session type"),
    driver1: str = Query(..., min_length=3, max_length=3, description="3-letter code for driver 1"),
    driver2: str = Query(..., min_length=3, max_length=3, description="3-letter code for driver 2"),
    lap1: str = Query(default="fastest", description="Lap identifier for driver 1 (number or 'fastest')"), # Add lap1 param
    lap2: str = Query(default="fastest", description="Lap identifier for driver 2 (number or 'fastest')")  # Add lap2 param
):
    """ Retrieves sector/segment comparison data between two drivers for specific laps, including SVG paths. """
    print(f"Received request for sector comparison: {year}, {event}, {session}, {driver1} (Lap {lap1}) vs {driver2} (Lap {lap2})") # Update log
    if driver1 == driver2:
        raise HTTPException(status_code=400, detail="Please select two different drivers.")
    # NOTE: This is live processing, not typically cached.
    try:
        # Pass lap identifiers to the processing function
        comparison_data = data_processing.fetch_and_process_sector_comparison(
            year, event, session, driver1, driver2, lap1_identifier=lap1, lap2_identifier=lap2
        )
        if comparison_data is None:
             raise HTTPException(status_code=404, detail=f"Sector comparison data could not be generated for the specified laps (Lap {lap1} vs Lap {lap2}). Check if laps exist and have telemetry.")
        return comparison_data
    except Exception as e:
        print(f"Error fetching sector comparison: {e}")
        # Use built-in Exception as per custom instructions
        raise HTTPException(status_code=500, detail=f"Failed to fetch sector comparison: {e}")


@app.get("/api/strategy/tires", dependencies=[Depends(get_api_key)])
async def get_tire_strategy(
    year: int = Query(..., description="Year of the season", example=2023),
    event: str = Query(..., description="Event name or Round Number", example="Bahrain Grand Prix"),
    session: str = Query(..., description="Session type")
):
    """ Retrieves tire stint data for all drivers in a session. """
    print(f"Received request for tire strategy: {year}, {event}, {session}")
    try:
        # Handle event name vs round number AND replace hyphens/spaces consistently
        if isinstance(event, str) and not event.isdigit():
            # Replace spaces AND hyphens with underscores for event names, then lowercase
            event_slug_corrected = event.replace(' ', '_').replace('-', '_').lower()
        else:
            # Assume it's a round number, format consistently (lowercase)
            event_slug_corrected = f"round_{event}".lower() # Processor uses lowercase round_
        cache_file = DATA_CACHE_PATH / str(year) / "charts" / f"{event_slug_corrected}_tirestrategy.json"
        print(f"Attempting to read cache file: {cache_file}") # Add log for debugging path
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
    session: str = Query(..., description="Session type (R or S)")
):
    """ Retrieves lap-by-lap position data for all drivers in a race or sprint session from cache. """
    print(f"Received request for cached lap positions: {year}, {event}, {session}")
    if session not in ['R', 'S']:
        raise HTTPException(status_code=400, detail="Position data is only available for Race (R) or Sprint (S) sessions.")
    try:
        # Handle event name vs round number AND replace hyphens/spaces consistently
        if isinstance(event, str) and not event.isdigit():
            # Replace spaces AND hyphens with underscores for event names, then lowercase
            event_slug_corrected = event.replace(' ', '_').replace('-', '_').lower()
        else:
            # Assume it's a round number, format consistently (lowercase)
            event_slug_corrected = f"round_{event}".lower() # Processor uses lowercase round_
        # Ensure the slug used for the filename is lowercase to match filesystem conventions
        # Cache file name might need session identifier if processor saves separately
        positions_file = DATA_CACHE_PATH / str(year) / "charts" / f"{event_slug_corrected}_positions.json" # Use corrected slug
        print(f"Attempting to read cache file: {positions_file}") # Add log for debugging path
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
        # --- Determine event_slug (consistent with processor, ensure lowercase for file ops) ---
        event_slug_raw = None
        try:
            # Try converting event to int first (for round number)
            round_num = int(event)
            schedule = ff1.get_event_schedule(year, include_testing=False)
            event_row = schedule[schedule['RoundNumber'] == round_num]
            if not event_row.empty:
                event_slug_raw = event_row['EventName'].iloc[0].replace(' ', '_')
            else:
                 raise HTTPException(status_code=404, detail=f"Event round {event} not found for {year}")
        except ValueError:
            # If not an integer, assume it's an event name
            event_slug_raw = event.replace(' ', '_')
            # Optional: Verify event name exists in schedule if needed
            # schedule = ff1.get_event_schedule(year, include_testing=False)
            # if not any(schedule['EventName'] == event):
            #     raise HTTPException(status_code=404, detail=f"Event name '{event}' not found for {year}")

        if not event_slug_raw: # Should not happen if logic above is correct
             raise HTTPException(status_code=400, detail="Could not determine event slug.")

        # Use lowercase slug for filesystem operations
        event_slug_lower = event_slug_raw.lower()

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

        # Use lowercase slug for globbing
        processed_files = set(f.name for f in event_results_path.glob(f"{event_slug_lower}_*.json"))
        print(f"Globbing for files matching: {event_slug_lower}_*.json in {event_results_path}") # Debug log
        print(f"Found files: {processed_files}") # Debug log

        for session_id in session_order:
            # Use lowercase slug for filename check
            filename = f"{event_slug_lower}_{session_id}.json"
            if filename in processed_files:
                # Check if we should skip parent Q/SQ if segments exist (using lowercase slug)
                if session_id == 'Q' and any(f"{event_slug_lower}_q{i}.json" in processed_files for i in [1, 2, 3]):
                    continue # Skip parent Q if Q1/Q2/Q3 exist
                if session_id == 'SQ' and any(f"{event_slug_lower}_sq{i}.json" in processed_files for i in [1, 2, 3]):
                    continue # Skip parent SQ if SQ1/SQ2/SQ3 exist

                available_sessions.append({
                    "name": session_order_map.get(session_id, session_id), # Use mapping
                    "type": session_id, # Use the specific identifier (FP1, Q1, R etc.)
                })

        print(f"Found available processed sessions for {year} {event_slug_lower}: {[s['type'] for s in available_sessions]}")
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
    session: str = Query(...)
):
    """ Retrieves results for a specific race or session. """
    print(f"Received request for specific race result: {year}, {event_slug}, {session}")
    try:
        # Ensure event_slug is properly formatted (lowercase with underscores)
        event_slug_lower = event_slug.lower().replace('-', '_')
        
        # Use the races directory to fetch session-specific results
        results_file = DATA_CACHE_PATH / str(year) / "races" / f"{event_slug_lower}_{session}.json"
        print(f"Looking for race results file: {results_file}")
        
        results_data = read_json_cache(results_file)
        if results_data is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Results not available for {year} {event_slug} {session}. Run processor script."
            )
        
        return results_data
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error fetching specific race results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch race results: {e}")
