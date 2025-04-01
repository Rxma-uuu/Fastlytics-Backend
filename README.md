# Fastlytics Backend API

This directory contains the backend components for the Fastlytics application.

## Components

*   **`main.py`**: A FastAPI application that serves Formula 1 data. It reads pre-processed data from the `data_cache` directory and can also fetch live data using the FastF1 library.
*   **`processor.py`**: A Python script that uses the FastF1 library to fetch historical F1 data (schedules, results, standings, lap times, etc.), processes it, and saves it into JSON files within the `data_cache` directory. This pre-processing step speeds up API responses for historical data.
*   **`data_cache/`**: Directory where the `processor.py` script stores the processed JSON data files, organized by year.
*   **`cache/`**: Directory used by the FastF1 library to cache raw API responses, reducing redundant external requests.
*   **`.env`**: Environment variables file. Should contain `FASTLYTICS_API_KEY` for securing the API endpoints and optionally `FRONTEND_URL` for CORS configuration.
*   **`requirements.txt`**: (Optional - If you create one) Lists Python dependencies.

## Setup & Usage

1.  **Dependencies:** Ensure you have Python installed. It's recommended to use a virtual environment. Install dependencies (primarily `fastapi`, `uvicorn`, `fastf1`, `pandas`, `numpy`, `python-dotenv`):
    ```bash
    pip install fastapi uvicorn "fastf1[full]" pandas numpy python-dotenv requests requests_cache
    # Or install from requirements.txt if provided
    # pip install -r requirements.txt
    ```
2.  **Environment Variables:** Create a `.env` file in this directory and add your desired `FASTLYTICS_API_KEY`.
    ```
    FASTLYTICS_API_KEY=your_secret_api_key_here
    FRONTEND_URL=http://localhost:5173 # Or your frontend's URL
    ```
3.  **Run Data Processor:** Execute the processor script to fetch and cache historical data. This can take some time initially.
    ```bash
    python processor.py
    ```
4.  **Run API Server:** Start the FastAPI server using Uvicorn.
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API will then be accessible, typically at `http://localhost:8000`. The `--reload` flag automatically restarts the server when code changes are detected (useful for development).
