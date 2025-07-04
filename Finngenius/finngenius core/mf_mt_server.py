# server/combined_app.py

import os
import time
import warnings
import json
import pickle
import traceback
import random
import math # For checking NaN
from datetime import datetime, timedelta

# Data/ML Libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

# Web Framework & Utilities
from flask import Flask, jsonify, request
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

# Suppress warnings (optional)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Configuration ---
# Load environment variables from .env file (should be in the server directory)
load_dotenv()

# --- !! IMPORTANT: Set these paths correctly !! ---
# Consider moving these to your .env file as well for flexibility
CSV_PATH = os.getenv('CSV_PATH', '/Users/amankumar/Desktop/ff/frontend/mutual_funds_india.csv') # Default if not in .env
MODEL_PKL_PATH = os.getenv('MODEL_PKL_PATH', '/Users/amankumar/Desktop/ff/frontend/illustrative_fund_finder_kaggle.pkl') # Default if not in .env
SERVER_PORT = int(os.getenv('PORT', 5005)) # Default port 5000 if not in .env

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_LLM_FINANCE = 'llama3-8b-8192' # Used for both stock analysis and jargon buster

# Caching Configuration (for yfinance)
CACHE_DURATION_SECONDS = 45 # How long to cache yfinance data

# --- Initialize Flask App ---
app = Flask(__name__)
# Enable CORS for your specific frontend origins during development/production
# Use "*" for local testing ONLY if necessary, but specific origins are safer.
CORS(app, resources={r"/api/*": {"origins": ["http://127.0.0.1:3000", "http://127.0.0.1:3001"]}})
# Alternatively, for broader local dev: CORS(app)
print("Flask app initialized.")
print(f"CORS enabled for origins: http://127.0.0.1:3000, http://127.0.0.1:3001")


# --- Global Variables ---
groq_client = None
# Stock Data Cache
ticker_cache = {} # { 'ticker_symbol': {'timestamp': ..., 'data': ...} }
# Mutual Fund Data & Model
mf_df = None # Renamed from df to avoid potential confusion
knn_model_bundle = None
knn_preprocessor = None
knn_model = None
knn_original_data = None
# IMPORTANT: These feature lists MUST exactly match those used when CREATING the PKL file
knn_numeric_features = ['fund_rating', 'return_1yr', 'return_3yr', 'return_5yr']
knn_categorical_features = ['risk_type', 'category']
# Quiz Data
quiz_data = [
    {
        "id": 1,
        "question": "What does NAV stand for in the context of Mutual Funds?",
        "options": ["Net Asset Value", "Nominal Asset Value", "Net Average Value", "National Asset Value"],
        "answer": "Net Asset Value"
    },
    {
        "id": 2,
        "question": "What is a SIP?",
        "options": ["Systematic Investment Plan", "Single Investment Payment", "Strategic Investment Portfolio", "Simple Income Plan"],
        "answer": "Systematic Investment Plan"
    },
    {
        "id": 3,
        "question": "Which type of mutual fund primarily invests in stocks of companies?",
        "options": ["Debt Fund", "Equity Fund", "Liquid Fund", "Gilt Fund"],
        "answer": "Equity Fund"
    },
    {
        "id": 4,
        "question": "ELSS funds offer tax benefits under which section of the Income Tax Act?",
        "options": ["Section 80C", "Section 80D", "Section 10(10D)", "Section 24B"],
        "answer": "Section 80C"
    },
    {
        "id": 5,
        "question": "What does 'Expense Ratio' represent?",
        "options": ["The fund's profit percentage", "The annual fee charged by the fund house", "The risk level of the fund", "The dividend paid to investors"],
        "answer": "The annual fee charged by the fund house"
    },
    {
        "id": 6,
        "question": "Which organisation regulates Mutual Funds in India?",
        "options": ["RBI (Reserve Bank of India)", "SEBI (Securities and Exchange Board of India)", "IRDAI (Insurance Regulatory and Development Authority of India)", "PFRDA (Pension Fund Regulatory and Development Authority)"],
        "answer": "SEBI (Securities and Exchange Board of India)"
    },
    {
        "id": 7,
        "question": "A fund that invests in a mix of stocks and bonds is typically called a:",
        "options": ["Sector Fund", "Index Fund", "Hybrid Fund", "Thematic Fund"],
        "answer": "Hybrid Fund"
    },
    {
        "id": 8,
        "question": "What is the main difference between a 'Direct Plan' and a 'Regular Plan' of a mutual fund?",
        "options": ["Investment strategy", "Fund manager", "Expense ratio (due to distributor commissions)", "Minimum investment amount"],
        "answer": "Expense ratio (due to distributor commissions)"
    },
    {
        "id": 9,
        "question": "What does AMC stand for in Mutual Funds?",
        "options": ["Account Management Charge", "Asset Management Company", "Average Monthly Cost", "Annual Maintenance Contract"],
        "answer": "Asset Management Company"
    },
    {
        "id": 10,
        "question": "KYC, often required for mutual fund investments, stands for:",
        "options": ["Know Your Customer", "Keep Your Capital", "Key Yield Calculation", "Knowledge Yields Capital"],
        "answer": "Know Your Customer"
    },
    {
        "id": 11, # Added one more
        "question": "Which type of fund is generally considered the lowest risk among mutual funds?",
        "options": ["Equity Fund (Small Cap)", "Thematic Fund", "Liquid Fund", "Sector Fund"],
        "answer": "Liquid Fund"
    }
]


# --- Initialization and Resource Loading ---
def initialize_resources():
    """Loads all necessary external resources and initializes clients."""
    global groq_client, mf_df, knn_model_bundle, knn_preprocessor, knn_model, knn_original_data
    print("\n--- Initializing Resources ---")
    assets_loaded_successfully = True

    # 1. Initialize Groq Client (Common for both features)
    print("Attempting to initialize Groq client...")
    if not GROQ_API_KEY:
        print("\n******************************************************")
        print("CRITICAL ERROR: GROQ_API_KEY not found in environment variables or .env file.")
        print("Please ensure a .env file exists in the 'server' directory with GROQ_API_KEY=your_key")
        print("AI-based features (Stock Analysis, Jargon Buster) will FAIL.")
        print("******************************************************\n")
        groq_client = None
        assets_loaded_successfully = False # Mark as critical failure
    else:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            # Test connection (optional but good)
            # groq_client.models.list() # Simple API call to check authentication
            print("Groq client initialized successfully.")
        except Exception as e:
            print(f"ERROR: Error initializing Groq client: {e}")
            traceback.print_exc()
            groq_client = None
            assets_loaded_successfully = False # Mark as critical failure

    # 2. Load Mutual Fund Data (CSV)
    print(f"\nAttempting to load Mutual Fund CSV from: {CSV_PATH}")
    try:
        if not os.path.exists(CSV_PATH):
            print(f"ERROR: Mutual Fund CSV file not found at {CSV_PATH}")
            mf_df = None
            assets_loaded_successfully = False # Critical for MF features
        else:
            df_temp = pd.read_csv(CSV_PATH, sep=',')
            print(f"MF CSV loaded. Initial shape: {df_temp.shape}")
            expected_cols_subset = ['Mutual Fund Name', 'category', 'risk_type', 'fund_rating', 'return_1yr']
            if df_temp.shape[1] <= 1 or not all(col in df_temp.columns for col in expected_cols_subset):
                 print(f"ERROR: MF CSV parsing likely failed or critical columns missing.")
                 print(f"       Found columns: {df_temp.columns.tolist()}")
                 mf_df = None
                 assets_loaded_successfully = False
            else:
                print("MF CSV parsing seems okay. Proceeding with cleaning...")
                if df_temp.columns[0].startswith('Unnamed:'):
                    df_temp = df_temp.drop(columns=[df_temp.columns[0]])
                    print("Dropped 'Unnamed: 0' column from MF data.")
                numeric_cols = ['fund_rating', 'return_1yr', 'return_3yr', 'return_5yr']
                for col in numeric_cols:
                    if col in df_temp.columns:
                        # More robust cleaning: handle potential strings, commas, hyphens etc.
                        df_temp[col] = df_temp[col].astype(str).str.replace(',', '', regex=False).str.strip()
                        df_temp[col] = df_temp[col].replace(['-', '%', 'N/A', 'NA', '', 'nan', 'None'], np.nan, regex=False)
                        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
                print(f"Numeric conversion applied to MF columns: {numeric_cols}")

                # Check and potentially scale return columns (IF they are percentages like 9.5, not 0.095)
                return_cols = ['return_1yr', 'return_3yr', 'return_5yr']
                for col in return_cols:
                    if col in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp[col]):
                        # Heuristic: If max value > 1.5 (i.e., 150%), assume it's percentage, needs division
                        if df_temp[col].max(skipna=True) > 1.5:
                             print(f"Rescaling MF '{col}' (dividing by 100). Max found: {df_temp[col].max(skipna=True)}")
                             df_temp[col] = df_temp[col] / 100.0
                        else:
                             print(f"MF '{col}' seems to be already in decimal format. No rescaling applied.")

                # Check and potentially scale fund_rating (IF it's out of 50 or similar)
                if 'fund_rating' in df_temp.columns and pd.api.types.is_numeric_dtype(df_temp['fund_rating']):
                     if df_temp['fund_rating'].max(skipna=True) > 6: # Assuming ratings are max 5
                         print(f"Rescaling MF 'fund_rating' (dividing by 10). Max found: {df_temp['fund_rating'].max(skipna=True)}")
                         df_temp['fund_rating'] = df_temp['fund_rating'] / 10.0
                     else:
                         print(f"MF 'fund_rating' seems to be within expected scale (<=6). No rescaling applied.")

                text_cols_to_lower = ['Mutual Fund Name', 'category', 'risk_type']
                for col in text_cols_to_lower:
                     if col in df_temp.columns:
                         df_temp[col] = df_temp[col].astype(str).str.lower().str.strip()
                         # Replace multiple spaces with single space
                         df_temp[col] = df_temp[col].replace(r'\s+', ' ', regex=True)
                print(f"Lowercased and stripped MF text columns: {text_cols_to_lower}")

                mf_df = df_temp.copy()
                print("Mutual Fund data cleaning and preparation complete.")
                print(f"Final `mf_df` shape: {mf_df.shape}")

    except Exception as e:
        print(f"ERROR: Error during MF data load/prep: {e}")
        traceback.print_exc()
        mf_df = None
        assets_loaded_successfully = False # Critical for MF features

    # 3. Load KNN Model Bundle (for Illustrative Finder)
    print(f"\nAttempting to load KNN model from: {MODEL_PKL_PATH}")
    if not assets_loaded_successfully or mf_df is None: # Skip if prior critical load failed
        print("Skipping KNN model load due to previous critical errors (Groq or MF Data).")
        knn_model_bundle = None
    else:
        try:
            if not os.path.exists(MODEL_PKL_PATH):
                 print(f"WARNING: KNN Model PKL file not found at {MODEL_PKL_PATH}. Illustrative Finder will be disabled.")
                 knn_model_bundle = None
            else:
                with open(MODEL_PKL_PATH, 'rb') as file:
                    knn_model_bundle_loaded = pickle.load(file)
                print("KNN PKL file loaded.")

                required_keys = ['preprocessor', 'nn_model', 'original_data_subset']
                if not isinstance(knn_model_bundle_loaded, dict) or not all(key in knn_model_bundle_loaded for key in required_keys):
                    print(f"ERROR: Loaded KNN model bundle is missing required keys ({required_keys}). Finder disabled.")
                    knn_model_bundle = None
                else:
                    print("KNN bundle structure seems okay. Checking components...")
                    temp_preprocessor = knn_model_bundle_loaded.get('preprocessor')
                    temp_model = knn_model_bundle_loaded.get('nn_model')
                    temp_original_data = knn_model_bundle_loaded.get('original_data_subset')

                    # Validate components
                    valid_preprocessor = hasattr(temp_preprocessor, 'transform')
                    valid_model = hasattr(temp_model, 'kneighbors')
                    valid_data = isinstance(temp_original_data, pd.DataFrame) and not temp_original_data.empty

                    if not (valid_preprocessor and valid_model and valid_data):
                        print("ERROR: One or more components in the KNN bundle are invalid (missing methods or empty data). Finder disabled.")
                        knn_model_bundle = None
                    else:
                        # Check if the required columns for KNN are present in the loaded original data
                        required_knn_cols = ['Mutual Fund Name'] + knn_numeric_features + knn_categorical_features
                        if not all(col in temp_original_data.columns for col in required_knn_cols):
                            print(f"ERROR: KNN model's original data subset missing required columns. Needed: {required_knn_cols}. Found: {temp_original_data.columns.tolist()}. Finder disabled.")
                            knn_model_bundle = None
                        else:
                            print("KNN components validated successfully.")
                            knn_preprocessor = temp_preprocessor
                            knn_model = temp_model
                            # IMPORTANT: Ensure the data used by KNN is cleaned consistently with mf_df
                            knn_original_data = temp_original_data.copy()
                            knn_text_cols_to_lower = ['Mutual Fund Name', 'category', 'risk_type']
                            for col in knn_text_cols_to_lower:
                                if col in knn_original_data.columns:
                                    knn_original_data[col] = knn_original_data[col].astype(str).str.lower().str.strip()
                                    knn_original_data[col] = knn_original_data[col].replace(r'\s+', ' ', regex=True)

                            print(f"Lowercased relevant text columns in KNN original data.")
                            print(f"KNN model bundle loaded successfully. Original data shape for KNN: {knn_original_data.shape}")
                            # Keep the bundle itself if needed elsewhere, though components are now globals
                            knn_model_bundle = knn_model_bundle_loaded

        except (pickle.UnpicklingError, EOFError) as pe:
             print(f"ERROR: Error unpickling KNN model file {MODEL_PKL_PATH}. File might be corrupted or incompatible: {pe}")
             traceback.print_exc()
             knn_model_bundle = None
        except Exception as e:
            print(f"ERROR: Unexpected error loading KNN model PKL file: {e}")
            traceback.print_exc()
            knn_model_bundle = None

    print("\n--- Resource Initialization Finished ---")

    if groq_client is None:
        print("CRITICAL WARNING: Groq client failed to initialize. AI features unavailable.")
    if mf_df is None:
        print("CRITICAL WARNING: Mutual Fund data failed to load. MF features unavailable.")
    if knn_model_bundle is None:
         print("WARNING: KNN Model bundle failed to load or validate. Illustrative Finder unavailable.")

    # Return True if core components loaded, False otherwise (might adjust definition of "core")
    return groq_client is not None and mf_df is not None


# --- Helper Functions ---

# Helper for formatting numbers in analysis/responses
def format_value(val, is_price=True, is_int=False):
    """Formats numbers for display, handling None/NaN."""
    if val is None or (isinstance(val, (float, np.number)) and math.isnan(val)):
        return 'N/A'
    try:
        if is_int: return f"{int(val):,}"
        if is_price: return f"{float(val):.2f}"
        return str(val)
    except (ValueError, TypeError):
        return 'N/A'

# --- Stock Data Feature Functions ---

def get_yf_data(ticker_symbol):
    """Fetches data from yfinance for stocks, using cache."""
    now = time.time()
    ticker_symbol = ticker_symbol.upper() # Standardize ticker

    # Check cache first
    if ticker_symbol in ticker_cache:
        cached_entry = ticker_cache[ticker_symbol]
        if now - cached_entry['timestamp'] < CACHE_DURATION_SECONDS:
            print(f"Cache hit for {ticker_symbol}")
            return cached_entry['data'] # Return cached data

    print(f"Fetching fresh data for {ticker_symbol} from yfinance...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        if not info or not info.get('symbol'): # Basic check if info is valid
             print(f"Warning: Received empty or invalid info dict for {ticker_symbol}")
             # Attempt fallback to history for price
             try:
                  hist_fallback = ticker.history(period="5d", interval="1d")
                  if not hist_fallback.empty:
                       last_close = hist_fallback['Close'].iloc[-1]
                       # Construct minimal info dict if possible
                       info = {'previousClose': last_close,
                               'shortName': info.get('shortName', ticker_symbol), # Try to keep name if available
                               'symbol': ticker_symbol}
                       # Try to get other fields if they exist in the original partial info
                       info['currentPrice'] = info.get('currentPrice')
                       info['regularMarketPrice'] = info.get('regularMarketPrice')
                       info['regularMarketOpen'] = info.get('regularMarketOpen') or info.get('open')
                       info['dayLow'] = info.get('dayLow')
                       info['dayHigh'] = info.get('dayHigh')
                       info['regularMarketVolume'] = info.get('regularMarketVolume') or info.get('volume')
                       print(f"Using fallback previousClose ({last_close:.2f}) from history as base info.")
                  else:
                       raise ValueError("No valid info or fallback history found.")
             except Exception as fallback_err:
                  print(f"Error in fallback history fetch for {ticker_symbol}: {fallback_err}")
                  # Use a very minimal structure indicating failure but allowing analysis attempt
                  info = {'symbol': ticker_symbol, 'shortName': ticker_symbol, 'error': 'Failed to fetch primary info'}
                  print(f"Could not retrieve valid info or fallback for {ticker_symbol}")
        else:
             print(f"Successfully fetched info for {ticker_symbol}.")


        # Fetch historical data for charts
        hist_period = "5d"
        hist_interval = "15m" # Try granular first
        history = None
        try:
            history = ticker.history(period=hist_period, interval=hist_interval, auto_adjust=False)
            if history.empty:
                 print(f"Warning: No 15m data for {ticker_symbol} in last 5d. Falling back to daily.")
                 hist_period="1mo" # Longer period for daily view
                 hist_interval = "1d"
                 history = ticker.history(period=hist_period, interval=hist_interval, auto_adjust=False)
                 if history.empty:
                      print(f"Warning: No daily data found for {ticker_symbol} in last 1mo either.")
                      # Proceed without chart data
            # Check if timezone info exists, localize if needed, convert to UTC for consistency
            if history is not None and not history.empty and history.index.tz is not None:
                print(f"History index has timezone ({history.index.tz}), converting to UTC.")
                history.index = history.index.tz_convert('UTC')
            elif history is not None and not history.empty:
                print(f"History index has no timezone, assuming UTC.")
                # history.index = history.index.tz_localize('UTC') # Optional: Force UTC if needed

        except Exception as hist_error:
            print(f"Warning: Error fetching history ({hist_period}/{hist_interval}) for {ticker_symbol}: {hist_error}. Trying daily fallback...")
            hist_period="1mo"
            hist_interval = "1d"
            try:
                 history = ticker.history(period=hist_period, interval=hist_interval, auto_adjust=False)
                 if history.empty:
                      print(f"Warning: No daily data for {ticker_symbol} even with fallback.")
                 elif history.index.tz is not None:
                      print(f"Daily history index has timezone ({history.index.tz}), converting to UTC.")
                      history.index = history.index.tz_convert('UTC')
                 else:
                      print(f"Daily history index has no timezone, assuming UTC.")
            except Exception as hist_error_daily:
                 print(f"Error fetching daily history fallback for {ticker_symbol}: {hist_error_daily}")
                 history = None # Ensure history is None if all attempts fail

        # Process history IF it was successfully fetched and not empty
        history_list = []
        if history is not None and not history.empty:
            history = history.copy() # Avoid SettingWithCopyWarning
            history.reset_index(inplace=True)
            time_column = None
            # Find the correct time column ('Datetime' or 'Date') - case-insensitive check
            potential_time_cols = [col for col in history.columns if col.lower() in ['datetime', 'date']]
            if potential_time_cols:
                 time_column = potential_time_cols[0] # Take the first match
                 print(f"Using time column: '{time_column}'")
            else:
                print(f"Error: Could not find standard time column in history for {ticker_symbol}. Columns: {history.columns}")

            if time_column:
                # Ensure the time column is datetime type before conversion
                try:
                    history[time_column] = pd.to_datetime(history[time_column], errors='coerce', utc=True) # Ensure UTC
                    # Convert Timestamp to Unix timestamp (seconds)
                    # Use .view('int64') // 10**9 for nanosecond precision to seconds
                    history['unix_time'] = history[time_column].astype(np.int64) // 10**9
                    history = history.dropna(subset=['unix_time']) # Remove rows where time conversion failed
                    history['unix_time'] = history['unix_time'].astype(int)

                    # Select relevant columns and rename for lightweight charts format
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    if all(col in history.columns for col in required_cols):
                        history_selected = history[['unix_time'] + required_cols].rename(
                            columns={'unix_time': 'time', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
                        )

                        # Convert numeric columns to float, coercing errors
                        for col in ['open', 'high', 'low', 'close']:
                            history_selected[col] = pd.to_numeric(history_selected[col], errors='coerce')

                        # Filter out rows with None/NaN in critical fields AFTER conversion
                        history_selected = history_selected.dropna(subset=['time', 'open', 'high', 'low', 'close'])
                        # Ensure numeric types are standard float (not object)
                        for col in ['open', 'high', 'low', 'close']:
                            history_selected[col] = history_selected[col].astype(float)


                        history_list = history_selected.to_dict('records')
                        print(f"Processed {len(history_list)} history records for chart.")
                    else:
                        print(f"Error: Missing required OHLC columns in history for {ticker_symbol}. Found: {history.columns}")
                except Exception as time_conv_err:
                    print(f"Error processing time column '{time_column}' for {ticker_symbol}: {time_conv_err}")

            # Else: Time column error already printed

        # --- Prepare final data structure ---
        data = {
            "info": info,
            "history": history_list # Use the processed (potentially empty) list
        }
        # Update cache
        ticker_cache[ticker_symbol] = {'timestamp': now, 'data': data}
        print(f"Successfully processed and cached data for {ticker_symbol}")
        return data

    except yf.exceptions.YFinanceTickerError as yf_err:
         print(f"TickerError for {ticker_symbol}: {yf_err}. Check if ticker symbol is valid.")
         # Optionally cache a 'not found' state? For now, just return None.
         return None
    except Exception as e:
        print(f"General error in get_yf_data for {ticker_symbol}: {e}")
        traceback.print_exc()
        # Clear cache entry on error to force refetch next time
        if ticker_symbol in ticker_cache:
             del ticker_cache[ticker_symbol]
        return None # Indicate failure

def get_groq_stock_analysis(ticker_symbol, yf_data):
    """Generates stock analysis text using Groq based on yfinance data."""
    if not groq_client:
        return "Groq client not initialized. Cannot generate analysis."
    if not yf_data or not yf_data.get('info') or yf_data['info'].get('error'): # Check for fetch error marker
        return "Insufficient or errored data provided to generate analysis."

    info = yf_data['info']
    # Safely get price info, providing defaults
    # Prefer 'regularMarketPrice' as it's often more reliable during market hours than 'currentPrice'
    price = info.get('regularMarketPrice') or info.get('currentPrice')
    prev_close = info.get('previousClose')
    day_open = info.get('regularMarketOpen') or info.get('open')
    day_low = info.get('dayLow')
    day_high = info.get('dayHigh')
    volume = info.get('regularMarketVolume') or info.get('volume')

    # Use prev close if current price is missing but prev close exists (e.g., after hours)
    if price is None and prev_close is not None:
         price = prev_close

    # Format values for the prompt, handling None/NaN gracefully using helper
    price_str = format_value(price)
    prev_close_str = format_value(prev_close)
    day_open_str = format_value(day_open)
    day_low_str = format_value(day_low)
    day_high_str = format_value(day_high)
    volume_str = format_value(volume, is_price=False, is_int=True)

    change_str = "N/A"
    percent_change_str = "N/A"
    try:
        # Ensure both price and prev_close are valid numbers before calculating change
        if price is not None and not math.isnan(float(price)) and \
           prev_close is not None and not math.isnan(float(prev_close)) and float(prev_close) != 0:
            price_f = float(price)
            prev_close_f = float(prev_close)
            change = price_f - prev_close_f
            percent_change = (change / prev_close_f) * 100
            change_str = f"{change:+.2f}" # Add sign explicitly
            percent_change_str = f"{percent_change:+.2f}%" # Add sign explicitly
    except (ValueError, TypeError, ZeroDivisionError) as calc_err:
        print(f"Warning: Could not calculate change for {ticker_symbol}: {calc_err}")
        pass # Keep as N/A if conversion or calculation fails

    short_name = info.get('shortName', ticker_symbol)

    prompt = f"""
    Analyze the current market situation for {short_name} ({ticker_symbol}) based ONLY on the following data points:
    - Current/Last Price: {price_str}
    - Previous Close: {prev_close_str}
    - Day's Open: {day_open_str}
    - Day's High: {day_high_str}
    - Day's Low: {day_low_str}
    - Change from Previous Close: {change_str} ({percent_change_str})
    - Volume: {volume_str}

    Provide a brief (2-4 sentences) market analysis. Focus on:
    1.  The stock's current price relative to its daily range (high/low) and the previous close.
    2.  The sentiment indicated by the price change (positive, negative, or neutral).
    3.  Mention the trading volume if available and note if it seems particularly high or low (if context allows, otherwise just state it).

    **Important Constraints:**
    *   **Stick to the provided data ONLY.** Do not invent trends, news, or reasons.
    *   **Do NOT give financial advice.** Do not suggest buying, selling, or holding.
    *   **Be objective and factual.** Describe what the numbers show.
    *   **Keep it concise.**
    """

    print(f"Sending stock analysis request to Groq for {ticker_symbol}")
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a concise financial data analyst providing objective summaries based *only* on the given data points. You do not give advice."},
                {"role": "user", "content": prompt},
            ],
            model=MODEL_LLM_FINANCE,
            temperature=0.3, # Slightly more deterministic
            max_tokens=150,
        )
        analysis = chat_completion.choices[0].message.content
        print(f"Received stock analysis from Groq for {ticker_symbol}")
        return analysis.strip() if analysis else "Analysis could not be generated."
    except Exception as e:
        print(f"Error calling Groq API for {ticker_symbol} stock analysis: {e}")
        traceback.print_exc()
        return "Error generating analysis via Groq."

# --- Mutual Fund Feature Functions ---

def explain_term_with_groq(term_or_question):
    """Uses Groq LLM to explain a financial term or answer a conceptual question (MF context)."""
    if groq_client is None:
        return "Error: Groq client not initialized. Cannot provide explanation."

    system_prompt = """
    You are an AI assistant specializing in explaining Indian Mutual Fund concepts and terminology in simple, clear terms.
    Your goal is to provide accurate, factual definitions and explanations suitable for a beginner.

    **RULES:**
    1.  **Focus on Explanation:** Directly answer the user's question or define the term provided.
    2.  **Simplicity:** Use easy-to-understand language. Avoid excessive financial jargon. Use analogies or simple examples if helpful.
    3.  **Context:** Assume the context is Indian Mutual Funds unless specified otherwise.
    4.  **Factual Accuracy:** Ensure information is correct.
    5.  **NO FINANCIAL ADVICE:** Explicitly state you cannot give financial advice. Do NOT recommend specific funds, AMCs, strategies, or investment actions. Do not give opinions on whether something is "good" or "bad" for investment.
    6.  **Unknown Terms:** If the term is obscure, ambiguous, or you genuinely don't know, state that you cannot provide information on that specific topic and suggest rephrasing or asking about a standard financial term.
    7.  **Conciseness:** Be brief and to the point. Aim for 2-4 sentences for simple definitions, perhaps a short paragraph for concepts.
    """
    try:
        print(f"Sending explanation request to Groq for: '{term_or_question[:100]}'") # Log truncated term
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please explain clearly and simply: {term_or_question}"} # Make it explicit
            ],
            model=MODEL_LLM_FINANCE, # Use the defined model
            temperature=0.2, # More factual
            max_tokens=250,
            top_p=0.9,
            stop=None
        )
        response = chat_completion.choices[0].message.content
        print(f"Received explanation from Groq.")
        # Add a disclaimer if not already present (simple check)
        if "cannot provide financial advice" not in response.lower() and "no financial advice" not in response.lower():
             response += "\n\n*Disclaimer: This information is for educational purposes only and does not constitute financial advice.*"
        return response.strip()
    except Exception as e:
        print(f"ERROR: Error communicating with Groq API for explanation: {e}")
        traceback.print_exc()
        return "Sorry, I encountered an error trying to get an explanation from the AI service."

def find_illustrative_funds_knn(risk_profile_name, num_results=5):
    """
    Finds illustrative fund examples similar to a generic risk profile using the loaded KNN model.
    """
    # Check prerequisites
    if knn_model_bundle is None or knn_preprocessor is None or knn_model is None or knn_original_data is None:
        return None, "Error: The Illustrative Finder model or its components are currently unavailable. Please try again later."
    if mf_df is None: # mf_df might be needed for fallback medians if knn_original_data has NaNs
         print("Warning: Main MF DataFrame (`mf_df`) is unavailable, fallback medians might be less accurate.")
         # Allow proceeding, but KNN data itself is primary


    print(f"Finding illustrative examples for profile: {risk_profile_name}")

    # Define representative target profiles (using *lowercase* for categorical values matching cleaned data)
    # Use median values from the *KNN's original data* for numeric defaults first.
    target_profiles = {}
    try:
        # Calculate medians carefully, handling potential empty slices
        def get_median(df, col, category=None):
            data_slice = df
            if category and 'category' in df.columns:
                 data_slice = df[df['category'] == category]
            if data_slice.empty or col not in data_slice.columns:
                return np.nan # Return NaN if slice is empty or column missing
            return data_slice[col].median()

        median_rating = get_median(knn_original_data, 'fund_rating')

        target_profiles = {
            "conservative": {
                'risk_type': 'low risk', # Match expected values in knn_original_data
                'category': 'debt',     # Match expected values in knn_original_data
                'fund_rating': median_rating,
                'return_1yr': get_median(knn_original_data, 'return_1yr', 'debt'),
                'return_3yr': get_median(knn_original_data, 'return_3yr', 'debt'),
                'return_5yr': get_median(knn_original_data, 'return_5yr', 'debt'),
            },
            "moderate": {
                # Adjust based on your data's common moderate risk/category values
                'risk_type': 'moderate risk', # Or 'moderately high risk', check knn_original_data['risk_type'].unique()
                'category': 'hybrid',
                'fund_rating': median_rating,
                'return_1yr': get_median(knn_original_data, 'return_1yr', 'hybrid'),
                'return_3yr': get_median(knn_original_data, 'return_3yr', 'hybrid'),
                'return_5yr': get_median(knn_original_data, 'return_5yr', 'hybrid'),
            },
            "aggressive": {
                'risk_type': 'very high risk', # Match expected values
                'category': 'equity',         # Match expected values
                'fund_rating': median_rating,
                'return_1yr': get_median(knn_original_data, 'return_1yr', 'equity'),
                'return_3yr': get_median(knn_original_data, 'return_3yr', 'equity'),
                'return_5yr': get_median(knn_original_data, 'return_5yr', 'equity'),
            }
        }
    except KeyError as e:
         print(f"ERROR: Missing expected column '{e}' in knn_original_data needed for profile definition.")
         return None, f"Internal error: Cannot define target profile due to missing data column '{e}'."
    except Exception as e:
        print(f"ERROR: Unexpected error calculating medians for target profiles: {e}")
        traceback.print_exc()
        return None, "Internal error: Could not define target profiles."

    # Clean input profile name
    profile_key = risk_profile_name.lower().strip()

    if profile_key not in target_profiles:
        valid_profiles = ', '.join([p.capitalize() for p in target_profiles.keys()])
        return None, f"Error: Invalid risk profile name '{risk_profile_name}'. Please choose from: {valid_profiles}."

    target_profile_dict = target_profiles[profile_key]

    # --- Fill any NaN medians with global median from `mf_df` as fallback ---
    print("Checking for NaN values in target profile definition...")
    needs_fallback = False
    for key, value in target_profile_dict.items():
        if pd.isna(value):
             needs_fallback = True
             print(f"Warning: Median for '{key}' in profile '{profile_key}' (category-specific from KNN data) resulted in NaN.")
             # Try fallback from the main mf_df
             if mf_df is not None and key in mf_df.columns and pd.api.types.is_numeric_dtype(mf_df[key]):
                  fallback_median = mf_df[key].median()
                  if pd.notna(fallback_median):
                      print(f"         Using global median fallback from `mf_df`: {fallback_median:.4f}")
                      target_profile_dict[key] = fallback_median
                  else:
                      print(f"         Warning: Global median for '{key}' in `mf_df` is also NaN. Defaulting to 0.")
                      target_profile_dict[key] = 0.0
             elif key in knn_categorical_features:
                 # Should not happen if profile definition is correct, but handle anyway
                 print(f"         Warning: Categorical feature '{key}' is unexpectedly NaN. Using mode from KNN data or default.")
                 try: target_profile_dict[key] = knn_original_data[key].mode()[0]
                 except: target_profile_dict[key] = 'unknown' # Absolute fallback
             else:
                  print(f"         ERROR: Cannot find fallback for numeric '{key}'. Defaulting to 0.")
                  target_profile_dict[key] = 0.0 # Default to 0 if no fallback possible

    if needs_fallback:
        print(f"Final target profile values after fallback checks: {target_profile_dict}")
    else:
        print(f"Using target profile values (no NaNs found): {target_profile_dict}")


    try:
        # Create a DataFrame with the target profile, ensuring column order matches preprocessor expectation
        target_df = pd.DataFrame([target_profile_dict])
        # Reorder columns exactly as expected by the preprocessor stored in the PKL
        # The order is implicitly defined by how the ColumnTransformer was created.
        # We rely on knn_numeric_features + knn_categorical_features having the correct order.
        required_cols_order = knn_numeric_features + knn_categorical_features
        try:
            target_df_ordered = target_df[required_cols_order]
        except KeyError as e:
            print(f"ERROR: Mismatch between target profile keys and expected KNN features. Missing: {e}")
            print(f"       Target profile keys: {target_profile_dict.keys()}")
            print(f"       Required order: {required_cols_order}")
            return None, "Internal Error: Feature mismatch during profile preparation."

        print("Preprocessing target profile...")
        target_processed = knn_preprocessor.transform(target_df_ordered)
        print(f"Target profile preprocessed shape: {target_processed.shape}")

        # Find neighbors
        # Request slightly more in case the 'closest' is an exact match to a hypothetical entry (unlikely but possible)
        n_neighbors_to_request = num_results + 1
        print(f"Finding {n_neighbors_to_request} nearest neighbors...")
        distances, indices = knn_model.kneighbors(target_processed, n_neighbors=n_neighbors_to_request)

        # Exclude self if the target profile somehow perfectly matches an entry (distance=0) - usually not needed for generic profiles
        # Assuming the first result might be too close/identical, start from index 0 or 1?
        # For generic profiles, the 0th index should be fine.
        start_index = 0
        neighbor_indices = indices[0][start_index : num_results + start_index]
        neighbor_distances = distances[0][start_index : num_results + start_index]

        # Get the actual fund data from the original (cleaned) KNN data subset
        similar_funds_df = knn_original_data.iloc[neighbor_indices].copy()
        similar_funds_df['similarity_distance'] = neighbor_distances # Add distance for context

        # --- Format results for frontend display ---
        display_columns_map = {
             # Internal Column Name : Display Name
             'Mutual Fund Name': 'Fund Name',
             'category': 'Category',
             'risk_type': 'Risk Type',
             'fund_rating': 'Rating',
             'return_1yr': '1yr Return (%)',
             'return_3yr': '3yr Return (%)',
             'return_5yr': '5yr Return (%)',
             'similarity_distance': 'Similarity Score' # Lower is more similar
        }
        # Select only the columns we want to display, if they exist
        columns_to_select = [col for col in display_columns_map.keys() if col in similar_funds_df.columns]
        results_df_display = similar_funds_df[columns_to_select].copy()

        # Replace any remaining Pandas NaNs with None for JSON compatibility
        results_df_display = results_df_display.where(pd.notnull(results_df_display), None)

        # Convert DataFrame rows to list of dictionaries
        results_list = results_df_display.to_dict('records')

        # Format the values within the dictionaries nicely
        formatted_results = []
        for record in results_list:
            formatted_record = {}
            for internal_col_name, display_name in display_columns_map.items():
                 if internal_col_name in record:
                    value = record[internal_col_name]
                    formatted_value = 'N/A' # Default
                    if value is not None:
                        try:
                            # Apply specific formatting based on column
                            if internal_col_name == 'fund_rating':
                                formatted_value = f"{float(value):.1f} / 5.0" # Assumes rating is 0-5 scale
                            elif 'return' in internal_col_name:
                                formatted_value = f"{float(value) * 100:.1f}" # Convert decimal return to percentage string
                            elif internal_col_name == 'similarity_distance':
                                formatted_value = f"{float(value):.4f}" # Show more precision for distance
                            elif internal_col_name in ['Mutual Fund Name', 'category', 'risk_type']:
                                formatted_value = str(value).title() # Capitalize words
                            else:
                                formatted_value = value # Keep other values as is
                        except (ValueError, TypeError) as format_err:
                            print(f"Warning: Could not format value '{value}' for col '{internal_col_name}': {format_err}")
                            # Keep 'N/A' if formatting fails
                    formatted_record[display_name] = formatted_value
                 else:
                      # If a mapped column wasn't even in the selection (shouldn't happen with check above, but safe)
                      formatted_record[display_name] = 'N/A'
            formatted_results.append(formatted_record)

        print(f"Found {len(formatted_results)} illustrative examples.")
        return formatted_results, None # Return list and no error

    except AttributeError as ae:
         # Common if preprocessor or model methods don't exist (e.g., loaded object isn't right type)
         print(f"ERROR: Attribute error during KNN processing (maybe model/preprocessor issue?): {ae}")
         traceback.print_exc()
         return None, f"Internal Error: Model component issue. Please try again later."
    except ValueError as ve:
         # E.g., if feature shapes mismatch during transform/kneighbors
         print(f"ERROR: Value error during KNN processing (maybe feature mismatch?): {ve}")
         traceback.print_exc()
         return None, f"Internal Error: Data mismatch during similarity check. Please try again later."
    except Exception as e:
        print(f"ERROR: Unexpected error during KNN search/processing for profile '{risk_profile_name}': {e}")
        traceback.print_exc()
        return None, f"An unexpected error occurred while finding similar funds. Please try again later."


# --- API Endpoints ---

# == Stock Data Endpoint ==
@app.route('/api/asset_data/<ticker_symbol>', methods=['GET'])
def get_asset_data(ticker_symbol):
    """Endpoint to get yfinance data and Groq analysis for a stock ticker."""
    print(f"\n--- Received request on /api/asset_data/{ticker_symbol} ---")
    if not ticker_symbol:
        return jsonify({"error": "Ticker symbol is required."}), 400

    ticker_symbol = ticker_symbol.upper() # Standardize

    # Fetch data using the helper (includes caching)
    yf_data = get_yf_data(ticker_symbol)

    if not yf_data: # Check if fetch failed entirely
        print(f"Error: Fetch failed for {ticker_symbol}")
        return jsonify({"error": f"Could not fetch data for {ticker_symbol} from source. Check symbol or try again later."}), 404 # Not Found or 503 Service Unavailable?
    if not yf_data.get('info') or yf_data.get('info',{}).get('error'): # Check if info part is missing or marked as error
        print(f"Error: Incomplete/errored data for {ticker_symbol}")
        # Send back minimal info if possible, plus error
        error_msg = yf_data.get('info',{}).get('error', 'Incomplete data received')
        return jsonify({
            "ticker": ticker_symbol,
            "error": f"{error_msg} for {ticker_symbol}.",
            "analysis": "Cannot generate analysis due to data error.",
            "chartData": []
            }), 500 # Internal Server Error or 503

    # Generate analysis using the fetched data
    analysis_text = get_groq_stock_analysis(ticker_symbol, yf_data)

    # Prepare response data, using .get() for safety and format_value for display
    info = yf_data.get('info', {})
    history_data = yf_data.get("history", []) # Already formatted for chart

    # Use regularMarketPrice preferentially if available
    current_price = info.get('regularMarketPrice') or info.get('currentPrice')

    response_data = {
        "ticker": ticker_symbol,
        "shortName": info.get('shortName', ticker_symbol),
        "currentPrice": current_price,
        "previousClose": info.get('previousClose'),
        "dayHigh": info.get('dayHigh'),
        "dayLow": info.get('dayLow'),
        "regularMarketOpen": info.get('regularMarketOpen') or info.get('open'),
        "volume": info.get('regularMarketVolume') or info.get('volume'),
        "marketCap": info.get('marketCap'),
        "timestamp": time.time(), # Server time when data was prepared
        "analysis": analysis_text,
        "chartData": history_data # Should be list of {'time': unix_ts, 'open': o, 'high': h, 'low': l, 'close': c}
    }

    # Convert any remaining NaN values to None for JSON compatibility
    for key, value in response_data.items():
         if isinstance(value, (float, np.number)) and math.isnan(value):
              response_data[key] = None
         # Special check for marketCap which might be large int
         elif key == 'marketCap' and value is not None:
             try:
                 response_data[key] = int(value) # Ensure it's standard int if not None
             except (ValueError, TypeError):
                 response_data[key] = None


    print(f"Successfully prepared response for {ticker_symbol}")
    return jsonify(response_data)


# == Mutual Fund Jargon Buster Endpoint ==
@app.route('/api/explain', methods=['POST'])
def explain_endpoint():
    """Receives a term/question and returns an explanation from Groq (MF context)."""
    print(f"\n--- Received request on /api/explain ---")
    if not groq_client:
        print("Error: Groq client not available for explanation.")
        return jsonify({"error": "AI explanation service is currently unavailable."}), 503

    try:
        data = request.get_json()
        if not data or 'term' not in data:
            print("Error: Missing 'term' in request body.")
            return jsonify({"error": "Missing 'term' (or question) in request body"}), 400

        term_or_question = data['term']
        if not isinstance(term_or_question, str) or not term_or_question.strip():
             print("Error: 'term' is empty or not a string.")
             return jsonify({"error": "Term/question cannot be empty"}), 400

        print(f"Processing explanation request for: '{term_or_question[:100]}'...")
        explanation = explain_term_with_groq(term_or_question)

        if explanation.startswith("Error:") or "error trying to get an explanation" in explanation:
             # Distinguish between Groq client init error vs. API call error
             status_code = 503 if "Groq client not initialized" in explanation else 500
             user_error = "Could not get explanation. The AI service might be unavailable or encountered an issue."
             print(f"Error generating explanation: {explanation}")
             return jsonify({"error": user_error}), status_code

        print("Successfully generated explanation.")
        return jsonify({"explanation": explanation})

    except Exception as e:
        print(f"CRITICAL Error in /api/explain endpoint handler: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred processing explanation."}), 500

# == Mutual Fund Illustrative Finder Endpoint ==
@app.route('/api/illustrate', methods=['POST'])
def illustrate_endpoint():
    """Receives a risk profile name and returns illustrative fund examples using KNN."""
    print(f"\n--- Received request on /api/illustrate ---")
    if knn_model_bundle is None: # Quick check if model loaded at all
         print("Error: KNN model bundle not loaded.")
         return jsonify({"error": "Illustrative Finder service is currently unavailable (model not loaded)."}), 503

    try:
        data = request.get_json()
        if not data or 'profile' not in data:
            print("Error: Missing 'profile' in request body.")
            return jsonify({"error": "Missing 'profile' (e.g., 'Conservative', 'Moderate', 'Aggressive') in request body"}), 400

        risk_profile_name = data['profile']
        valid_profiles = ['conservative', 'moderate', 'aggressive']
        if not isinstance(risk_profile_name, str) or not risk_profile_name.strip() or risk_profile_name.lower() not in valid_profiles:
             print(f"Error: Invalid 'profile' value received: {risk_profile_name}")
             return jsonify({"error": "Invalid 'profile' value. Choose 'Conservative', 'Moderate', or 'Aggressive'."}), 400

        print(f"Processing illustration request for profile: '{risk_profile_name}'")
        examples, error_msg = find_illustrative_funds_knn(risk_profile_name, num_results=5)

        if error_msg:
             print(f"Finder function returned error: {error_msg}")
             # Determine if it's a temporary issue (model unavailable) or internal processing error
             status_code = 503 if "model is currently unavailable" in error_msg.lower() else 500
             user_error_message = "Could not retrieve illustrative examples. " + (
                 "The Finder service might be temporarily unavailable." if status_code == 503 else "An internal error occurred."
             )
             return jsonify({"error": user_error_message}), status_code
        elif not examples:
             print(f"Finder function returned no examples for {risk_profile_name}, although no error reported.")
             # This might indicate an issue with the KNN data or profile matching, treat as potentially an internal issue
             return jsonify({"error": f"Could not find illustrative examples matching '{risk_profile_name}'. Please try another profile or check back later."}), 500
        else:
             message = (
                 f"Found {len(examples)} illustrative examples broadly similar to a generic '{risk_profile_name.capitalize()}' profile based on available data (risk, category, rating, historical returns)."
                 "\n\n**Note:** Lower 'Similarity Score' indicates higher similarity within the model's parameters. These are **NOT** recommendations."
                 "\n\n**Disclaimer:** Data is historical, may not be current, and similarity is based on limited features. Always consult a SEBI-registered investment advisor before making any investment decisions."
             )
             print(f"Successfully found {len(examples)} examples for {risk_profile_name}.")
             return jsonify({"message": message, "examples": examples})

    except Exception as e:
        print(f"CRITICAL Error in /api/illustrate endpoint handler: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred finding illustrations."}), 500

# == Mutual Fund Quiz Endpoint ==
@app.route('/api/quiz/questions', methods=['GET'])
def get_quiz_questions():
    """Returns a shuffled list of mutual fund quiz questions."""
    print("\n--- Received request on /api/quiz/questions ---")
    try:
        # Shuffle the global list *copy* each time to provide variety
        questions_to_send = quiz_data[:] # Create a shallow copy
        random.shuffle(questions_to_send)

        # Optionally limit the number of questions sent
        # num_questions_to_send = 10
        # questions_to_send = questions_to_send[:num_questions_to_send]

        print(f"Returning {len(questions_to_send)} shuffled quiz questions.")
        # Return the shuffled list as JSON
        return jsonify(questions_to_send)
    except Exception as e:
        print(f"CRITICAL Error in /api/quiz/questions endpoint handler: {e}")
        traceback.print_exc()
        # Return a JSON error response
        return jsonify({"error": "An internal server error occurred while fetching quiz questions."}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # Load resources first
    if initialize_resources():
        print("\n--- Essential resources loaded successfully (Groq Client, MF Data) ---")
        if knn_model_bundle is None:
             print("--- WARNING: KNN Model failed to load. Illustrative Finder endpoint (/api/illustrate) will return errors. ---")
        print(f"--- Starting Flask Server on http://127.0.0.1:{SERVER_PORT} ---")
        print("--- Accessible Endpoints: ---")
        print(f"    GET  /api/asset_data/<TICKER>   (e.g., /api/asset_data/RELIANCE.NS)")
        print(f"    POST /api/explain               (Body: {{'term': 'your_term'}})")
        print(f"    POST /api/illustrate            (Body: {{'profile': 'Conservative|Moderate|Aggressive'}})")
        print(f"    GET  /api/quiz/questions")
        print("--- Press CTRL+C to quit ---")
        # Set debug=False for production, True for development (enables auto-reload)
        # Use host='0.0.0.0' to make it accessible from other devices on the network if needed
        app.run(host='127.0.0.1', port=SERVER_PORT, debug=True)
    else:
        print("\n********************************************************************")
        print("CRITICAL FAILURE: Failed to load essential resources during startup.")
        print("Common issues: GROQ_API_KEY missing/invalid in .env, CSV/PKL file path incorrect or file corrupted.")
        print("The Flask server cannot start reliably. Please check the error messages above.")
        print("********************************************************************")
        exit(1) # Exit if critical resources failed