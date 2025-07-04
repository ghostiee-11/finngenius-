# --- serverh.py ---
import os
import json
import uuid
from datetime import datetime, timezone, timedelta
import logging
import threading
import re
import time
import random
from urllib.parse import urljoin
import numpy as np # Often useful for finance calcs, ensure installed
from math import isfinite # For checking float results

# --- Flask ---
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- Core Dependencies ---
import requests
from bs4 import BeautifulSoup
import joblib
import pandas as pd
import yfinance as yf



# --- Twilio ---
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    logging.warning("Twilio library not found. WhatsApp alerts will be disabled.")
    TWILIO_AVAILABLE = False
    Client = None
    TwilioRestException = Exception # Define as base Exception if unavailable

# --- Chatbot Dependencies (Langchain & AI Models) ---
try:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        DEFAULT_PDF_LOADER = PyMuPDFLoader
        logging.info("PyMuPDF found, will use for PDF loading.")
    except ImportError:
        from langchain_community.document_loaders import PyPDFLoader
        DEFAULT_PDF_LOADER = PyPDFLoader
        logging.info("PyMuPDF not found, using PyPDFLoader for PDF loading.")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    missing_package = str(e).split("'")[-2] if "No module named" in str(e) else "Langchain/Groq/HF"
    logging.error(f"Import failed for {missing_package}: {e}. Chatbot functionality disabled.")
    LANGCHAIN_AVAILABLE = False

# --- Scikit-learn (for Scam Check) ---
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    SKLEARN_AVAILABLE = True
except ImportError as e:
    logging.error(f"Scikit-learn import failed: {e}. Scam checker disabled.")
    SKLEARN_AVAILABLE = False

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

# --- Constants ---
SCRAPE_INTERVAL_SECONDS = 3600 # Increased interval for less frequent scraping maybe
MAX_ARTICLES_PER_SOURCE = 10
FLASK_PORT = 5001
SCAM_PROBABILITY_THRESHOLD = 65
ALERT_CHECK_INTERVAL_SECONDS = 60
CACHE_EXPIRY_SECONDS = 120 # Stock price cache expiry

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths (Ensure these paths are correct for your setup)
VECTORIZER_PATH = os.path.join(BASE_DIR, "count_vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "multinomial_nb_model.pkl")
STOCK_LIST_CSV_PATH = os.path.join(BASE_DIR, "stocks.csv")
PDF_DIRECTORY = os.path.join(BASE_DIR, 'pdfs')
INPUT_KNOWLEDGE_FILE = os.path.join(BASE_DIR, 'user_knowledge_input.json')
USER_KNOWLEDGE_FILE = os.path.join(BASE_DIR, 'user_knowledge.json')
VECTORSTORE_PATH = os.path.join(BASE_DIR, "faiss_index_finedu")

# API Endpoints
NEWS_API_ENDPOINT = '/api/finance-news'
CHECK_SCAM_ENDPOINT = '/api/check'
STOCKS_API_ENDPOINT = '/api/stocks'
STOCK_PRICE_API_ENDPOINT = '/api/stock-price'
ALERTS_API_ENDPOINT = '/api/alerts'
CHAT_API_ENDPOINT = '/api/chat'
GOAL_PLAN_ENDPOINT = '/api/goal-plan'                 # <-- NEW
GOAL_HOLDINGS_ENDPOINT = '/api/goal-holdings'         # <-- NEW
GOAL_PERFORMANCE_ENDPOINT = '/api/goal-performance'   # <-- NEW

# --- Globals ---
latest_articles = []
stock_list = []
active_alerts = {}
stock_data_lock = threading.Lock()
data_lock = threading.Lock()
alerts_lock = threading.Lock()
nb_model = None
count_vectorizer = None
llm = None # Will be initialized by initialize_chatbot
embeddings = None
vectorstore = None
user_memories = {}
rag_chain = None
twilio_client = None
TWILIO_WHATSAPP_NUMBER_FORMATTED = None
price_cache = {} # For stock prices
cache_lock = threading.Lock()
# --- New Globals for Goal Planning (In-memory demo storage) ---
# WARNING: Use a database in production!
user_goals = {}         # { "user_id": { goal_details, plan_summary, target_allocation, market_context, insights } }
user_holdings = {}      # { "user_id": [ { holding_details_1 }, { holding_details_2 } ] }
goal_data_lock = threading.Lock() # Lock for accessing user_goals and user_holdings
allocationModels = {}  # or list, or custom class instance


# --- Hardcoded Credentials (!!! VERY INSECURE - USE ENV VARIABLES IN PRODUCTION !!!) ---
# Replace placeholders with your actual keys ONLY if you understand the risk
# It's highly recommended to load these from environment variables instead.
HARDCODED_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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") # Example: Try env var first
HARDCODED_TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "AC7579fef985f8a5b8c335f320887ffc2f") # Placeholder
HARDCODED_TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "531e3387ee075f8fde2d0882b4cbeef8") # Placeholder
HARDCODED_TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "+14155238886") # Placeholder
# --- ---

# --- Flask App Setup ---
app = Flask(__name__)
# Allow requests from typical local development origins
cors = CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:8000", "http://127.0.0.1:8000", # Common Python http.server ports
    "http://localhost:3000", "http://127.0.0.1:3000", # Common React/Node dev ports
    "http://localhost:5500", "http://127.0.0.1:5500", # Common Live Server ports
    "null" # For requests from local file:// URLs (like opening index.html directly)
]}})

# --- Function Definitions ---

# --- Add ALL your existing functions here ---
# initialize_twilio, load_scam_models, initialize_chatbot, load_stock_list,
# get_live_stock_price (important!), generate_title_from_url, scrape_site,
# scrape_all_sources, initial_scrape_and_update, periodic_scraper_thread_func,
# check_links, send_whatsapp_alert, check_alerts_periodically,
# update_knowledge, get_memory_for_user, find_relevant_tool_link, generate_quiz
# ... Make sure they are defined correctly before the API routes that use them ...
# --- ---

# --- * Example: Placeholder for initialize_chatbot if missing from above * ---
def initialize_chatbot():
    global llm, embeddings, vectorstore, rag_chain
    if not LANGCHAIN_AVAILABLE:
        logging.error("Langchain components unavailable. Cannot initialize chatbot.")
        return False
    try:
        # Initialize LLM (Groq)
        groq_api_key = HARDCODED_GROQ_API_KEY
        if not groq_api_key or "gsk_" not in groq_api_key: # Basic check
            raise ValueError("Hardcoded GROQ_API_KEY is missing or invalid.")
        llm = ChatGroq(temperature=0.4, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
        logging.info(f"Groq LLM initialized (model: {llm.model_name}).")

        # Initialize Embeddings (HuggingFace)
        model_kwargs = {'device': 'cpu'}; encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        logging.info("HuggingFace Embeddings initialized.")

        # Load or Build Vector Store
        if os.path.exists(VECTORSTORE_PATH) and embeddings:
            logging.info(f"Loading existing vector store from {VECTORSTORE_PATH}...")
            try:
                 vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True) # Added flag
                 logging.info("Vector store loaded successfully.")
            except Exception as load_err:
                 logging.error(f"Error loading vector store (may need rebuild): {load_err}. Rebuilding...")
                 vectorstore = None # Force rebuild
        else:
             logging.info(f"Vector store not found or embeddings missing. Will attempt to build.")
             vectorstore = None

        if vectorstore is None and embeddings: # Build if needed/failed load
             # --- Build FAISS Index logic (copy from your original file) ---
             logging.info("Attempting to build new vector store...")
             # ... (rest of your PDF loading, splitting, FAISS.from_documents logic) ...
             # Make sure PDF_DIRECTORY and documents logic is present
             # Example Snippet:
             if not os.path.exists(PDF_DIRECTORY): logging.error(f"PDF directory not found: {PDF_DIRECTORY}."); return False
             pdf_list = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]
             if not pdf_list: logging.error(f"No PDF files found in: {PDF_DIRECTORY}"); return False
             documents = []
             for pdf_file in pdf_list:
                pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
                try:
                    loader = DEFAULT_PDF_LOADER(pdf_path); docs = loader.load()
                    if docs: documents.extend(docs); logging.info(f"Loaded {len(docs)} from '{pdf_file}'")
                except Exception as e: logging.warning(f"Error loading {pdf_path}: {e}")
             if not documents: logging.error("No documents loaded."); return False
             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
             split_docs = text_splitter.split_documents(documents)
             logging.info(f"Split into {len(split_docs)} chunks.")
             try:
                 logging.info("Building FAISS index with local embeddings...")
                 vectorstore = FAISS.from_documents(split_docs, embeddings)
                 vectorstore.save_local(VECTORSTORE_PATH)
                 logging.info(f"Vector store built and saved to {VECTORSTORE_PATH}.")
             except Exception as build_err:
                 logging.error(f"Fatal: Error building vector store: {build_err}", exc_info=True)
                 vectorstore = None; return False
             # --- End Build Logic ---

        if llm and vectorstore: # Define RAG Chain if components ready
             # --- RAG Chain Definition (copy from your original file) ---
             contextualize_q_system_prompt = """Given a chat history...""" # Your full prompt
             contextualize_q_prompt = ChatPromptTemplate.from_messages([...]) # Your full prompt template
             rag_system_prompt = """You are 'FinBot'...""" # Your full RAG system prompt
             rag_prompt = ChatPromptTemplate.from_messages([...]) # Your full prompt template
             retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
             history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
             question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)
             rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
             logging.info("RAG chain created successfully.")
             return True
        else:
             logging.error("LLM or Vectorstore unavailable after init. Cannot create RAG chain.")
             return False

    except Exception as e:
        logging.error(f"Fatal: Error during chatbot initialization: {e}", exc_info=True)
        llm = vectorstore = rag_chain = None
        return False
# --- * End Placeholder * ---

# --- Allocation Model Selection Function (needed for Goal Planning) ---
def selectAllocationModel(years, risk):
    """Selects a predefined allocation model based on years and risk."""
    timeCategory = 'long' # Default
    if years <= 3: timeCategory = 'short'
    elif years <= 7: timeCategory = 'medium'
    riskCategory = risk.lower() if risk else 'moderate'
    modelKey = f"{timeCategory}_{riskCategory}"
    # Reference the globally defined allocationModels dictionary
    return allocationModels.get(modelKey)


# --- NEW Helper Functions for Goal Planning ---

def parse_return_range(range_str, default_avg=0.08):
    """Parses strings like '10%-15%' into an average decimal rate."""
    if not isinstance(range_str, str): return default_avg
    try:
        # Find numbers, including decimals
        nums = [float(x) for x in re.findall(r'\d+\.?\d*', range_str)]
        if len(nums) >= 2:
            # Average the first two numbers found
            return (nums[0] + nums[1]) / 2 / 100.0
        elif len(nums) == 1:
            # Use the single number found
            return nums[0] / 100.0
        else:
            # Return default if no numbers found
            return default_avg
    except Exception as e:
        logging.warning(f"Error parsing return range '{range_str}': {e}. Using default {default_avg}")
        return default_avg

def calculate_sip(future_value, annual_rate, years):
    """Calculates the required monthly SIP (assuming end-of-period payments)."""
    if not isfinite(future_value) or future_value <= 0: return 0 # No SIP needed if target is 0 or less
    if years <= 0: return float('inf') # Cannot reach in 0 time
    if annual_rate <= -1: return float('inf') # Cannot reach with <= -100% return
    try:
        n_months = years * 12
        if annual_rate == 0: # Handle zero rate
             return future_value / n_months if n_months > 0 else float('inf')

        monthly_rate = annual_rate / 12.0
        # Formula: P = FV * r / (((1 + r)^n) - 1)
        denominator = (1 + monthly_rate)**n_months - 1
        # Check for edge cases: denominator near zero indicates extremely high required SIP or impossible goal
        if abs(denominator) < 1e-9:
            logging.warning(f"SIP calculation denominator near zero (rate={annual_rate}, years={years}). Goal likely unfeasible.")
            return float('inf')
        sip = future_value * monthly_rate / denominator
        return sip if isfinite(sip) else float('inf') # Return inf if overflow happens
    except (OverflowError, ValueError) as e:
        logging.error(f"Error calculating SIP (FV={future_value}, Rate={annual_rate}, Years={years}): {e}")
        return float('inf')
# --- Add this function definition if it's missing ---
def get_live_stock_price(symbol):
    """Fetches live or recent stock price using yfinance."""
    # Clean symbol and add .NS if needed (common for Indian stocks)
    logging.debug(f"Fetching price for symbol: {symbol}")
    yf_symbol = symbol.strip().upper()
    if '.' not in yf_symbol: # Add .NS if no exchange specified
        yf_symbol += '.NS'
        logging.debug(f"Assuming NSE, using symbol: {yf_symbol}")
    elif not yf_symbol.endswith(('.NS', '.BO')):
        logging.warning(f"Symbol {symbol} has unusual suffix. Proceeding as is: {yf_symbol}")

    try:
        # --- Check Cache First ---
        now = time.time()
        with cache_lock:
            cached_data = price_cache.get(yf_symbol)
            if cached_data and (now - cached_data['timestamp'] < CACHE_EXPIRY_SECONDS):
                logging.info(f"Returning cached price for {yf_symbol}: {cached_data['price']}")
                # Return structure expected by callers
                return {
                    'price': cached_data['price'],
                    'currency': cached_data.get('currency', 'INR'),
                    'note': cached_data.get('note', '') + ' (Cached)',
                    'source': 'cache'
                 }
        # --- End Cache Check ---

        logging.info(f"Cache miss/expired for {yf_symbol}. Fetching live price via yfinance...")
        stock = yf.Ticker(yf_symbol)

        # --- Method 1: Fast Info (Often real-time for liquid stocks) ---
        # Use 'fast_info' which is generally quicker and often has live data
        # Common keys: 'lastPrice', 'regularMarketPrice', 'currentPrice'
        fast_info = stock.fast_info
        price = fast_info.get('lastPrice') or fast_info.get('regularMarketPrice') or fast_info.get('currentPrice')

        if price is not None and isinstance(price, (int, float)) and price > 0:
            logging.info(f"Price found via fast_info for {yf_symbol}: {price}")
            price_result = {'price': float(price), 'currency': fast_info.get('currency', 'INR'), 'source': 'fast_info'}
            # Update cache
            with cache_lock: price_cache[yf_symbol] = {'price': price_result['price'], 'currency': price_result['currency'], 'timestamp': now}
            return price_result

        # --- Method 2: History (Fallback for less liquid or delayed data) ---
        logging.debug(f"fast_info price not suitable ({price}) for {yf_symbol}. Trying history...")
        # Fetch 1 day history, interval 1m might give more recent data if available, else defaults
        # Use period='1d' to get the most recent trading day's data
        hist = stock.history(period='1d', interval='1m', auto_adjust=True) # Try 1 minute interval first

        # If 1m fails or is empty, try daily
        if hist.empty:
             logging.debug(f"1m interval history empty for {yf_symbol}. Trying daily interval...")
             hist = stock.history(period='1d', auto_adjust=True) # Daily interval

        if not hist.empty:
            # Get the last available closing price
            last_close = hist['Close'].iloc[-1]
            if last_close is not None and isinstance(last_close, (int, float)) and last_close > 0:
                logging.info(f"Price found via history (last close) for {yf_symbol}: {last_close}")
                price_result = {'price': float(last_close), 'currency': fast_info.get('currency', 'INR'), 'note': 'Last close', 'source': 'history'} # Use currency from fast_info if possible
                # Update cache
                with cache_lock: price_cache[yf_symbol] = {'price': price_result['price'], 'currency': price_result['currency'], 'timestamp': now, 'note': 'Last close'}
                return price_result

        # --- Method 3: Slower info dictionary (More data but potentially slower) ---
        # If fast_info and history fail, try the full info dictionary as a last resort
        # This is generally slower than fast_info
        logging.debug(f"History price not suitable for {yf_symbol}. Trying full info dict...")
        info = stock.info
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') # Add previousClose as another fallback

        if price is not None and isinstance(price, (int, float)) and price > 0:
            logging.info(f"Price found via full info dict for {yf_symbol}: {price}")
            price_result = {'price': float(price), 'currency': info.get('currency', 'INR'), 'source': 'info_dict'}
            # Update cache
            with cache_lock: price_cache[yf_symbol] = {'price': price_result['price'], 'currency': price_result['currency'], 'timestamp': now}
            return price_result

        # --- If all methods fail ---
        logging.warning(f"Could not get valid > 0 price for {yf_symbol} using fast_info, history, or info.")
        return {"error": f"Could not retrieve valid price for {yf_symbol}", "symbol": symbol}

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Network connection error fetching price for {yf_symbol}: {e}")
        return {"error": f"Network connection error for {yf_symbol}", "symbol": symbol}
    except Exception as e:
        err_str = str(e).lower()
        common_yf_errors = ["no data found", "symbol may be delisted", "404 client error", "indexerror: single positional indexer is out-of-bounds"]
        # Check if it's a common yfinance "not found" type error
        if any(sub in err_str for sub in common_yf_errors):
            log_level = logging.WARNING # Less severe log for common errors
            error_message = f"Symbol {yf_symbol} invalid or data unavailable via yfinance."
        else:
            log_level = logging.ERROR # Log unexpected errors more severely
            error_message = f"Unexpected error fetching price for {yf_symbol}."

        logging.log(log_level, f"{error_message} Error: {e}", exc_info=(log_level == logging.ERROR)) # Show traceback for unexpected errors
        return {"error": error_message, "symbol": symbol}
# --- End of get_live_stock_price definition ---
def calculate_fv(present_value, monthly_investment, annual_rate, years):
     """Calculates Future Value of initial + monthly investments (end-of-period)."""
     try:
        n_months = years * 12
        if n_months <= 0: return float(present_value) # No time to grow

        if annual_rate == 0: # Handle zero rate
             return float(present_value + (monthly_investment * n_months))

        monthly_rate = annual_rate / 12.0
        rate_factor = (1 + monthly_rate)**n_months

        fv_initial = present_value * rate_factor
        # FV of SIP = P * [((1 + r)^n - 1) / r]
        fv_monthly = 0
        if monthly_rate != 0:
             denominator = monthly_rate
             if abs(denominator) < 1e-9: # Should not happen if annual_rate != 0, but safety check
                 fv_monthly = monthly_investment * n_months # Approx as 0% growth if rate is tiny
             else:
                 fv_monthly = monthly_investment * (rate_factor - 1) / denominator

        total_fv = fv_initial + fv_monthly
        return total_fv if isfinite(total_fv) else float('inf') # Handle potential overflow
     except (OverflowError, ValueError) as e:
         logging.error(f"Error calculating FV (PV={present_value}, PMT={monthly_investment}, Rate={annual_rate}, Years={years}): {e}")
         return float(present_value) # Return initial value on math error


# --- Backend Price Fetcher (Uses yfinance for Stocks, Simulates Others) ---
def get_current_price_backend(holding):
    """Gets live price for Stocks using yfinance, simulates for others."""
    # Default values
    current_value = 0.0
    invested_amount = float(holding.get('investedAmount', 0))

    try:
        asset_type = holding.get('type')
        symbol = holding.get('ticker') or holding.get('symbol') # Stock or Crypto symbol
        fund_name = holding.get('fundName')
        quantity = float(holding.get('quantity', 1)) if holding.get('quantity') is not None else 1.0
        purchase_price = float(holding.get('purchasePrice', 0)) if holding.get('purchasePrice') is not None else 0.0
        # Calculate base value per unit if possible, otherwise use invested amount as fallback base for simulation
        base_value_per_unit = purchase_price if purchase_price > 0 else (invested_amount / quantity if quantity > 0 else 0)

        logging.debug(f"Getting price for Type: {asset_type}, Symbol: {symbol}, Fund: {fund_name}, Qty: {quantity}, PP: {purchase_price}, Inv: {invested_amount}")

        if asset_type == 'Stock' and symbol:
            # Ensure get_live_stock_price is defined and works
            price_data = get_live_stock_price(symbol)
            if 'price' in price_data and price_data['price'] is not None:
                current_value = float(price_data['price']) * quantity
                logging.debug(f"Stock {symbol} live price: {price_data['price']:.2f}, Total: {current_value:.2f}")
            else:
                 logging.warning(f"yfinance failed for {symbol}. Simulating price.")
                 sim_price_per_unit = base_value_per_unit * (1 + (random.random() - 0.4) * 0.1) # Simulate smaller change
                 current_value = sim_price_per_unit * quantity

        elif asset_type == 'Mutual Fund':
             logging.debug(f"Simulating price for MF: {fund_name}")
             sim_nav = base_value_per_unit * (1 + (random.random() - 0.45) * 0.2) # Simulate +/- 10% NAV change
             current_value = sim_nav * quantity

        elif asset_type == 'Gold':
            logging.debug(f"Simulating price for Gold: {holding.get('description')}")
            sim_price_per_unit = base_value_per_unit * (1 + (random.random() - 0.4) * 0.3) # +/- 15%
            current_value = sim_price_per_unit * quantity

        elif asset_type == 'Crypto' and symbol:
            logging.debug(f"Simulating price for Crypto: {symbol}")
            sim_price_per_unit = base_value_per_unit * (1 + (random.random() - 0.4) * 0.8) # +/- 40%
            current_value = sim_price_per_unit * quantity

        elif asset_type == 'FD':
            # Simulate modest growth based on time elapsed (approximate)
            try:
                added_dt = datetime.fromisoformat(holding.get('added_at', datetime.now(timezone.utc).isoformat())).replace(tzinfo=timezone.utc)
                days_held = max(0, (datetime.now(timezone.utc) - added_dt).days)
                sim_interest_rate = 0.07 # Assume 7% p.a. for simulation
                growth_factor = (sim_interest_rate / 365) * days_held
                current_value = invested_amount * (1 + growth_factor)
            except Exception:
                 current_value = invested_amount * (1 + random.random() * 0.01) # Simpler fallback
            logging.debug(f"Simulating price for FD: {holding.get('description')}")

        elif asset_type == 'Cash':
             current_value = invested_amount # Assume no change
             logging.debug(f"Price for Cash: {current_value:.2f}")
        else:
             logging.warning(f"Unknown asset type for price fetch: {asset_type}. Using invested amount.")
             current_value = invested_amount

        # Ensure the final value is a float
        current_value = float(current_value) if isfinite(current_value) else 0.0

    except Exception as e:
        logging.error(f"Error getting/simulating price for {holding.get('description', symbol)}: {e}", exc_info=True)
        current_value = invested_amount # Fallback to invested amount on any error

    # Return the calculated total current value for the holding
    return current_value


# --- NEW GOAL PLANNING API Endpoints ---

@app.route(GOAL_PLAN_ENDPOINT, methods=['POST'])
def handle_goal_plan_api():
    """Orchestrates goal planning using AI and calculations."""
    endpoint_start_time = time.time()
    logging.info(f"API Req Start: {GOAL_PLAN_ENDPOINT}")
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 415
    if not llm:
        logging.error(f"{GOAL_PLAN_ENDPOINT} Error: LLM not initialized.")
        return jsonify({"error": "AI Service unavailable"}), 503

    data = request.get_json()
    user_id = data.get('user_id', 'default_user')
    goal_name = data.get('goal_name')
    target_amount_str = data.get('target_amount')
    time_horizon_str = data.get('time_horizon')
    initial_investment_str = data.get('initial_investment', '0')
    monthly_investment_str = data.get('monthly_investment')
    risk_tolerance = data.get('risk_tolerance')

    # --- Server-Side Input Validation ---
    errors = {}
    target_amount, time_horizon, initial_investment, monthly_investment = 0.0, 0, 0.0, 0.0
    if not goal_name: errors['goal_name'] = "Required"
    if not risk_tolerance: errors['risk_tolerance'] = "Required"
    try: target_amount = float(target_amount_str); assert target_amount > 0
    except: errors['target_amount'] = "Must be positive number"
    try: time_horizon = int(time_horizon_str); assert time_horizon > 0
    except: errors['time_horizon'] = "Must be positive integer years"
    try: initial_investment = float(initial_investment_str); assert initial_investment >= 0
    except: errors['initial_investment'] = "Must be non-negative number"
    try: monthly_investment = float(monthly_investment_str); assert monthly_investment >= 0
    except: errors['monthly_investment'] = "Must be non-negative number"

    if errors:
        logging.warning(f"{GOAL_PLAN_ENDPOINT} Validation failed for user {user_id}: {errors}")
        return jsonify({"error": "Validation failed", "details": errors}), 400

    # --- Call Groq for Market Context & Return Assumptions ---
    market_context = None
    cagr_ranges = {}
    assumed_rate_for_plan = 0.08 # Default
    groq_call_start = time.time()
    try:
        prompt = f"""Analyze current Indian/global economic outlook (inflation, rates, growth) for a {time_horizon}-year investment horizon. Provide indicative expected annual return (CAGR) ranges for: Indian Large-Cap Equities, Indian Mid/Small-Cap Equities, Diversified Equity Mutual Funds, Indian Debt Instruments, Gold (INR), Major Cryptocurrencies (BTC/ETH - note volatility). Output ONLY valid JSON: {{"analysis_summary": "Brief summary...", "expected_cagr_ranges": {{"indian_large_cap": {{"range": "X%-Y%", "notes": "..."}}, ...}} }}"""
        logging.info(f"Requesting market context from Groq (User: {user_id}, Years: {time_horizon})...")
        completion = llm.invoke(prompt) # Use global llm
        raw_response = completion.content.strip()
        logging.debug(f"Groq market context raw response: {raw_response[:200]}...")
        json_start = raw_response.find('{'); json_end = raw_response.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
             market_context = json.loads(raw_response[json_start:json_end])
             cagr_ranges = market_context.get("expected_cagr_ranges", {})
             key_map = { 'low': 'indian_debt', 'moderate': 'diversified_equity_mf', 'high': 'indian_mid_small_cap'}
             selected_key = key_map.get(risk_tolerance.lower(), 'diversified_equity_mf')
             range_str = cagr_ranges.get(selected_key, {}).get('range')
             assumed_rate_for_plan = parse_return_range(range_str, default_avg=(0.06 if risk_tolerance.lower()=='low' else (0.10 if risk_tolerance.lower()=='high' else 0.08)))
             logging.info(f"User {user_id}: Using assumed rate {assumed_rate_for_plan:.2%} (Risk: {risk_tolerance}, AI Key: {selected_key}, AI Range: '{range_str}')")
        else:
             logging.warning(f"User {user_id}: Could not extract JSON from Groq market context response. Using default rate.")
             market_context = {"analysis_summary": "AI analysis unavailable, using defaults.", "expected_cagr_ranges": {}}
             assumed_rate_for_plan = 0.06 if risk_tolerance.lower()=='low' else (0.10 if risk_tolerance.lower()=='high' else 0.08)

    except Exception as e:
        logging.error(f"Error calling Groq for market context (User: {user_id}): {e}", exc_info=False) # Less verbose log
        market_context = {"analysis_summary": f"Error fetching AI analysis. Using defaults.", "expected_cagr_ranges": {}}
        assumed_rate_for_plan = 0.06 if risk_tolerance.lower()=='low' else (0.10 if risk_tolerance.lower()=='high' else 0.08)
    logging.info(f"Groq market context call took {time.time() - groq_call_start:.2f}s")


    # --- Calculations ---
    calc_start = time.time()
    target_needed_from_savings = target_amount - calculate_fv(initial_investment, 0, assumed_rate_for_plan, time_horizon)
    required_sip = calculate_sip(max(0, target_needed_from_savings), assumed_rate_for_plan, time_horizon)
    projected_fv = calculate_fv(initial_investment, monthly_investment, assumed_rate_for_plan, time_horizon)
    shortfall_surplus = projected_fv - target_amount

    # --- Determine Target Allocation ---
    selected_model = selectAllocationModel(time_horizon, risk_tolerance)
    target_allocation_pct = None
    if selected_model:
        target_allocation_pct = { # Uppercase keys consistent with performance calc
             'Equity': selected_model.get('equity', 0), 'Debt': selected_model.get('debt', 0),
             'Gold': selected_model.get('gold', 0), 'Cash': selected_model.get('cash', 0), 'Crypto': 0 # Default Crypto target
         }
    logging.info(f"Calculations took {time.time() - calc_start:.3f}s")

    # --- Generate Illustrative Insights using Groq ---
    insights = {"ai_generated_notes": ["AI insights temporarily unavailable."] } # Default
    # (Keep the insight generation logic as before, maybe add a check if llm is available)
    if llm:
        groq_call_start_2 = time.time()
        try:
            equity_focus = target_allocation_pct.get('Equity',0) if target_allocation_pct else 0 # Safely get value
            insight_prompt = f"""User: '{risk_tolerance}' risk, goal ₹{target_amount:,.0f} in {time_horizon} years. Target alloc ~{equity_focus}% Equity. Market context: "{market_context.get('analysis_summary', 'N/A')}". Provide 2-3 brief insights/considerations (diversification, risk, opportunities). No investment advice. Output ONLY valid JSON: {{"ai_generated_notes": ["Insight 1...", "Insight 2..."]}}"""
            logging.info(f"Requesting plan insights from Groq (User: {user_id})...")
            completion = llm.invoke(insight_prompt)
            raw_response = completion.content.strip()
            logging.debug(f"Groq insights raw response: {raw_response[:200]}...")
            json_start = raw_response.find('{'); json_end = raw_response.rfind('}') + 1
            if json_start != -1 and json_end > json_start: insights = json.loads(raw_response[json_start:json_end])
            else: logging.warning(f"User {user_id}: Could not extract JSON from Groq insights response.")
        except Exception as e:
            logging.error(f"Error calling Groq for insights (User: {user_id}): {e}", exc_info=False)
            insights = {"ai_generated_notes": [f"Error generating AI insights."]}
        logging.info(f"Groq insights call took {time.time() - groq_call_start_2:.2f}s")


    # --- Store Plan and Prepare Response ---
    plan_data = {
        "goal_details": { "name": goal_name, "target": target_amount, "years": time_horizon, "initial": initial_investment, "monthly_plan": monthly_investment, "risk": risk_tolerance },
        "plan_summary": { "required_monthly_sip": required_sip if isfinite(required_sip) else None, "projected_final_value": projected_fv if isfinite(projected_fv) else None, "shortfall_or_surplus": shortfall_surplus if isfinite(shortfall_surplus) else None, "assumed_annual_return": assumed_rate_for_plan },
        "target_allocation": target_allocation_pct,
        "market_context": market_context,
        "insights": insights,
        "last_updated": datetime.now(timezone.utc).isoformat()
    }

    with goal_data_lock:
        user_goals[user_id] = plan_data # Store/overwrite plan
        # Initialize holdings array if user is new to goal planning
        if user_id not in user_holdings:
             user_holdings[user_id] = []

    logging.info(f"{GOAL_PLAN_ENDPOINT} Request completed in {time.time() - endpoint_start_time:.2f}s for user {user_id}")
    return jsonify(plan_data), 200


@app.route(GOAL_HOLDINGS_ENDPOINT, methods=['GET', 'POST', 'DELETE'])
def handle_goal_holdings_api():
    # Use 'default_user' if 'user_id' is not in query params
    user_id_req = request.args.get('user_id', 'default_user')
    logging.info(f"API Req: {GOAL_HOLDINGS_ENDPOINT} ({request.method}) for user '{user_id_req}'")

    with goal_data_lock:
        # Ensure user exists in holdings dict, initialize if not
        if user_id_req not in user_holdings:
            user_holdings[user_id_req] = []

        # --- GET Request ---
        if request.method == 'GET':
            holdings = user_holdings.get(user_id_req, []) # Get copy
            logging.info(f"Returning {len(holdings)} holdings for user {user_id_req}")
            return jsonify(holdings)

        # --- POST Request (Add Holding) ---
        elif request.method == 'POST':
            if not request.is_json: return jsonify({"error": "Request must be JSON"}), 415
            data = request.get_json()
            if not data: return jsonify({"error": "Empty request body"}), 400

            # Simple validation - more robust validation recommended
            holding_type = data.get('type')
            asset_class = data.get('assetClass')
            # Invested amount is crucial for fallback calculations
            try: invested_amount = float(data.get('investedAmount', 0)); assert invested_amount >= 0
            except: return jsonify({"error": "Valid non-negative investedAmount required"}), 400
             # Purchase price and quantity are important too
            try: quantity = float(data.get('quantity', 0)) if data.get('quantity') is not None else None
            except: return jsonify({"error": "Invalid quantity"}), 400
            try: purchase_price = float(data.get('purchasePrice', 0)) if data.get('purchasePrice') is not None else None
            except: return jsonify({"error": "Invalid purchasePrice"}), 400


            if not holding_type or not asset_class:
                 return jsonify({"error": "Missing required fields (type, assetClass)"}), 400
            # Basic check: Need quantity and price OR invested amount usually
            if not invested_amount and (quantity is None or purchase_price is None):
                 logging.warning("Holding added with insufficient data (no invested amount or qty/price)")
                 # Allow adding but performance calc might be inaccurate

            new_holding = { "id": str(uuid.uuid4()), "added_at": datetime.now(timezone.utc).isoformat() }
            # Include all possible fields from frontend
            fields_to_copy = ["type", "assetClass", "investedAmount", "ticker", "fundName", "description", "quantity", "purchasePrice", "symbol"]
            for field in fields_to_copy:
                if data.get(field) is not None: # Copy if present in request
                     # Convert numerical fields explicitly
                     if field in ['investedAmount', 'quantity', 'purchasePrice']:
                         try: new_holding[field] = float(data[field])
                         except (ValueError, TypeError): new_holding[field] = 0 # Default to 0 on conversion error
                     else:
                         new_holding[field] = data[field]

             # Ensure investedAmount is a float
            new_holding['investedAmount'] = float(new_holding.get('investedAmount', 0))

            user_holdings[user_id_req].append(new_holding)
            logging.info(f"Added holding {new_holding['id']} ({new_holding['type']}) for user {user_id_req}")
            return jsonify({"message": "Holding added", "holding_id": new_holding['id']}), 201

        # --- DELETE Request ---
        elif request.method == 'DELETE':
             holding_id = request.args.get('holding_id')
             if not holding_id: return jsonify({"error": "holding_id query parameter required"}), 400

             holdings_for_user = user_holdings.get(user_id_req, [])
             initial_len = len(holdings_for_user)
             # Filter out the holding with the matching ID
             user_holdings[user_id_req] = [h for h in holdings_for_user if h.get('id') != holding_id]

             if len(user_holdings[user_id_req]) < initial_len:
                 logging.info(f"Removed holding {holding_id} for user {user_id_req}")
                 return jsonify({"message": "Holding removed"})
             else:
                 logging.warning(f"Holding ID {holding_id} not found for user {user_id_req} to delete.")
                 return jsonify({"error": "Holding not found"}), 404

    # Fallback for unsupported methods on this route
    return jsonify({"error": "Method not allowed"}), 405


@app.route(GOAL_PERFORMANCE_ENDPOINT, methods=['GET'])
def handle_goal_performance_api():
    user_id_req = request.args.get('user_id', 'default_user')
    endpoint_start_time = time.time()
    logging.info(f"API Req Start: {GOAL_PERFORMANCE_ENDPOINT} for user '{user_id_req}'")

    with goal_data_lock:
        current_plan = user_goals.get(user_id_req)
        holdings = user_holdings.get(user_id_req, []) # Get a copy

    if not current_plan:
        logging.warning(f"{GOAL_PERFORMANCE_ENDPOINT} Error: No plan found for user {user_id_req}")
        return jsonify({"error": "No active goal plan found for user."}), 404

    target_allocation = current_plan.get('target_allocation') # Uppercase keys
    if not target_allocation:
        logging.warning(f"Target allocation missing for user {user_id_req}. Cannot calculate drift.")
        # Proceed without drift calculation

    # --- Calculate Performance ---
    total_current_value = 0.0
    total_invested_value = 0.0
    # Initialize with all expected keys
    value_by_asset_class = { 'Equity': 0.0, 'Debt': 0.0, 'Gold': 0.0, 'Crypto': 0.0, 'Cash': 0.0 }
    calculation_errors = []

    if not holdings:
        logging.info(f"No holdings for user {user_id_req}. Performance is zero.")
    else:
        logging.info(f"Calculating performance for {len(holdings)} holdings for user {user_id_req}...")
        for holding in holdings:
            holding_value = 0.0
            invested = float(holding.get('investedAmount', 0))
            # Get current value using the backend function
            try:
                 holding_value = get_current_price_backend(holding) # This returns TOTAL value for the holding
                 holding_value = float(holding_value) if isfinite(holding_value) else 0.0 # Ensure float
            except Exception as price_err:
                 logging.error(f"Error getting price for holding ID {holding.get('id','N/A')}: {price_err}")
                 holding_value = invested # Fallback to invested value on error
                 calculation_errors.append(f"Price error for {holding.get('type','Unknown')}")

            total_current_value += holding_value
            total_invested_value += invested

            asset_class = holding.get('assetClass', 'Cash') # Default to Cash
            if asset_class in value_by_asset_class:
                value_by_asset_class[asset_class] += holding_value
            else:
                logging.warning(f"Holding ID {holding.get('id','N/A')} has unexpected asset class '{asset_class}'. Treating as Cash.")
                value_by_asset_class['Cash'] += holding_value

    # Calculate Actual Allocation %
    actual_allocation_pct = { key: 0.0 for key in value_by_asset_class } # Initialize
    if total_current_value > 1e-6: # Avoid division by zero or tiny numbers
        for key, value in value_by_asset_class.items():
            actual_allocation_pct[key] = (value / total_current_value) * 100

    # Calculate Drift %
    allocation_drift_pct = { key: None for key in value_by_asset_class } # Initialize with None
    if target_allocation:
         for key in value_by_asset_class:
             target = target_allocation.get(key, 0) # Get target % (e.g., target_allocation['Equity'])
             actual = actual_allocation_pct.get(key, 0)
             allocation_drift_pct[key] = actual - target # Calculate drift: Actual - Target
    # else: Drift remains None

    # Calculate Gain/Loss
    gain_loss = total_current_value - total_invested_value
    gain_loss_percent = (gain_loss / total_invested_value * 100) if total_invested_value > 1e-6 else 0

    snapshot = {
        "total_current_value": total_current_value,
        "total_invested_value": total_invested_value,
        "overall_gain_loss": gain_loss,
        "overall_gain_loss_percent": gain_loss_percent,
        "actual_allocation_percent": actual_allocation_pct,
        "allocation_drift_percent": allocation_drift_pct, # Can contain None if no target
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "calculation_notes": calculation_errors if calculation_errors else None # Include errors if any
    }
    logging.info(f"{GOAL_PERFORMANCE_ENDPOINT} Request completed in {time.time() - endpoint_start_time:.2f}s for user {user_id_req}")
    return jsonify(snapshot)


# --- Existing API Endpoints (Placeholders - ensure full implementation exists) ---
@app.route(NEWS_API_ENDPOINT)
def get_finance_news_api(): logging.info(f"Req: {NEWS_API_ENDPOINT}"); return jsonify(latest_articles)

@app.route(CHECK_SCAM_ENDPOINT, methods=['POST'])
def check_scam_api(): logging.info(f"Req: {CHECK_SCAM_ENDPOINT}"); return jsonify({"error": "Not fully implemented in combined example"}), 501

@app.route(STOCKS_API_ENDPOINT)
def get_stock_list_api(): logging.info(f"Req: {STOCKS_API_ENDPOINT}"); return jsonify(stock_list)

@app.route(f'{STOCK_PRICE_API_ENDPOINT}/<string:symbol>')
def get_stock_price_api(symbol):
     # This endpoint now uses the same logic as the goal performance price fetch
     logging.info(f"API req: {STOCK_PRICE_API_ENDPOINT}/{symbol}")
     price_result = get_live_stock_price(symbol) # Use the yfinance fetcher directly
     if "error" in price_result:
        status_code = 500; err_msg_lower = price_result["error"].lower()
        if "not found" in err_msg_lower or "invalid" in err_msg_lower: status_code = 404
        elif "network connection" in err_msg_lower: status_code = 504
        elif "could not retrieve" in err_msg_lower: status_code = 503
        return jsonify({"error": price_result['error'], "symbol": symbol}), status_code
     else:
         response_data = { 'symbol': symbol, 'price': price_result['price'], 'currency': price_result.get('currency', 'INR'), 'timestamp': time.time(), 'note': price_result.get('note')}
         return jsonify(response_data), 200


@app.route(ALERTS_API_ENDPOINT, methods=['POST'])
def create_alert_api(): logging.info(f"Req: {ALERTS_API_ENDPOINT}"); return jsonify({"error": "Not fully implemented in combined example"}), 501

@app.route(CHAT_API_ENDPOINT, methods=['POST'])
def handle_chat_api(): logging.info(f"Req: {CHAT_API_ENDPOINT}"); return jsonify({"error": "Not fully implemented in combined example"}), 501

@app.route('/')
def index_route():
    # Keep your original status display logic here
     return "<h1>Goal Planning & FinGenius Backend Running</h1>"


# --- Main Execution ---
if __name__ == '__main__':
    print("--- FinGenius Backend Starting (with Goal Planning) ---")
    print("Loading stock list...")
    # load_stock_list(STOCK_LIST_CSV_PATH) # Ensure this function is defined above
    print("Loading scam checking models...")
    # load_scam_models() # Ensure this function is defined above
    print("Initializing Chatbot Components (this may take time)...")
    if not initialize_chatbot(): # Ensure this function is defined and called
        print("ERROR: Chatbot initialization failed. Chat/Goal APIs may be affected.")
    else:
        print("Chatbot components initialized.")
    print("Performing initial news scrape...")
    # initial_scrape_and_update() # Ensure this function is defined above
    print("Initializing Twilio client...")
    # initialize_twilio() # Ensure this function is defined above
    print("Starting background threads...")
    # Ensure these functions are defined above
    # threading.Thread(target=periodic_scraper_thread_func, name="NewsScraperThread", daemon=True).start()
    # threading.Thread(target=check_alerts_periodically, name="AlertCheckerThread", daemon=True).start()
    print(f"Attempting to start Flask server on port {FLASK_PORT}...")
    try:
        from waitress import serve
        print(f"Running with Waitress server on http://0.0.0.0:{FLASK_PORT}")
        serve(app, host="0.0.0.0", port=FLASK_PORT, threads=8)
    except ImportError:
        print("Waitress not found. Using Flask development server (for testing only).")
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)

    logging.info("FinGenius Flask Backend Shut Down.")