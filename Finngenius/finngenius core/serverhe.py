# --- serverh.py ---
import os
import json
import uuid
from datetime import datetime, timezone
import logging
import threading
import re
import time
import random
from urllib.parse import urljoin

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
    TwilioRestException = Exception

# --- Chatbot Dependencies (Langchain & AI Models) ---
try:
    from langchain_groq import ChatGroq                     # LLM
    from langchain_huggingface import HuggingFaceEmbeddings # <<< USING HUGGINGFACE >>>
    from langchain_community.vectorstores import FAISS
    # Use PyMuPDFLoader if available, fallback to PyPDFLoader
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
SCRAPE_INTERVAL_SECONDS = 300
MAX_ARTICLES_PER_SOURCE = 10
FLASK_PORT = 5001
SCAM_PROBABILITY_THRESHOLD = 65
ALERT_CHECK_INTERVAL_SECONDS = 60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
VECTORIZER_PATH = os.path.join(BASE_DIR, "count_vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "multinomial_nb_model.pkl")
STOCK_LIST_CSV_PATH = os.path.join(BASE_DIR, "stocks.csv")
PDF_DIRECTORY = os.path.join(BASE_DIR, 'pdfs')
INPUT_KNOWLEDGE_FILE = os.path.join(BASE_DIR, 'user_knowledge_input.json')
USER_KNOWLEDGE_FILE = os.path.join(BASE_DIR, 'user_knowledge.json')
VECTORSTORE_PATH = os.path.join(BASE_DIR, "faiss_index_finedu") # Using local index

# API Endpoints
NEWS_API_ENDPOINT = '/api/finance-news'
CHECK_SCAM_ENDPOINT = '/api/check'
STOCKS_API_ENDPOINT = '/api/stocks'
STOCK_PRICE_API_ENDPOINT = '/api/stock-price'
ALERTS_API_ENDPOINT = '/api/alerts'
CHAT_API_ENDPOINT = '/api/chat'

# --- Globals ---
latest_articles = []
stock_list = []
active_alerts = {}
stock_data_lock = threading.Lock()
data_lock = threading.Lock()
alerts_lock = threading.Lock()
nb_model = None
count_vectorizer = None
llm = None
embeddings = None
vectorstore = None
user_memories = {}
rag_chain = None
twilio_client = None
TWILIO_WHATSAPP_NUMBER_FORMATTED = None
price_cache = {}
cache_lock = threading.Lock()
CACHE_EXPIRY_SECONDS = 60 * 2 

# --- Hardcoded Credentials (!!! SECURITY RISK !!!) ---
HARDCODED_GROQ_API_KEY = "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"        # Replace!
# HARDCODED_GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE" # No longer needed for embeddings
HARDCODED_TWILIO_ACCOUNT_SID = "AC7579fef985f8a5b8c335f320887ffc2f" # Replace or keep placeholder if unused
HARDCODED_TWILIO_AUTH_TOKEN = "efe7b140e2851f91dd2c7579a0058733"      # Replace or keep placeholder if unused
HARDCODED_TWILIO_WHATSAPP_NUMBER = "+14155238886"  #+1 (415) 523-8886             # Replace!
# --- ---

# --- Flask App Setup ---
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:8000", "http://127.0.0.1:8000",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "null",
    "http://127.0.0.1:3001"  # <-- Ensure this is correctly added
]}})

# --- Function Definitions ---

def initialize_twilio():
    """Initializes the Twilio client using hardcoded credentials."""
    global twilio_client, TWILIO_WHATSAPP_NUMBER_FORMATTED
    if not TWILIO_AVAILABLE or Client is None:
        logging.warning("Twilio library not available. Skipping init.")
        return

    account_sid = HARDCODED_TWILIO_ACCOUNT_SID
    auth_token = HARDCODED_TWILIO_AUTH_TOKEN
    from_number = HARDCODED_TWILIO_WHATSAPP_NUMBER

    if not account_sid or not auth_token or not from_number or "ACxx" in account_sid or "YOUR_" in account_sid:
        logging.warning("Hardcoded Twilio credentials incomplete or placeholders. WhatsApp alerts disabled.")
        return

    if not from_number.startswith('+'):
        logging.error(f"Hardcoded TWILIO_WHATSAPP_NUMBER '{from_number}' missing '+'. Alerts may fail.")

    try:
        twilio_client = Client(account_sid, auth_token)
        if not from_number.startswith('whatsapp:'):
            TWILIO_WHATSAPP_NUMBER_FORMATTED = f"whatsapp:{from_number}"
        else:
            TWILIO_WHATSAPP_NUMBER_FORMATTED = from_number
        logging.info(f"Twilio client initialized using number {TWILIO_WHATSAPP_NUMBER_FORMATTED}.")
    except Exception as e:
        logging.error(f"Failed to initialize Twilio client: {e}")
        twilio_client = None
        TWILIO_WHATSAPP_NUMBER_FORMATTED = None

def load_scam_models():
    """Loads the CountVectorizer and Naive Bayes model from pkl files."""
    global count_vectorizer, nb_model
    if not SKLEARN_AVAILABLE:
        logging.warning("Scikit-learn not available. Skipping scam model loading.")
        return
    try:
        logging.info(f"Attempting to load CountVectorizer from: {VECTORIZER_PATH}")
        if os.path.exists(VECTORIZER_PATH):
            count_vectorizer = joblib.load(VECTORIZER_PATH)
            logging.info("CountVectorizer loaded successfully.")
        else:
            logging.error(f"Vectorizer file not found: {VECTORIZER_PATH}")
            count_vectorizer = None

        logging.info(f"Attempting to load Naive Bayes model from: {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            nb_model = joblib.load(MODEL_PATH)
            logging.info("Naive Bayes model loaded successfully.")
        else:
            logging.error(f"Model file not found: {MODEL_PATH}")
            nb_model = None
    except FileNotFoundError as e:
         logging.error(f"Error loading scam models: File not found - {e}")
         count_vectorizer, nb_model = None, None
    except Exception as e:
        if "InconsistentVersionWarning" in str(e):
             logging.warning(f"Scikit-learn version mismatch loading models: {e}. Results may be inconsistent.")
        else:
            logging.error(f"An unexpected error occurred loading scam models: {e}", exc_info=True)

def initialize_chatbot():
    """Initializes Chatbot (Groq LLM, HuggingFace Embeddings), Vector Store, and RAG Chain."""
    global llm, embeddings, vectorstore, rag_chain
    if not LANGCHAIN_AVAILABLE:
        logging.error("Langchain components unavailable. Cannot initialize chatbot.")
        return False

    # --- Initialize LLM (Groq) ---
    try:
        groq_api_key = HARDCODED_GROQ_API_KEY
        if not groq_api_key or "YOUR_" in groq_api_key:
            raise ValueError("Hardcoded GROQ_API_KEY is missing or placeholder.")

        llm = ChatGroq(temperature=0.4, groq_api_key=groq_api_key, model_name="llama3-8b-8192")
        logging.info(f"Groq LLM initialized (model: {llm.model_name}).")
    except Exception as e:
        logging.error(f"Fatal: Error initializing Groq LLM: {e}", exc_info=True)
        llm = None
        return False

    # --- Initialize Embeddings (HuggingFace - Local) ---
    try:
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logging.info("HuggingFace Embeddings initialized (model: all-MiniLM-L6-v2).")
    except Exception as e:
        logging.error(f"Fatal: Error initializing HuggingFace Embeddings: {e}", exc_info=True)
        embeddings = None
        return False

    # --- Load or Build Vector Store ---
    vectorstore = None
    if os.path.exists(VECTORSTORE_PATH):
        logging.info(f"Existing vector store found at {VECTORSTORE_PATH}, but will be overwritten due to embedding model change.")
    else:
        logging.info(f"Vector store not found at {VECTORSTORE_PATH}. Building new.")

    # --- Build FAISS Index using local embeddings ---
    logging.info("Building new vector store with HuggingFace Embeddings...")
    if not os.path.exists(PDF_DIRECTORY):
        logging.error(f"PDF directory not found: {PDF_DIRECTORY}.")
        return False

    pdf_list = [f for f in os.listdir(PDF_DIRECTORY) if f.lower().endswith('.pdf')]
    if not pdf_list:
        logging.error(f"No PDF files found in: {PDF_DIRECTORY}")
        return False

    documents = []
    logging.info(f"Loading PDFs using {DEFAULT_PDF_LOADER.__name__}...")
    for pdf_file in pdf_list:
        pdf_path = os.path.join(PDF_DIRECTORY, pdf_file)
        try:
            loader = DEFAULT_PDF_LOADER(pdf_path)
            docs = loader.load()
            if docs:
                documents.extend(docs)
                logging.info(f"Loaded {len(docs)} pages from '{pdf_file}'")
            else:
                logging.warning(f"Loader returned no documents for {pdf_file}.")
        except Exception as e:
            logging.warning(f"Error loading {pdf_path}: {str(e)}")

    if not documents:
        logging.error("No documents loaded. Cannot build.")
        return False

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(split_docs)} document chunks.")

    if embeddings is None:
        logging.error("Embeddings unavailable for FAISS build.")
        return False

    try:
        logging.info("Generating LOCAL embeddings and building FAISS index (this may take time)...")
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        logging.info("FAISS index built successfully. Saving...")
        vectorstore.save_local(VECTORSTORE_PATH)
        logging.info(f"Vector store built locally and saved to {VECTORSTORE_PATH}.")
    except Exception as e:
        logging.error(f"Fatal: Error building/saving LOCAL vector store: {e}", exc_info=True)
        vectorstore = None
        return False

    # --- Define RAG Prompts & Chain ---
    if llm is None or vectorstore is None:
        logging.error("LLM or Vectorstore unavailable for RAG chain.")
        return False

    try:
        contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        rag_system_prompt = """You are 'FinBot', a friendly and certified financial educator on the FinGenius platform. Your primary goal is to provide accurate, clear, and helpful financial information based *strictly and exclusively* on the provided context documents below.

**Instructions:**
1.  **Answer based ONLY on Context:** If the answer to the user's question is present in the 'Context' section below, synthesize the information and provide a concise answer.
2.  **No External Knowledge:** Do NOT use any prior knowledge or information outside the provided 'Context'.
3.  **Acknowledge Limitations:** If the answer cannot be found within the 'Context', clearly state that you don't have information on that specific topic from the available FinEdu documents. Do not apologize excessively or offer to search elsewhere. Just state the limitation.
4.  **Do Not Advise:** Never provide financial advice (e.g., "you should buy X", "invest in Y"). Stick to explaining concepts and information found in the context.
5.  **Persona:** Maintain a helpful, professional, and educational tone.
6.  **Tailor Complexity:** Use the 'User Knowledge Profile' (Scale 0-20) to adjust your language. Lower scores need simpler terms; higher scores can handle more technical language, but always prioritize clarity.
7.  **Current Page Context:** Be aware of the 'Current Page Context' if provided, and try to relate your answer if relevant and supported by the document 'Context'.
8.  **Finance Focus:** Do not engage with or answer questions unrelated to finance or the content of the documents. Politely redirect back to financial topics.
9.  **Formatting:** When presenting lists, steps, or roadmaps (like the Trading Terminal content), **use Markdown bullet points (`*` or `-`) for clarity.** Use **bold text (`**text**`)** for headings or key terms within the list items where appropriate. Structure the response logically with clear sections if applicable. Start with a brief introductory sentence if presenting a list/roadmap. End lists/roadmaps derived from specific documents with a disclaimer like: "*Please note: This information is based on the provided documents and may not be fully comprehensive.*"

**User Knowledge Profile (0-20):**
{knowledge_levels}

**Current Page Context:**
{current_page_context}

**Context (Use ONLY this information):**
{context}

**Question:**
{input}

**Answer:**"""

        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", rag_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        logging.info("RAG chain created successfully (using local embeddings).")
        return True
    except Exception as e:
        logging.error(f"Fatal: Error creating RAG chain: {e}", exc_info=True)
        rag_chain = None
        return False


def load_stock_list(csv_path):
    global stock_list
    logging.info(f"Loading stocks from: {csv_path}")
    if not os.path.exists(csv_path): logging.error(f"CSV not found: {csv_path}"); stock_list = []; return False
    try:
        df = pd.read_csv(csv_path, usecols=['Symbol', 'Name'], delimiter=';'); df.dropna(subset=['Symbol', 'Name'], inplace=True)
        df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper(); df['Name'] = df['Name'].astype(str).str.strip()
        df.drop_duplicates(subset=['Symbol'], keep='first', inplace=True); temp_list = df.to_dict('records')
        if not temp_list: logging.warning(f"No stocks in {csv_path}."); stock_list = []; return False
        with stock_data_lock: stock_list = temp_list
        logging.info(f"Loaded {len(stock_list)} stocks.")
        return True
    except Exception as e: logging.error(f"Error loading stocks {csv_path}: {e}"); stock_list = []; return False

def get_live_stock_price(symbol):
    """
    Fetches the live stock/index/crypto price using yfinance,
    handling symbol variations more carefully.
    """
    original_symbol = symbol # Keep original for logging/errors if needed
    logging.debug(f"Fetching price for: {original_symbol}")
    yf_symbol = symbol.strip().upper()

    # --- CORRECTED SYMBOL LOGIC ---
    # Don't add .NS to known index formats or crypto
    if yf_symbol.startswith('^') or '-USD' in yf_symbol:
        logging.debug(f"Using symbol as is (index/crypto): {yf_symbol}")
    # Only add .NS if it doesn't end in .NS/.BO and isn't index/crypto
    elif not yf_symbol.endswith(('.NS', '.BO')):
        yf_symbol += '.NS'
        logging.debug(f"Appended .NS, using: {yf_symbol}")
    else:
         logging.debug(f"Using symbol as is (already suffixed): {yf_symbol}")
    # --- END CORRECTION ---

    try:
        stock = yf.Ticker(yf_symbol)
        # Use fast_info for potentially quicker price retrieval
        info = stock.fast_info
        price = info.get('last_price') # fast_info uses 'last_price'

        if price is not None and price > 0:
            return {'price': float(price), 'currency': info.get('currency', 'USD' if '-USD' in yf_symbol else 'INR'), 'source': 'fast_info'}
        else:
            # Fallback to the regular info dictionary if fast_info didn't work
            logging.debug(f"fast_info price missing/invalid for {yf_symbol}. Trying regular info...")
            info = stock.info # This can be slower
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if price is not None and price > 0:
                 return {'price': float(price), 'currency': info.get('currency', 'USD' if '-USD' in yf_symbol else 'INR'), 'source': 'info_dict'}
            else:
                 # Final fallback: try history
                 logging.debug(f"Regular info price missing/invalid for {yf_symbol}. Trying history...")
                 hist = stock.history(period='1d')
                 if not hist.empty:
                    close = hist['Close'].iloc[-1]
                    if close is not None and close > 0:
                         # Attempt to get currency from info even if price wasn't there
                        try:
                            currency = stock.info.get('currency', 'USD' if '-USD' in yf_symbol else 'INR')
                        except:
                            currency = 'USD' if '-USD' in yf_symbol else 'INR' # Default on error
                        return {'price': float(close), 'currency': currency, 'note': 'Last close', 'source': 'history'}

            # If all methods fail
            logging.warning(f"Could not get valid > 0 price for {yf_symbol} (Original: {original_symbol}) using any method.")
            return {"error": f"Could not retrieve valid price for {original_symbol} ({yf_symbol})"}

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error fetching {yf_symbol}: {e}")
        return {"error": f"Network connection error for {original_symbol}"}
    except Exception as e:
        # Check if it's a known yfinance issue related to invalid symbol after our logic
        err_str = str(e).lower()
        if "no data found for symbol" in err_str or "symbol may be delisted" in err_str or "'none' object has no attribute" in err_str:
             logging.warning(f"yfinance could not find data for '{yf_symbol}' (Original: {original_symbol}): {e}")
             return {"error": f"Symbol {original_symbol} ({yf_symbol}) not found or data unavailable"}
        # Log other unexpected errors more verbosely
        logging.error(f"Unexpected price fetch error for {yf_symbol} (Original: {original_symbol}): {e}", exc_info=True)
        return {"error": f"Error fetching price for {original_symbol}"}

def generate_title_from_url(url):
    try:
        lp = url.split('/')[-1];
        if '?' in lp: lp = lp.split('?')[0]
        if '#' in lp: lp = lp.split('#')[0]
        lp = os.path.splitext(lp)[0]; words = re.split(r'[-_.,~+=% ]', lp); words = [w for w in words if w and not w.isdigit()]
        if not words: return "Untitled Article"
        if len(words) > 1 and words[-1].isdigit() and len(words[-1]) > 4: words.pop()
        title = ' '.join(w.capitalize() for w in words if len(w) > 1 or w.isdigit());
        return title if len(title) > 5 else "Untitled Article"
    except Exception as e: logging.warning(f"Title gen error {url}: {e}"); return "Untitled Article"

def scrape_site(name, url, headers, selector, link_prefix="", link_base_url="", title_in_link=False):
    """Generic scraping function."""
    articles = []
    seen_urls = set()
    logging.info(f"Attempting to scrape {name} from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=25)
        response.raise_for_status()
        logging.info(f"{name} status: {response.status_code}")
        soup = BeautifulSoup(response.text, "html.parser")
        candidates = soup.select(selector)
        if not candidates:
            logging.warning(f"No candidates for selector '{selector}' on {name} ({url})")
            return []

        logging.info(f"Found {len(candidates)} potential elements on {name}.")
        count = 0
        for item in candidates:
            if count >= MAX_ARTICLES_PER_SOURCE:
                break

            link_tag = item if item.name == 'a' else item.find('a')
            if not link_tag:
                continue

            link = link_tag.get("href")
            if not link:
                continue

            # --- URL Handling ---
            if link.startswith('/') and link_base_url:
                link = urljoin(link_base_url, link)
            elif not link.startswith(('http://', 'https://')):
                # *** CORRECTED PART ***
                if link_base_url:
                    link = urljoin(link_base_url, link)
                    # Check again if it's now a valid http/https URL after joining
                    if not link.startswith(('http://', 'https://')):
                        logging.debug(f"Skipping invalidly resolved link: {link} on {name}")
                        continue # Skip to next item in loop
                else:
                    # If no base URL, skip non-absolute links
                    logging.debug(f"Skipping non-HTTP/relative link without base: {link} on {name}")
                    continue # Skip to next item in loop
                # *** END CORRECTION ***

            # --- Prefix Check and Deduplication ---
            if link in seen_urls:
                continue
            if link_prefix and not link.startswith(link_prefix):
                if not link_base_url or not link.startswith(link_base_url):
                     logging.debug(f"Skipping link due to prefix mismatch: {link} on {name}")
                     continue

            seen_urls.add(link)

            # --- Title Extraction ---
            title = ""
            if title_in_link: title = link_tag.get('title', '').strip()
            if not title: title = link_tag.get_text(strip=True)
            if not title:
                header_tag = link_tag.find(['h2', 'h3', 'h4', 'span'], {'class': re.compile(r'title|headline|heading', re.I)})
                if header_tag:
                    title = header_tag.get_text(strip=True)
            if not title or len(title) < 10:
                logging.debug(f"Title '{title}' seems short/empty, generating from URL: {link}")
                title = generate_title_from_url(link)

            articles.append({"title": title, "url": link})
            count += 1
            # --- End of loop ---

    except requests.exceptions.Timeout:
        logging.error(f"Timeout scraping {name}: {url}")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Request Exception scraping {name} ({url}): {e}")
        return []
    except Exception as e:
        logging.error(f"General Scraping Exception for {name} ({url}): {e}", exc_info=True)
        return []

    logging.info(f"Scraped {len(articles)} articles from {name}.")
    return articles

def scrape_all_sources():
    headers = {"User-Agent": "Mozilla/5.0..."}; sources = [{"name": "ZeeBusiness", "url": "https://www.zeebiz.com/latest-news", "selector": "div.newslist-sec a", "link_base_url": "https://www.zeebiz.com"}, {"name": "Moneycontrol", "url": "https://www.moneycontrol.com/news/business/", "selector": "li.clearfix h2 a, div.cat_listing li a", "link_prefix": "https://www.moneycontrol.com/news/", "title_in_link": True}, {"name": "EconomicTimes", "url": "https://economictimes.indiatimes.com/markets/stocks/news", "selector": ".eachStory a", "link_base_url": "https://economictimes.indiatimes.com"}, {"name": "BusinessStandard", "url": "https://www.business-standard.com/markets/news", "selector": ".listing-txt a.title", "link_base_url": "https://www.business-standard.com"}, {"name": "Mint", "url": "https://www.livemint.com/market/stock-market-news", "selector": "h2.headline a", "link_prefix": "https://www.livemint.com/"}, {"name": "NDTV Profit", "url": "https://www.ndtv.com/business/latest", "selector": ".news_Itm-cont a", "link_prefix": "https://www.ndtv.com/business/", "title_in_link": True} ]; results = []; threads = []
    def wrapper(cfg):
        try: results.extend(scrape_site(headers=headers, **cfg))
        except Exception as e: logging.error(f"Wrapper error {cfg.get('name')}: {e}")
    for s in sources: t = threading.Thread(target=wrapper, args=(s,), daemon=True, name=f"Scraper-{s.get('name')}"); threads.append(t); t.start()
    for t in threads: t.join(timeout=45)
    unique = {};
    for a in results:
        if a and isinstance(a, dict) and a.get('url'):
            if a['url'] not in unique: unique[a['url']] = a
        else: logging.warning(f"Invalid article skipped: {a}")
    unique_list = list(unique.values()); random.shuffle(unique_list); logging.info(f"Unique articles scraped: {len(unique_list)}"); return unique_list

def scrape_site(name, url, headers, selector, link_prefix="", link_base_url="", title_in_link=False):
    """Generic scraping function."""
    articles = []
    seen_urls = set()
    logging.info(f"Attempting to scrape {name} from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=25)
        response.raise_for_status()
        logging.info(f"{name} status: {response.status_code}")
        soup = BeautifulSoup(response.text, "html.parser")
        candidates = soup.select(selector)
        if not candidates:
            logging.warning(f"No candidates for selector '{selector}' on {name} ({url})")
            return []

        logging.info(f"Found {len(candidates)} potential elements on {name}.")
        count = 0
        for item in candidates:
            if count >= MAX_ARTICLES_PER_SOURCE:
                break

            link_tag = item if item.name == 'a' else item.find('a')
            if not link_tag:
                continue

            link = link_tag.get("href")
            if not link:
                continue

            # --- URL Handling ---
            if link.startswith('/') and link_base_url:
                link = urljoin(link_base_url, link)
            elif not link.startswith(('http://', 'https://')):
                # *** CORRECTED PART ***
                if link_base_url:
                    link = urljoin(link_base_url, link)
                    # Check again if it's now a valid http/https URL after joining
                    if not link.startswith(('http://', 'https://')):
                        logging.debug(f"Skipping invalidly resolved link: {link} on {name}")
                        continue # Skip to next item in loop
                else:
                    # If no base URL, skip non-absolute links
                    logging.debug(f"Skipping non-HTTP/relative link without base: {link} on {name}")
                    continue # Skip to next item in loop
                # *** END CORRECTION ***

            # --- Prefix Check and Deduplication ---
            if link in seen_urls:
                continue
            if link_prefix and not link.startswith(link_prefix):
                if not link_base_url or not link.startswith(link_base_url):
                     logging.debug(f"Skipping link due to prefix mismatch: {link} on {name}")
                     continue

            seen_urls.add(link)

            # --- Title Extraction ---
            title = ""
            if title_in_link: title = link_tag.get('title', '').strip()
            if not title: title = link_tag.get_text(strip=True)
            if not title:
                header_tag = link_tag.find(['h2', 'h3', 'h4', 'span'], {'class': re.compile(r'title|headline|heading', re.I)})
                if header_tag:
                    title = header_tag.get_text(strip=True)
            if not title or len(title) < 10:
                logging.debug(f"Title '{title}' seems short/empty, generating from URL: {link}")
                title = generate_title_from_url(link)

            articles.append({"title": title, "url": link})
            count += 1
            # --- End of loop ---

    except requests.exceptions.Timeout:
        logging.error(f"Timeout scraping {name}: {url}")
        return []
    except requests.exceptions.RequestException as e:
        logging.error(f"Request Exception scraping {name} ({url}): {e}")
        return []
    except Exception as e:
        logging.error(f"General Scraping Exception for {name} ({url}): {e}", exc_info=True)
        return []

    logging.info(f"Scraped {len(articles)} articles from {name}.")
    return articles
# Function 1
def initial_scrape_and_update():
    """Performs a single initial news scrape and updates the global list."""
    global latest_articles  # Declare global at the start
    logging.info("Performing initial news article scrape...")
    try:
        # Scrape all sources first
        arts = scrape_all_sources()
        # Use a lock to safely update the global variable
        with data_lock:
            latest_articles = arts # Assign the scraped articles
        # Log after successfully updating (and outside the lock)
        logging.info(f"Initial news scrape complete. Found {len(latest_articles)} articles.")
    except Exception as e:
        # Log the error if scraping fails
        logging.error(f"Initial news scrape failed: {e}", exc_info=True)
        logging.warning("Server starting with empty news list due to scrape failure.")
        # Ensure the list is empty in case of failure, using the lock
        with data_lock:
            latest_articles = []
def periodic_scraper_thread_func():
    """Background thread for periodic news scraping."""
    global latest_articles # Declare global at the start
    logging.info("News scraper thread started.")
    first_run = True # Use a different variable name

    while True:
        try: # Top-level try for the loop iteration
            if not first_run:
                logging.info(f"News task: Sleeping {SCRAPE_INTERVAL_SECONDS}s...")
                time.sleep(SCRAPE_INTERVAL_SECONDS)
            else:
                # Only set first_run to False after the first iteration logic
                first_run = False

            logging.info("News task: Starting scheduled scrape...")

            # --- Scrape Articles ---
            try: # Specific try block for the scraping process
                arts = scrape_all_sources()

                # Check the result of scraping
                if arts is not None: # Check if scrape didn't fail and return None
                    if arts: # Check if the list is not empty
                        with data_lock: # Use lock to update shared list
                            latest_articles = arts
                        logging.info(f"News task: Updated list ({len(arts)} items).")
                    else:
                        # Scrape succeeded but found 0 articles
                        logging.warning("News task: Scrape returned no articles (empty list). List not updated.")
                else:
                    # Scrape function itself failed and returned None
                    logging.error("News task: Scrape function failed and returned None. List not updated.")

            except Exception as scrape_e:
                logging.error(f"Error during scrape_all_sources call: {scrape_e}", exc_info=True)
            # --- End Scrape Articles ---

        except Exception as loop_e: # Catch errors in the main loop (like time.sleep)
            logging.error(f"Error in periodic scraper loop (outside scraping): {loop_e}", exc_info=True)
            # Prevent tight error loop if sleep fails or other loop error occurs
            time.sleep(60) # Sleep for a minute before retrying loop iteration

# Function 2 (Make sure this starts on a new line with no indentation)
def check_links(message):
    """Checks message for suspicious links based on keywords and domain length."""
    try: # <<< Main TRY block starts here
        # Regex to find potential URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)

        # List of keywords often found in scam/spam links
        suspicious_keywords = [
            'guaranteed', 'free', 'bonus', 'win', 'claim', 'reward', 'prize', 'urgent',
            'verify', 'login', 'account', 'update', 'confirm', 'limited', 'offer', 'cash',
            'instant', 'profit', 'hack', 'trick', 'secret', 'double', 'crypto', 'loan',
            'investment', 'earn', 'fast', 'easy', 'risk-free'
        ]
        flagged = [] # List to store flagged URLs

        # --- Loop through found URLs (should be INSIDE the main try block) ---
        for u in urls:
            low_u = u.lower()
            domain = "" # Initialize domain for this URL

            # --- Safely extract domain ---
            try: # Inner try for domain parsing
                if low_u.startswith(('http://', 'https://')):
                    domain = low_u.split('/')[2]
                else:
                    domain = low_u.split('/')[0]
            except IndexError:
                logging.warning(f"Could not parse domain from potential URL: {u}")
                domain = "" # Reset domain if parsing failed
            # --- End domain extraction ---

            keyword_match = False
            # Check if any suspicious keyword is in the URL
            if any(k in low_u for k in suspicious_keywords):
                flagged.append(u)
                keyword_match = True # Mark that we found a keyword match

            # If no keyword match, check for short domain heuristic
            elif domain and '.' in domain: # Use elif to avoid double flagging
                try: # Inner try for domain part splitting
                    domain_parts = domain.split('.')
                    # Check if the main part of the domain (before TLD) is short
                    # And exclude known shorteners
                    # Ensure domain_parts[0] exists
                    if domain_parts and len(domain_parts[0]) < 5 and not any(s in domain for s in ['bit.ly', 't.co', 'goo.gl', 'tinyurl.com']):
                        flagged.append(u)
                except Exception as dp_err:
                     logging.warning(f"Error parsing domain parts for {domain}: {dp_err}")
            # --- End of single URL processing ---

        # --- Return unique flagged URLs (still inside the main try block) ---
        return list(set(flagged))

    # --- Main EXCEPT block for the entire function ---
    except Exception as e:
        logging.error(f"Unexpected error during link check function: {e}", exc_info=True)
        return [] # Return empty list on any major error

def send_whatsapp_alert(alert):
    if not twilio_client or not TWILIO_WHATSAPP_NUMBER_FORMATTED: logging.error("Twilio unavailable. Cannot send."); return False
    num = alert.get('whatsapp_number', '').strip(); aid = alert.get('alert_id', 'UNK');
    if not re.match(r'^\+\d{10,15}$', num): logging.error(f"Invalid recipient {aid}: '{num}'."); return False
    to_num = f"whatsapp:{num}"; price = f"{alert.get('last_checked_price'):.2f}" if alert.get('last_checked_price') is not None else "N/A"; target = alert.get('target_price', 0.0); symbol = alert.get('symbol', 'N/A'); cond = alert.get('condition', '?')
    body = f"🚨 *Stock Alert!* 🚨\n\n*Symbol:* {symbol}\n*Condition Met:* Price {cond} {target:.2f}\n*Current Price:* {price}\n\n_(Alert ID: ...{aid[-6:]})_"
    try:
        logging.info(f"Sending Twilio {aid} to {to_num} from {TWILIO_WHATSAPP_NUMBER_FORMATTED}")
        msg = twilio_client.messages.create(from_=TWILIO_WHATSAPP_NUMBER_FORMATTED, body=body, to=to_num)
        if msg.status in ['queued', 'sending', 'sent', 'delivered']: logging.info(f"Twilio success ...{num[-4:]}. SID: {msg.sid}, Status: {msg.status}"); return True
        else: err = f"Code: {msg.error_code}" if msg.error_code else ""; logging.error(f"Twilio failure SID {msg.sid}. Status: {msg.status}. {err} Msg: {msg.error_message}"); return False
    except TwilioRestException as e:
        if e.code == 21614: logging.error(f"Twilio Err(21614): Recipient {to_num} opted-out/blocked {TWILIO_WHATSAPP_NUMBER_FORMATTED}.")
        elif e.code == 21211: logging.error(f"Twilio Err(21211): Invalid recipient {to_num}.")
        elif e.code == 21610: logging.error(f"Twilio Err(21610): To unsubscribed {to_num}.")
        else: logging.error(f"Twilio API err {aid} to {to_num}: Code {e.code}, {e}")
        logging.error(f"Failed FROM: {TWILIO_WHATSAPP_NUMBER_FORMATTED}")
        return False
    except Exception as e: logging.error(f"Unexpected send error {aid} to {to_num}: {e}", exc_info=True); return False

def check_alerts_periodically():
    """Background thread to check active alerts."""
    logging.info("Alert checker thread started.")
    while True:
        logging.debug("Checking alerts...")
        alerts_ids_to_check = []
        try: # Wrap lock access for getting IDs
            with alerts_lock:
                # Create a list of IDs to check to avoid modifying dict while iterating
                alerts_ids_to_check = [aid for aid, data in active_alerts.items() if data.get('status') == 'active']
        except Exception as lock_err:
             logging.error(f"Error accessing alerts_lock to get IDs: {lock_err}", exc_info=True)
             time.sleep(5) # Sleep briefly before retrying loop
             continue

        if not alerts_ids_to_check:
            logging.debug("No active alerts.")
        else:
            logging.info(f"Checking {len(alerts_ids_to_check)} alerts...")
            for aid in alerts_ids_to_check: # Use the copied list of IDs
                # --- Start processing single alert ---
                try: # Add a try block for processing each individual alert
                    alert_copy_to_send = None
                    alert_data_current = None

                    # --- Fetch current alert data safely ---
                    with alerts_lock:
                        alert_data_current = active_alerts.get(aid)
                        # *** CORRECTED PART ***
                        # Check again if it's still active after potentially waiting for lock
                        # Put the check on a new line with proper indentation
                        if not alert_data_current or alert_data_current.get('status') != 'active':
                            logging.debug(f"Alert {aid} inactive/gone upon check start. Skipping.")
                            continue # Skip to the next alert ID in the loop
                        # *** END CORRECTION ***

                    # Use fetched data outside the lock for network call
                    sym = alert_data_current['symbol']
                    target = alert_data_current['target_price']
                    cond = alert_data_current['condition']

                    # --- Get Live Price ---
                    price_res = get_live_stock_price(sym)
                    price = None
                    check_trigger = False

                    # --- Update Price/Error Status (inside lock) ---
                    if "price" in price_res:
                        price = price_res["price"]
                        check_trigger = True
                        with alerts_lock:
                            alert = active_alerts.get(aid) # Re-get inside lock before updating
                            if alert and alert.get('status') == 'active':
                                alert['last_checked_price'] = price
                                alert['check_error_count'] = 0
                    elif "error" in price_res:
                        logging.warning(f"Price fetch error {sym} (Alert {aid}): {price_res['error']}")
                        with alerts_lock:
                            alert = active_alerts.get(aid) # Re-get inside lock
                            if alert and alert.get('status') == 'active':
                                err_cnt = alert.get('check_error_count', 0) + 1
                                alert['check_error_count'] = err_cnt
                                if err_cnt >= 5:
                                    alert['status'] = 'error'
                                    alert['error_details'] = f"Failed fetch 5 times: {price_res['error']}"
                                    logging.error(f"Alert {aid} for {sym} disabled due to repeated errors.")
                    else:
                        logging.error(f"Unexpected price result {sym}: {price_res}")

                    # --- Trigger Check & Send Alert ---
                    if check_trigger and price is not None:
                        triggered = False
                        try:
                            curr_f = float(price); target_f = float(target)
                            if cond == '>=' and curr_f >= target_f: triggered = True
                            elif cond == '<=' and curr_f <= target_f: triggered = True
                            elif cond == '>' and curr_f > target_f: triggered = True
                            elif cond == '<' and curr_f < target_f: triggered = True
                        except Exception as comp_err:
                            logging.error(f"Price compare error {aid}: {comp_err}")
                            triggered = False

                        if triggered:
                            logging.info(f"ALERT TRIGGERED: ID={aid}, Sym={sym}, Cond=({cond} {target}), Curr={price}")
                            with alerts_lock:
                                alert_trig = active_alerts.get(aid)
                                if alert_trig and alert_trig.get('status') == 'active':
                                    alert_copy_to_send = alert_trig.copy()
                                    alert_copy_to_send['last_checked_price'] = price
                                else:
                                    logging.warning(f"Alert {aid} triggered but inactive before sending.")

                            if alert_copy_to_send:
                                success = send_whatsapp_alert(alert_copy_to_send)
                                if success:
                                    with alerts_lock:
                                        alert_final = active_alerts.get(aid)
                                        if alert_final and alert_final.get('status') == 'active':
                                            alert_final['status'] = 'triggered'
                                            alert_final['triggered_at'] = datetime.now(timezone.utc).isoformat()
                                            logging.info(f"Alert {aid} status set to triggered.")
                                        else:
                                            logging.warning(f"Alert {aid} sent but status changed before final update.")
                                else:
                                    logging.error(f"Notification failed for {aid}. Stays active.")

                except Exception as e: # Catch errors during processing of THIS alert
                    logging.error(f"Error processing alert {aid}: {e}", exc_info=True)
                    try: # Attempt to mark this specific alert as error
                        with alerts_lock:
                            alert = active_alerts.get(aid)
                            if alert and alert.get('status') != 'error':
                                alert['status'] = 'error'
                                alert['error_details'] = f"Check error: {str(e)[:100]}"
                                logging.error(f"Marked alert {aid} as error due to processing exception.")
                    except Exception as lock_err:
                        logging.error(f"Failed to mark alert {aid} as error after exception: {lock_err}")
                # --- End processing single alert ---
            # --- End of loop for alert IDs ---

        # Wait before starting the next full check cycle
        time.sleep(ALERT_CHECK_INTERVAL_SECONDS)

def update_knowledge(user_id: str, topic: str) -> dict:
    """Updates user knowledge profile and returns the current profile."""
    user_data = {}
    # Try loading writable file first, then input file, then empty
    try:
        # *** CORRECTED PART ***
        if os.path.exists(USER_KNOWLEDGE_FILE):
            # Indent the 'with' block under the 'if'
            with open(USER_KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
        # *** END CORRECTION ***
        else:
            # This 'else' corresponds to 'if os.path.exists(USER_KNOWLEDGE_FILE)'
            if os.path.exists(INPUT_KNOWLEDGE_FILE):
                with open(INPUT_KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                try:
                    with open(USER_KNOWLEDGE_FILE, 'w', encoding='utf-8') as wf:
                        json.dump(user_data, wf, indent=2)
                    logging.info(f"Copied initial knowledge data from {INPUT_KNOWLEDGE_FILE} to {USER_KNOWLEDGE_FILE}")
                except IOError as write_err:
                     logging.error(f"Could not copy initial knowledge data: {write_err}")
            else:
                 # This 'else' corresponds to 'if os.path.exists(INPUT_KNOWLEDGE_FILE)'
                 user_data = {} # Start fresh if neither exists
                 logging.info("No existing knowledge file found. Starting fresh.")

    except json.JSONDecodeError:
         logging.warning(f"Error decoding JSON from {USER_KNOWLEDGE_FILE}. Re-initializing.")
         user_data = {} # Start fresh on error
    except Exception as e:
        logging.warning(f"Error reading knowledge files: {e}. Starting fresh.")
        user_data = {} # Start fresh on other errors

    # Initialize profile if user doesn't exist
    if user_id not in user_data:
        user_data[user_id] = {
            "general": 0, "budgeting": 0, "investing": 0, "credit": 0, "taxes": 0,
            "retirement": 0, "insurance": 0, "stocks": 0, "mutual_funds": 0, "options": 0,
            "last_updated": None
        }
        logging.info(f"Created new knowledge profile for user {user_id}")

    # Get the profile dictionary (now guaranteed to exist)
    profile = user_data[user_id]

    # Apply knowledge decay
    last_updated_str = profile.get("last_updated")
    if last_updated_str:
        try:
            last_updated_dt = datetime.fromisoformat(last_updated_str).replace(tzinfo=timezone.utc)
            days_passed = (datetime.now(timezone.utc) - last_updated_dt).days
            if days_passed > 0:
                decay = 0.98 ** days_passed
                for key in profile:
                    if isinstance(profile[key], (int, float)) and key != "last_updated":
                        profile[key] = max(0, int(profile[key] * decay))
                logging.info(f"Applied knowledge decay for user {user_id} over {days_passed} days.")
        except Exception as e:
             logging.warning(f"Error calculating knowledge decay for user {user_id}: {e}")

    # Update topic score
    norm_topic = topic.lower().replace(" ", "_") if topic else "general"
    updated = False
    if norm_topic in profile and norm_topic != "last_updated":
        profile[norm_topic] = min(profile.get(norm_topic, 0) + 2, 20)
        updated = True
    else:
        for key in profile:
            if key != "last_updated" and key in norm_topic:
                profile[key] = min(profile.get(key, 0) + 2, 20)
                updated = True
                break
    if not updated and norm_topic not in ["other", "general"]:
        profile["general"] = min(profile.get("general", 0) + 1, 20)
    elif norm_topic == "general":
        profile["general"] = min(profile.get("general", 0) + 1, 20)

    profile["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Save updated data
    try:
        with open(USER_KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2)
    except IOError as e:
        logging.error(f"Error writing knowledge file {USER_KNOWLEDGE_FILE}: {e}")

    # Return the potentially modified profile dictionary
    return profile

def get_memory_for_user(user_id):
    global user_memories;
    if user_id not in user_memories:
        logging.info(f"New memory {user_id}"); user_memories[user_id] = ConversationBufferWindowMemory(memory_key="chat_history", input_key="input", output_key="answer", return_messages=True, k=5)
    return user_memories[user_id]

TOOL_KEYWORDS = { "sip": "/sip_calculator_standalone.html", "systematic investment plan": "/sip_calculator_standalone.html", "compound interest": "/financial-tools.html#compound-interest", "compounding": "/financial-tools.html#compound-interest", "savings": "/financial-tools.html#savings", "tax": "/financial-tools.html#income-tax", "income tax": "/financial-tools.html#income-tax", "nps": "/financial-tools.html#nps", "national pension scheme": "/financial-tools.html#nps", "insurance": "/financial-tools.html#life-insurance", "life insurance": "/financial-tools.html#life-insurance", "term plan": "/financial-tools.html#term-insurance", "health insurance": "/financial-tools.html#health-insurance", "fd": "/financial-tools.html#fd", "fixed deposit": "/financial-tools.html#fd", "ppf": "/financial-tools.html#ppf", "public provident fund": "/financial-tools.html#ppf", }
def find_relevant_tool_link(text):
    if not text: return None;
    low = text.lower()
    for k, u in TOOL_KEYWORDS.items():
        if re.search(r'\b' + re.escape(k) + r'\b', low):
            t = k.replace("_", " ").title()
            if "calculator" not in t.lower(): t += " Calculator"
            return {"title": t, "url": u}
    return None

def generate_quiz(context_text):
    """Generates a multiple-choice quiz question based on provided text using LLM."""
    global llm
    if not llm:
        logging.error("LLM not available for quiz generation.")
        return None
    if not context_text or len(context_text) < 100:
        logging.warning("Not enough context provided to generate quiz.")
        return None

    try: # Main try block for the LLM call
        logging.info("Attempting to generate quiz...")
        quiz_prompt = f"""Based on the key concepts in the following financial text, create one simple multiple-choice question (MCQ).
The question should have 3 options (A, B, C), with only one correct answer clearly identifiable from the text.
Format the output strictly as a valid JSON object with no surrounding text or markdown markers:
{{
"question": "Your question here?",
"options": {{
"A": "Option A text",
"B": "Option B text",
"C": "Option C text"
}},
"correct_option": "A"
}}

Context Text:
{context_text[:1500]}"""

        quiz_response = llm.invoke(quiz_prompt)
        response_content = quiz_response.content.strip()

        # --- Inner try block for JSON parsing ---
        json_str = None
        try:
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                # *** CORRECTED PART ***
                if json_start != -1 and json_end != -1:
                    json_str = response_content[json_start:json_end]
                else:
                    # If still no JSON block found, log and return
                    logging.warning(f"Could not extract JSON block from quiz response: {response_content}")
                    return None
                # *** END CORRECTION ***

            quiz_data = json.loads(json_str)

            # Validate structure
            if ('question' in quiz_data and 'options' in quiz_data and
                'correct_option' in quiz_data and isinstance(quiz_data['options'], dict) and
                len(quiz_data['options']) == 3 and quiz_data['correct_option'] in quiz_data['options']):
                logging.info("Quiz generated successfully.")
                return quiz_data
            else:
                logging.warning(f"LLM response for quiz was JSON but invalid structure: {json_str}")
                return None

        except json.JSONDecodeError as json_e:
            logging.warning(f"Failed to decode JSON from quiz response: {json_e}\nRaw String Attempted: '{json_str}'\nFull Response: {response_content}")
            return None
        # --- End Inner try-except ---

    except Exception as e: # Catch errors from llm.invoke or other unexpected issues
        logging.error(f"Error generating quiz (outside JSON parsing): {e}", exc_info=True)
        return None
@app.route(NEWS_API_ENDPOINT)
def get_finance_news_api():
    logging.info(f"API req: {NEWS_API_ENDPOINT}");
    try:
        with data_lock: data = list(latest_articles)
        logging.info(f"Sending {len(data)} news.")
        return jsonify(data)
    except Exception as e: logging.error(f"News API error: {e}", exc_info=True); return jsonify({"error": "Failed news."}), 500

@app.route(CHECK_SCAM_ENDPOINT, methods=['POST'])
def check_scam_api():
    logging.info(f"API req: {CHECK_SCAM_ENDPOINT}")
    if not SKLEARN_AVAILABLE or nb_model is None or count_vectorizer is None: logging.warning("Scam models/sklearn unavailable."); return jsonify({"error": "Scam checker unavailable."}), 503
    if not request.is_json: return jsonify({"error": "Need JSON"}), 415
    data = request.get_json()
    if not data: return jsonify({"error": "Empty body"}), 400
    msg = data.get('message', '').strip()
    if not msg: return jsonify({"error": "'message' required"}), 400
    try:
        if not hasattr(count_vectorizer, 'transform') or not hasattr(nb_model, 'predict_proba'): logging.error("Invalid models loaded."); raise TypeError("Invalid model/vectorizer")
        feats = count_vectorizer.transform([msg]); proba = nb_model.predict_proba(feats)
        if proba.shape == (1, 2): scam_prob = proba[0][1]
        else: logging.error(f"Shape error: {proba.shape}"); raise ValueError("Shape mismatch")
        scam_pct = round(scam_prob * 100, 2); links = check_links(msg); verdict = bool(scam_pct > SCAM_PROBABILITY_THRESHOLD)
        logging.info(f"Scam check: Prob={scam_pct}%, Verdict={verdict} for '{msg[:50]}...'")
        return jsonify({"scam_probability": scam_pct, "suspicious_links": links, "is_scam": verdict})
    except Exception as e: logging.error(f"Scam prediction error: {e}", exc_info=True); return jsonify({"error": "Analysis error."}), 500

@app.route(STOCKS_API_ENDPOINT)
def get_stock_list_api():
    logging.info(f"API req: {STOCKS_API_ENDPOINT}");
    with stock_data_lock:
        if not stock_list: logging.warning("Stock list empty."); return jsonify({"error": "Stocks unavailable.", "stocks": []}), 503
        data = list(stock_list)
    logging.info(f"Sending {len(data)} stocks.")
    return jsonify(data)

@app.route(f'{STOCK_PRICE_API_ENDPOINT}/<string:symbol>')
def get_stock_price_api(symbol):
    logging.info(f"API request received for stock price: {symbol}")
    if not symbol: return jsonify({"error": "Symbol cannot be empty"}), 400

    symbol_cleaned = symbol.strip().upper()
    now = time.time()

    # --- Check Cache ---
        # --- Check Cache ---
    with cache_lock:
        cached_data = price_cache.get(symbol_cleaned)
        if cached_data and (now - cached_data['timestamp'] < CACHE_EXPIRY_SECONDS):
            logging.info(f"Returning cached price for {symbol_cleaned}")
            # Return a copy to avoid modifying cache directly if needed later
            response_data = cached_data.copy()
            # Update timestamp in response for clarity, but use original symbol
            response_data['timestamp'] = now
            response_data['symbol'] = symbol # Use original requested symbol

            # --- FIX for TypeError ---
            # Handle potential None for 'note' before adding "(Cached)"
            existing_note = response_data.get('note') # Get the note, could be None or a string
            if isinstance(existing_note, str) and existing_note: # Check if it's a non-empty string
                response_data['note'] = existing_note + ' (Cached)'
            else: # If note was None, empty string, or not present, just add "(Cached)"
                response_data['note'] = '(Cached)'
            # --- END FIX ---

            return jsonify(response_data), 200
    # --- End Cache Check ---
    # --- End Cache Check ---

    logging.info(f"Cache miss or expired for {symbol_cleaned}. Fetching live price...")
    price_result = get_live_stock_price(symbol_cleaned) # Call yfinance

    if "error" in price_result:
        # (Keep existing error handling)
        status_code = 500; err_msg_lower = price_result["error"].lower()
        if "not found" in err_msg_lower or "invalid" in err_msg_lower: status_code = 404
        elif "network connection" in err_msg_lower: status_code = 504
        elif "could not retrieve" in err_msg_lower: status_code = 503
        logging.warning(f"Price error for {symbol_cleaned}: {price_result['error']} (Status: {status_code})")
        return jsonify({"error": price_result['error'], "symbol": symbol}), status_code
    else:
        # --- Update Cache ---
        response_symbol = symbol_cleaned.replace('.NS', '').replace('.BO', '')
        response_data = {
            'symbol': response_symbol, # Store cleaned symbol in cache key
            'price': price_result['price'],
            'currency': price_result.get('currency', 'INR'),
            'timestamp': now, # Use the fetch time for cache expiry check
            'note': price_result.get('note')
        }
        with cache_lock:
            price_cache[symbol_cleaned] = response_data # Store fetched data
        logging.info(f"Cached new price for {symbol_cleaned}")
        # --- End Update Cache ---

        # Return the fresh data (use original requested symbol for user)
        response_data_for_user = response_data.copy()
        response_data_for_user['symbol'] = symbol
        logging.info(f"API sending price data for {symbol_cleaned}: {response_data['price']}")
        return jsonify(response_data_for_user), 200

@app.route(ALERTS_API_ENDPOINT, methods=['POST'])
def create_alert_api():
    """API endpoint for creating alerts."""
    logging.info(f"API req: {ALERTS_API_ENDPOINT} POST")
    if not request.is_json:
        return jsonify({"error": "Need JSON request body"}), 415

    data = request.get_json()
    if not data:
        return jsonify({"error": "Empty request body"}), 400

    # Extract and validate data
    sym = data.get('symbol', '').strip().upper()
    price_str = data.get('target_price')
    num = data.get('whatsapp_number', '').strip()
    cond = data.get('condition', '>=').strip() # Default condition

    if not sym:
        return jsonify({"error": "'symbol' is required"}), 400

    # Validate symbol against loaded stock list
    with stock_data_lock:
        if not stock_list:
            logging.error("Alert creation error: Stock list unavailable.")
            return jsonify({"error": "Stocks unavailable."}), 503
        norm_sym = sym.replace('.NS', '').replace('.BO', '')
        if not any(s['Symbol'] == norm_sym for s in stock_list):
            logging.warning(f"Alert attempt for unknown symbol: {sym}")
            return jsonify({"error": f"Symbol '{sym}' not found in supported list."}), 400

    # Validate target price
    if price_str is None:
        return jsonify({"error": "'target_price' is required"}), 400
    # *** CORRECTED PRICE VALIDATION BLOCK ***
    try:
        price = float(price_str)
        # Check if price is positive *after* successful conversion
        if price <= 0:
            # Raise ValueError to be caught by the except block below
            raise ValueError("Target price must be a positive number.")
    except (ValueError, TypeError): # Catch errors from float() or the explicit raise
        return jsonify({"error": "Invalid 'target_price'. Must be a positive number."}), 400
    # *** END CORRECTION ***

    # Validate WhatsApp number
    if not num or not re.match(r'^\+\d{10,15}$', num):
        return jsonify({"error": "Invalid 'whatsapp_number'. Format: +CountryCodeNumber (11-16 digits total)."}), 400

    # Validate condition
    if cond not in ['>=', '<=', '>', '<']:
        return jsonify({"error": "Invalid 'condition'. Accepted values are '>=', '<=', '>', '<'."}), 400

    # Create alert data
    aid = str(uuid.uuid4())
    alert = {
        'alert_id': aid,
        'symbol': sym, # Store original symbol
        'target_price': price,
        'whatsapp_number': num,
        'condition': cond,
        'status': 'active', # Initial status
        'created_at': datetime.now(timezone.utc).isoformat(),
        'last_checked_price': None,
        'triggered_at': None,
        'check_error_count': 0,
        'error_details': None
    }

    # Store the alert
    try:
        with alerts_lock:
            active_alerts[aid] = alert
        logging.info(f"Stored alert: ID={aid}, Sym={sym}, Target={cond}{price}, Num=...{num[-4:]}")
        return jsonify({"message": "Alert set!", "alert_id": aid}), 201 # 201 Created
    except Exception as e:
        logging.error(f"Alert store error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error storing alert."}), 500
@app.route(CHAT_API_ENDPOINT, methods=['POST'])
def handle_chat_api():
    """Handles chat requests using RAG."""
    # Ensure RAG chain is ready
    if not LANGCHAIN_AVAILABLE or not rag_chain or not llm:
        logging.error("Chat API called but RAG/LLM/Langchain unavailable.")
        return jsonify({"error": "Chat service unavailable."}), 503

    data = request.get_json()
    if not data:
         return jsonify({"error": "Need JSON body"}), 400

    user_input = data.get('user_input')
    user_id = data.get('user_id')
    current_page_context = data.get('current_context', 'General Browsing')

    if not user_input or not user_id:
        return jsonify({"error": "user_input and user_id required"}), 400

    logging.info(f"API RAG Chat req user {user_id}. Input: '{user_input}'. Ctx: '{current_page_context}'")

    response_data = {}
    suggested_link = None
    quiz_data = None
    retrieved_context_for_quiz = None

    try: # Main try block for the entire chat handling logic
        # *** CORRECTED LINE ***
        # Check for keywords indicating a quiz request or a simple greeting
        is_quiz_request = any(k in user_input.lower() for k in ['quiz', 'test me', 'question'])
        is_greeting = any(k in user_input.lower() for k in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'])
        # *** END CORRECTION ***

        # Get User Memory
        user_memory = get_memory_for_user(user_id)
        memory_vars = user_memory.load_memory_variables({})
        chat_history_messages = memory_vars.get('chat_history', [])

        # --- Process based on intent ---
        if not is_greeting and not is_quiz_request: # Assume financial query
            # --- Inner try block for topic detection ---
            detected_topic = "general" # Default topic
            try:
                topic_response = llm.invoke(f"Main financial topic (investing, credit, SIP, stocks, retirement, budgeting, insurance, taxes, general, other) in: '{user_input}'")
                topic_content = topic_response.content.strip().lower().replace(" ", "_")
                # Check if the response is meaningful before assigning
                if topic_content and "other" not in topic_content:
                    detected_topic = topic_content
                logging.info(f"Detected topic: {detected_topic}")
            except Exception as topic_err:
                logging.warning(f"LLM topic detect failed: {topic_err}. Using 'general'.")
                # Keep detected_topic as "general"
            # --- End inner try-except ---

            # --- Continue with RAG ---
            knowledge_profile = update_knowledge(user_id, detected_topic)
            logging.debug(f"Invoking RAG. History len: {len(chat_history_messages)}")
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history_messages,
                "knowledge_levels": json.dumps(knowledge_profile),
                "current_page_context": current_page_context
            })
            response_data["answer"] = response.get("answer", "Sorry, no specific info found in FinEdu docs.")
            response_data["detected_topic"] = detected_topic # Return the determined topic
            logging.debug("RAG response received.")

            suggested_link = find_relevant_tool_link(user_input) or find_relevant_tool_link(response_data["answer"])
            retrieved_docs = response.get("context", [])
            if retrieved_docs:
                logging.debug(f"Retrieved {len(retrieved_docs)} docs.")
                retrieved_context_for_quiz = " ".join([doc.page_content for doc in retrieved_docs])
                if random.random() < 0.3:
                    quiz_data = generate_quiz(retrieved_context_for_quiz)
            else:
                logging.debug("No docs retrieved.")

        elif is_quiz_request:
            general_context = "Finance basics: budgeting, saving, investing, credit, insurance."
            quiz_data = generate_quiz(general_context)
            response_data["answer"] = "Okay, test your knowledge:" if quiz_data else "Couldn't generate quiz now!"

        elif is_greeting:
            response_data["answer"] = random.choice(["Hello! Ask about finance.", "Hi! Welcome to FinEdu.", "Greetings! How can I help?"])
            logging.info("Handled greeting.")

        else: # Fallback / Off-topic
            logging.info("Intent unclear/off-topic.")
            try:
                response_data["answer"] = llm.invoke(f"User: '{user_input}'. Politely steer back to FinEdu finance topics.").content
            except Exception as llm_err:
                logging.error(f"LLM redirect error: {llm_err}")
                response_data["answer"] = "Ask me about finance topics."

        # --- Assemble final JSON ---
        response_data["quiz"] = quiz_data
        response_data["suggested_links"] = [suggested_link] if suggested_link else []

        # --- Save context AFTER getting response ---
        try:
            user_memory.save_context({"input": user_input}, {"answer": response_data.get("answer", "")})
            logging.debug(f"Saved context {user_id}")
        except Exception as mem_err:
            logging.error(f"Failed save context {user_id}: {mem_err}")

        # --- Return successful response ---
        return jsonify(response_data)

    # --- Catch errors during the overall chat handling process ---
    except Exception as e:
        logging.error(f"Chat API error user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Problem processing request."}), 500
        
@app.route('/')
def index_route():
    scam_status = "Ready" if (SKLEARN_AVAILABLE and nb_model and count_vectorizer) else "UNAVAILABLE"
    with stock_data_lock: stock_count = len(stock_list); stock_list_status = f"Loaded ({stock_count})" if stock_list else "Not Loaded"
    with alerts_lock: alert_count = len(active_alerts); alert_api_status = f"Ready ({alert_count} Active)"
    with data_lock: news_count = len(latest_articles); news_api_status = f"Ready ({news_count} Articles)"
    message_service_status = "Twilio Configured" if twilio_client else "Twilio NOT Configured"
    chatbot_status = "Ready" if (LANGCHAIN_AVAILABLE and rag_chain) else "UNAVAILABLE"
    return (f"<h2>FinGenius Flask Backend - Status: Running</h2>"
            f"<b>Stock List API:</b> {stock_list_status}<br>"
            f"<b>Stock Price API:</b> Ready<br>"
            f"<b>Alerts API:</b> {alert_api_status}<br>"
            f"<b>News API:</b> {news_api_status}<br>"
            f"<b>Messaging Service:</b> {message_service_status}<br>"
            f"<b>Scam Check API:</b> {scam_status}<br>"
            f"<b>Chatbot API:</b> {chatbot_status}<br>"
            f"<hr><i>API Base URL: /api</i>")

if __name__ == '__main__':
    print("--- FinGenius Backend Starting ---")
    print("Loading stock list...")
    load_stock_list(STOCK_LIST_CSV_PATH)
    print("Loading scam checking models...")
    load_scam_models()
    print("Initializing Chatbot Components...")
    if not initialize_chatbot(): print("ERROR: Chatbot init failed. Chat API unavailable.")
    print("Performing initial news scrape...")
    initial_scrape_and_update()
    print("Initializing Twilio client...")
    initialize_twilio()
    print("Starting background threads...")
    threading.Thread(target=periodic_scraper_thread_func, name="NewsScraperThread", daemon=True).start()
    threading.Thread(target=check_alerts_periodically, name="AlertCheckerThread", daemon=True).start()
    print(f"Attempting to start server on port {FLASK_PORT}...")
    try:
        from waitress import serve
        print(f"Running with Waitress server on http://0.0.0.0:{FLASK_PORT}")
        serve(app, host="0.0.0.0", port=FLASK_PORT, threads=8) # Using Waitress
    except ImportError:
        print("Waitress not found. Using Flask dev server (testing only).")
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False) # Fallback
    logging.info("FinGenius Flask Backend Shut Down.")