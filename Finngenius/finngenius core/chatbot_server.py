# chatbot_server.py

import os
import logging
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import pickle
import time
from collections import deque
import re
# import pandas as pd # No longer needed as stock CSV removed
import requests
import yfinance as yf
from newsapi import NewsApiClient

# --- Configuration ---
load_dotenv()
RAG_DOCS_DIR = "rag_docs"; INDEX_FILE = "vector_index.faiss"; METADATA_FILE = "chunks_metadata.pkl"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'; GROQ_MODEL_NAME = 'llama3-8b-8192'
GROQ_API_KEY = os.getenv("GROQ_API_KEY"); NEWS_API_KEY = os.getenv("NEWS_API_KEY")
CHUNK_SIZE = 500; CHUNK_OVERLAP = 50; TOP_K_RESULTS = 3
SERVER_PORT = 5003; ALLOWED_ORIGIN = "http://127.0.0.1:3000","http://127.0.0.1:3001" # *** UPDATE THIS ***
MAX_HISTORY_TURNS = 5
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# --- Validation ---
if not GROQ_API_KEY: logging.error("FATAL: GROQ_API_KEY not found. Exiting."); exit(1)
if not NEWS_API_KEY: logging.warning("NewsAPI key not found. News fetching disabled.")

# --- Explicit Index/Commodity Mappings ---
EXPLICIT_ITEM_MAP = {
    "nifty": "^NSEI", "nifty 50": "^NSEI", "sensex": "^BSESN",
    "bank nifty": "^NSEBANK", "banknifty": "^NSEBANK",
    "dow jones": "^DJI", "dow": "^DJI", "nasdaq": "^IXIC",
    "s&p 500": "^GSPC", "s&p": "^GSPC",
    "gold": "GC=F", "gold price": "GC=F",
}

# --- Global Variables ---
app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": ALLOWED_ORIGIN}, r"/clear": {"origins": ALLOWED_ORIGIN}})
groq_client = Groq(api_key=GROQ_API_KEY)
newsapi = NewsApiClient(api_key=NEWS_API_KEY) if NEWS_API_KEY else None
embedding_model = None; vector_index = None; chunks_with_metadata = []
chat_history = deque(maxlen=MAX_HISTORY_TURNS * 2)

# --- Helper Functions (chunk_text, get_latest_doc_mtime) ---
def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def get_latest_doc_mtime(directory_path):
    latest_mtime = 0
    try:
        if not os.path.exists(directory_path): return 0
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".pdf"): filepath = os.path.join(directory_path, filename);
            try: mtime = os.path.getmtime(filepath); latest_mtime = max(latest_mtime, mtime)
            except OSError as e: logging.warning(f"Could not get mtime for {filepath}: {e}")
    except Exception as e: logging.error(f"Error checking doc mtimes in {directory_path}: {e}"); return -1
    return latest_mtime

# --- Data Loading / Processing Functions (process_pdfs) ---
# Removed load_stock_data
def process_pdfs(directory_path):
    processed_chunks = []; pdf_count = 0; logging.info(f"Starting PDF processing: '{directory_path}'..."); start_time = time.time()
    try:
        if not os.path.exists(directory_path): logging.warning(f"RAG dir missing: '{directory_path}'"); return []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".pdf"):
                pdf_count += 1; filepath = os.path.join(directory_path, filename); logging.info(f"Processing PDF: {filename}...")
                try:
                    doc = fitz.open(filepath)
                    for page_num, page in enumerate(doc):
                        text = page.get_text("text")
                        if text and text.strip():
                            for chunk in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP): processed_chunks.append({"text": chunk, "source": f"{filename}-p{page_num + 1}"})
                    doc.close()
                except Exception as e: logging.error(f"Error processing {filename}: {e}")
        logging.info(f"PDF processing done ({time.time() - start_time:.2f}s). Processed {pdf_count} PDFs, extracted {len(processed_chunks)} chunks.")
        if not processed_chunks and pdf_count > 0: logging.warning("No text chunks extracted.")
        return processed_chunks
    except Exception as e: logging.error(f"PDF processing error: {e}", exc_info=True); return []

# --- Indexing Functions (build_and_save_index, load_index_and_metadata) ---
def build_and_save_index(chunks_data):
    global embedding_model; local_vector_index = None
    if not chunks_data: logging.warning("No chunks to build index."); return None, None
    try:
        if embedding_model is None: logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}..."); start_time_model = time.time(); embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME); logging.info(f"Model loaded ({time.time() - start_time_model:.2f}s).")
        logging.info(f"Creating embeddings for {len(chunks_data)} chunks..."); start_time_emb = time.time(); embeddings = embedding_model.encode([c['text'] for c in chunks_data], show_progress_bar=True); embeddings = np.array(embeddings).astype('float32'); logging.info(f"Embeddings created ({embeddings.shape}, {time.time() - start_time_emb:.2f}s).")
        logging.info("Building FAISS index..."); start_time_idx = time.time(); index_dimension = embeddings.shape[1]; local_vector_index = faiss.IndexFlatL2(index_dimension); local_vector_index.add(embeddings); logging.info(f"FAISS index built ({local_vector_index.ntotal}, {time.time() - start_time_idx:.2f}s).")
        logging.info(f"Saving index to {INDEX_FILE}..."); faiss.write_index(local_vector_index, INDEX_FILE); logging.info(f"Saving metadata to {METADATA_FILE}...");
        with open(METADATA_FILE, 'wb') as f: pickle.dump(chunks_data, f); logging.info("Index/metadata saved."); return local_vector_index, chunks_data
    except Exception as e: logging.error(f"Index build/save error: {e}", exc_info=True); return None, None
def load_index_and_metadata():
    global embedding_model; loaded_index = None; loaded_metadata = None
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE): logging.info("Index/metadata missing."); return None, None
    try:
        logging.info(f"Loading FAISS index from {INDEX_FILE}..."); start_time = time.time(); loaded_index = faiss.read_index(INDEX_FILE); logging.info(f"Index loaded ({loaded_index.ntotal}, {time.time() - start_time:.2f}s).")
        logging.info(f"Loading metadata from {METADATA_FILE}..."); start_time = time.time();
        with open(METADATA_FILE, 'rb') as f: loaded_metadata = pickle.load(f); logging.info(f"Metadata loaded ({len(loaded_metadata)}, {time.time() - start_time:.2f}s).")
        if embedding_model is None: logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}..."); start_model_load = time.time(); embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME); logging.info(f"Model loaded ({time.time() - start_model_load:.2f}s).")
        return loaded_index, loaded_metadata
    except Exception as e: logging.error(f"Error loading index/metadata: {e}", exc_info=True); return None, None

# --- External API Functions (get_live_stock_data, get_latest_financial_news) ---
def get_live_stock_data(ticker): # Fetches indices/commodities too
    logging.info(f"Attempting yfinance fetch for: {ticker}")
    try:
        item = yf.Ticker(ticker); hist = item.history(period="2d")
        if hist.empty: info = item.info; logging.debug(f"yfinance history empty for {ticker}.")
        else: info = {'currentPrice': hist['Close'].iloc[-1]}; info.update(item.info); info['currentPrice'] = hist['Close'].iloc[-1]; logging.debug(f"yfinance history found for {ticker}.")
        quote_type = info.get('quoteType', '').upper(); has_price = info.get('regularMarketPrice') is not None or info.get('currentPrice') is not None
        if not info or not has_price: logging.warning(f"yfinance insufficient data for {ticker}."); return None
        price = info.get('currentPrice', info.get('regularMarketPrice')); change = info.get('regularMarketChange'); change_percent = info.get('regularMarketChangePercent')
        data = {"symbol": info.get("symbol", ticker), "name": info.get("shortName", info.get("longName", "N/A")), "price": price, "change": change, "changePercent": change_percent, "market": info.get("exchangeName", info.get("exchange", "N/A")), "currency": info.get("currency", "N/A"), "volume": info.get("regularMarketVolume", info.get("volume", "N/A")), "marketCap": info.get("marketCap", "N/A")}
        cleaned_data = {k: v for k, v in data.items() if v is not None and v != "N/A"}
        if 'price' not in cleaned_data: logging.warning(f"yfinance fetch OK but price missing after cleaning for {ticker}."); return None
        logging.info(f"yfinance fetch successful for: {ticker}")
        return cleaned_data
    except Exception as e: logging.error(f"yfinance exception fetching {ticker}: {e}"); return None
def get_latest_financial_news(query="finance OR stock market", country="in", max_articles=5):
    if not newsapi: logging.warning("NewsAPI disabled."); return None
    search_params = {'q': query, 'language': 'en', 'sort_by': 'publishedAt', 'page_size': max_articles}
    if country: search_params['country'] = country; search_params['category'] = 'business'; api_method = newsapi.get_top_headlines; logging.info(f"Fetching top headlines (country='{country}')")
    else: api_method = newsapi.get_everything; logging.info(f"Fetching global news (query='{query}')")
    try:
        response_data = api_method(**search_params); articles_out = []
        if response_data and response_data.get('status') == 'ok':
            articles_in = response_data.get('articles', []); logging.info(f"NewsAPI OK. Received: {len(articles_in)} articles.")
            if not articles_in: logging.warning("NewsAPI OK but article list empty."); return None
            for article in articles_in: articles_out.append({"title": article.get('title'), "source": article.get('source', {}).get('name'), "url": article.get('url'), "publishedAt": article.get('publishedAt')})
            return articles_out
        else: error_msg = response_data.get('message', 'Unknown') if response_data else 'No response'; logging.error(f"NewsAPI error: {error_msg}"); return None
    except Exception as e: logging.error(f"NewsAPI exception: {e}", exc_info=True); return None

# --- RAG Search Function ---
def search_similar_chunks(query, k=TOP_K_RESULTS):
    if vector_index is None or embedding_model is None: logging.error("RAG components not ready."); return []
    if not chunks_with_metadata: logging.warning("No RAG chunks loaded."); return []
    if k <= 0: logging.warning("RAG TOP_K must be > 0."); return []
    try:
        logging.info(f"Encoding query for RAG: '{query[:50]}...'"); query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        logging.info(f"Searching RAG index for top {k}..."); distances, indices = vector_index.search(query_embedding, k)
        results = [];
        if indices.size > 0: valid_indices = [i for i in indices[0] if 0 <= i < len(chunks_with_metadata)]; results = [chunks_with_metadata[i] for i in valid_indices]
        logging.info(f"Found {len(results)} relevant RAG chunks."); return results
    except Exception as e: logging.error(f"RAG search error: {e}", exc_info=True); return []

# --- Core LLM Interaction Function ---
def generate_response_with_groq(query, context_chunks, history, live_data=None, intent="unknown"):
    live_data = live_data or {}; prompt_sections = []; has_live_data = False; analysis_prompt_addition = ""
    use_rag = intent in ["general_chat", "get_definition", "unknown"]

    # Prepare Live Data String
    live_data_str = ""
    if live_data.get('market_summaries') or live_data.get('news'):
        has_live_data = True; live_data_str += f"--- Start of Today's Live Data ---\n"
        if live_data.get('market_summaries'):
            live_data_str += "Current Market Data:\n";
            for name, summary in live_data['market_summaries'].items():
                change_pct_val = summary.get('changePercent','N/A'); change_val = summary.get('change', 'N/A'); price_val = summary.get('price', 'N/A'); currency_val = summary.get('currency', '')
                if isinstance(change_pct_val,(int, float)):
                    try: change_pct_s = f"{change_pct_val * 100:.2f}%" if -1.01 < change_pct_val < 1.01 else f"{change_pct_val:.2f}%"
                    except TypeError: change_pct_s = change_pct_val
                else: change_pct_s = change_pct_val
                live_data_str += f"  {summary.get('name', name)}: {price_val} {currency_val} (Change: {change_val}, {change_pct_s})\n"
            live_data_str += "\n"
            if intent == "get_analysis": analysis_prompt_addition = f"User is asking for analysis. Provide brief analysis based ONLY on live market data/news."
        if live_data.get('news'):
            live_data_str += "Recent Financial News Headlines (Max 3):\n"
            for i, news in enumerate(live_data['news']):
                if i >= 3: break; live_data_str += f"  {i+1}. {news.get('title','N/A')} ({news.get('source','N/A')})\n"
            live_data_str += "\n"
        live_data_str += f"--- End of Today's Live Data ---\n"; prompt_sections.append(live_data_str)

    # Prepare RAG Context String (Conditionally)
    rag_context_str = ""
    if use_rag and context_chunks:
        context_content = "\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['text']}" for chunk in context_chunks])
        rag_context_header = "--- Start of Document Context (Background Info) ---"; rag_context_str = (f"{rag_context_header}\n{context_content}\n--- End of Document Context ---\n");
        prompt_sections.append(rag_context_str); logging.info(f"Included {len(context_chunks)} RAG chunks for intent: {intent}")
    elif context_chunks: logging.info(f"Query intent is '{intent}'; OMITTING {len(context_chunks)} RAG chunks.")

    # Combine Supporting Info
    supporting_info = "\n\n".join(prompt_sections)
    if not supporting_info: supporting_info = "NOTE: No specific live data or document context available."

    # Select System Prompt Based on Intent
    system_prompt_content = ( # Default V4 prompt
        "You are a highly knowledgeable and conversational **AI Financial Assistant** designed to help users with real-time market insights, conceptual understanding, and financial guidance.\n\n"

    "**🧠 RESPONSE STRATEGY:**\n"
    "1. **Live Market Data (Indices, Gold, News):**\n"
    "   - If user asks about *today’s* market status, index levels, gold prices, or finance news, rely **ONLY on the 'Live Data'** or 'Recent News Headlines'.\n"
    "   - If live data exists, summarize it naturally and conversationally. Highlight key movements (up/down), sentiment, and trends.\n"
    "\n"
    "2. **Live Data Unavailable:**\n"
    "   - If required live data is MISSING or EMPTY, clearly inform the user:\n"
    "     👉 'I couldn’t fetch live data for that right now.'\n"
    "   - ⚠️ DO NOT use 'Document Context' or assumptions in place of live data.\n"
    "\n"
    "3. **Conceptual or General Finance Topics (Land, FD, SIP, etc.):**\n"
    "   - Use your general financial knowledge to explain clearly.\n"
    "   - You MAY enhance responses with relevant 'Document Context' if available, but prioritize **clarity, simplicity, and accuracy**.\n"
    "\n"
    "4. **Conversation Memory:**\n"
    "   - Respect prior context and user preferences.\n"
    "   - Avoid repeating already shared info.\n"
    "\n"
    "5. **Stock Prices Restriction:**\n"
    "   - ❌ DO NOT provide prices or forecasts for individual stocks (e.g., TCS, Apple, Reliance).\n"
    "   - ✅ You MAY discuss broader indices (e.g., NIFTY, NASDAQ) and market sectors in general terms.\n\n"

    "**💬 OUTPUT STYLE:**\n"
    "- Use a **natural, friendly tone** – like a helpful financial expert.\n"
    "- Keep answers **concise, clear, and direct.**\n"
    "- ❗ If live data is unavailable, mention it upfront without fallback guesses.\n"
    "- Avoid robotic or overly formal phrasing.\n"
    "- Never repeat context headers like '--- Start/End ---'.\n"
    "- Politely refuse inappropriate or irrelevant requests.\n"
    )
    if intent == "get_price":
        system_prompt_content = (
        "You are a precise and concise **Market Data Reporter**.\n"
        "1. Provide **ONLY** current price or level of indices (e.g., Nifty 50) or commodities (e.g., Gold), using the 'Live Data' section.\n"
        "2. Format clearly: 'Nifty 50 is currently at 22,345.'\n"
        "3. If no data is available, say: 'I couldn't fetch live data for Nifty 50.'\n"
        "4. ❌ DO NOT include individual stock prices.\n"
        "5. ❌ DO NOT use Document Context."
    )

    
    
    elif intent == "get_analysis":
        system_prompt_content = (
        "You are an expert **Market Analyst**.\n"
        "1. Analyze **ONLY** what's in the 'Live Data' or 'Recent News Headlines'.\n"
        "2. Provide a short market outlook – direction, volatility, investor sentiment.\n"
        "3. Example: 'Markets are mildly bearish today amid global slowdown concerns.'\n"
        "4. ⚠️ If no live data exists, clearly state you cannot analyze right now.\n"
        "5. ❌ DO NOT analyze or mention individual stocks.\n"
        "6. ❌ DO NOT use Document Context."
    )

    
    elif intent == "get_news":
     system_prompt_content = (
        "You are a friendly **Financial News Curator**.\n"
        "1. Summarize 2–3 of the most relevant headlines from the 'Recent News Headlines'.\n"
        "2. Sound natural and informative.\n"
        "3. If no news is present, say: 'There’s no latest financial news available at the moment.'\n"
        "4. ❌ Do NOT use Document Context or guess.\n"
    )

    
    elif intent == "get_definition":
        system_prompt_content = (
        "You are a clear and engaging **Financial Educator**.\n"
        "1. Explain the topic (e.g., SIP, Land Investment, FD) in simple terms.\n"
        "2. Optionally include helpful examples or analogies.\n"
        "3. You MAY use 'Document Context' if it adds relevant detail, but keep clarity first.\n"
        "4. Use bullet points or short paragraphs if needed for better understanding.\n"
    )

    # Construct Final Messages
    messages = [{"role": "system", "content": system_prompt_content}]
    if history: messages.extend(list(history))
    final_user_content = f"{supporting_info}\n\n{analysis_prompt_addition}\n\nUser Question: {query}"
    messages.append({"role": "user", "content": final_user_content})

    logging.info(f"Sending request to Groq ({GROQ_MODEL_NAME}) with intent '{intent}'. Messages: {len(messages)}")
    # print(f"\n--- FINAL PROMPT (Intent: {intent}) ---\nSystem: {messages[0]['content']}\nUser: {final_user_content}\n---------------------------\n")

    try:
        chat_completion = groq_client.chat.completions.create(messages=messages, model=GROQ_MODEL_NAME, temperature=0.4)
        response = chat_completion.choices[0].message.content; cleaned_response = response.strip()
        logging.info("Received response from Groq."); return cleaned_response
    except Exception as e: logging.error(f"Error calling Groq API: {e}", exc_info=True); return "Sorry, an error occurred processing the request."

# --- Initialization Function ---
def initialize_rag():
    global vector_index, chunks_with_metadata, embedding_model, chat_history
    # Removed load_stock_data() call
    logging.info("Starting RAG system initialization...")
    force_rebuild = False; index_mtime = 0
    if os.path.exists(INDEX_FILE):
        try: index_mtime = os.path.getmtime(INDEX_FILE)
        except OSError: force_rebuild = True; logging.warning(f"Could not get mtime for {INDEX_FILE}.")
    else: force_rebuild = True; logging.info(f"{INDEX_FILE} not found.")
    if not force_rebuild:
        if not os.path.exists(RAG_DOCS_DIR): logging.warning(f"RAG dir '{RAG_DOCS_DIR}' missing.")
        else:
            latest_doc_mtime = get_latest_doc_mtime(RAG_DOCS_DIR)
            if latest_doc_mtime == -1: force_rebuild = True; logging.warning("Error checking PDF mtimes.")
            elif latest_doc_mtime > index_mtime: force_rebuild = True; logging.info("PDFs modified.")
            elif not any(f.lower().endswith(".pdf") for f in os.listdir(RAG_DOCS_DIR) if os.path.isfile(os.path.join(RAG_DOCS_DIR, f))): logging.warning(f"No PDFs in '{RAG_DOCS_DIR}'.")
    if not force_rebuild:
        logging.info("Attempting to load existing RAG index/metadata...")
        loaded_index, loaded_metadata = load_index_and_metadata()
        if loaded_index is not None and loaded_metadata is not None: vector_index = loaded_index; chunks_with_metadata = loaded_metadata; logging.info("Loaded RAG index/metadata.")
        else: logging.warning("Failed load. Forcing rebuild."); force_rebuild = True
    if force_rebuild:
        if not os.path.exists(RAG_DOCS_DIR) or not any(f.lower().endswith(".pdf") for f in os.listdir(RAG_DOCS_DIR) if os.path.isfile(os.path.join(RAG_DOCS_DIR, f))):
            logging.error(f"Cannot rebuild RAG: Dir '{RAG_DOCS_DIR}' missing or no PDFs. RAG disabled."); vector_index = None; chunks_with_metadata = []
            if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE);
            if os.path.exists(METADATA_FILE): os.remove(METADATA_FILE);
            if embedding_model is None: # Try load model anyway
                 # *** CORRECTED SYNTAX HERE ***
                 try:
                     logging.info(f"(Rebuild failed) Loading embedding model: {EMBEDDING_MODEL_NAME}...")
                     embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                     logging.info("(Rebuild failed) Embedding model loaded.")
                 except Exception as e:
                     logging.error(f"Failed loading embedding model even after rebuild failure: {e}")
        else:
            logging.info("Rebuilding RAG index from PDFs..."); processed_chunks = process_pdfs(RAG_DOCS_DIR)
            if processed_chunks:
                new_index, new_metadata = build_and_save_index(processed_chunks)
                if new_index is not None and new_metadata is not None: vector_index = new_index; chunks_with_metadata = new_metadata; logging.info("RAG rebuild successful.")
                else: logging.error("RAG rebuild failed."); vector_index = None; chunks_with_metadata = []
            else: logging.error("Failed processing PDFs for rebuild."); vector_index = None; chunks_with_metadata = []
    chat_history.clear(); logging.info(f"Chat history initialized (max turns: {MAX_HISTORY_TURNS}).")
    if vector_index is None or not chunks_with_metadata: logging.warning("RAG system init done, RAG NOT available.")
    else: logging.info(f"RAG system init done. RAG ready ({vector_index.ntotal} chunks).")
    # Removed stock CSV status log

# --- Simple Intent Classification ---
def classify_intent(query_lower):
    # Focus on intents relevant to indices, gold, news, concepts
    if ("price of" in query_lower or "how much is" in query_lower or query_lower.endswith(" price")) and \
       any(key in query_lower for key in EXPLICIT_ITEM_MAP):
        return "get_price"
    if ("analysis of" in query_lower or "analyze" in query_lower or "how is" in query_lower or "status of" in query_lower or "market summary" in query_lower or "market overview" in query_lower) and \
       (any(key in query_lower for key in EXPLICIT_ITEM_MAP) or "market" in query_lower):
         return "get_analysis"
    if "news" in query_lower or "headlines" in query_lower: return "get_news"
    if "what is" in query_lower or "what are" in query_lower or "explain" in query_lower or "define" in query_lower or "teach me" in query_lower or "how to" in query_lower or "land investment" in query_lower or "fixed deposit" in query_lower or " fd " in query_lower:
        if not any(kw in query_lower for kw in ["how is", "how much", "price of", "analysis of"]): return "get_definition" # Avoid overlap
    # Add specific denial for individual stocks if needed
    # stock_company_keywords = [...] # Define keywords indicating specific companies not in EXPLICIT_ITEM_MAP
    # if any(kw in query_lower for kw in stock_company_keywords): return "deny_stock_specific"
    return "general_chat"

# --- Flask API Endpoints ---
@app.route('/query', methods=['POST'])
def handle_query():
    global chat_history, EXPLICIT_ITEM_MAP
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(); query = data.get('query')
    if not query: return jsonify({"error": "Missing 'query'"}), 400

    query_lower = query.lower(); live_data = {}; found_item_symbol = None

    # 1. Classify Intent
    intent = classify_intent(query_lower); logging.info(f"Query: '{query}' | Intent: {intent}")

    # 2. Pre-computation: Fetch Live Data (Indices, Gold, News) based on Intent
    logging.info(f"Starting data fetch for intent '{intent}'...")
    items_to_fetch = [] # List of tuples (display_name, yfinance_symbol)

    # A. Identify Target Item (Index/Gold) if intent needs it
    if intent in ["get_price", "get_analysis"]:
        target_item_found = False; query_words = query.split()
        for phrase_len in range(min(5, len(query_words)), 0, -1):
            phrase = " ".join(query_words[:phrase_len]).lower()
            if phrase in EXPLICIT_ITEM_MAP: found_item_symbol = EXPLICIT_ITEM_MAP[phrase]; target_item_found = True; logging.info(f"Explicit map: '{phrase}' -> {found_item_symbol}"); break
        if not target_item_found:
            for word in query_words:
                if word in EXPLICIT_ITEM_MAP: found_item_symbol = EXPLICIT_ITEM_MAP[word]; target_item_found = True; logging.info(f"Explicit map: '{word}' -> {found_item_symbol}"); break
        if not target_item_found: logging.info("No specific index/gold item identified in query.")

    # B. Determine what data to fetch based on intent and identified item
    if found_item_symbol and intent == "get_price":
        item_name = [k for k, v in EXPLICIT_ITEM_MAP.items() if v == found_item_symbol][0].title()
        items_to_fetch.append((item_name, found_item_symbol))
    elif intent == "get_analysis":
        if found_item_symbol: # Fetch specific item for analysis
            item_name = [k for k, v in EXPLICIT_ITEM_MAP.items() if v == found_item_symbol][0].title()
            items_to_fetch.append((item_name, found_item_symbol))
        # Always fetch defaults for general market context during analysis
        items_to_fetch.extend([('Nifty 50', '^NSEI'), ('Sensex', '^BSESN'), ('Bank Nifty', '^NSEBANK'), ('Gold', 'GC=F')])

    # C. Fetch Market Data for required items
    if items_to_fetch:
        items_to_fetch = list(dict.fromkeys(items_to_fetch)) # Remove duplicates
        market_summaries = {}
        logging.info(f"Fetching market data for: {[f'{name}({ticker})' for name, ticker in items_to_fetch]}")
        for name, ticker_symbol in items_to_fetch:
            summary = get_live_stock_data(ticker_symbol)
            if summary: market_summaries[name] = summary; market_summaries[name]['name'] = summary.get('name', name)
            else: logging.warning(f"Failed yfinance fetch for {name} ({ticker_symbol})")
        if market_summaries: live_data['market_summaries'] = market_summaries

    # D. Fetch News (if intent is news)
    if intent == "get_news":
        logging.info("Detected news intent..."); target_country = 'in'; query_term = "finance OR stock market OR economy"
        if "global" in query_lower or "world" in query_lower: target_country = None
        elif "us news" in query_lower: target_country = 'in'
        news_articles = get_latest_financial_news(query=query_term, country=target_country)
        if news_articles: live_data['news'] = news_articles
        else: logging.warning("News fetch failed or no articles.")

    logging.info(f"Live data fetch attempt done. Keys available: {list(live_data.keys())}")

    # 3. RAG Retrieval (Run regardless, used conditionally in prompt)
    relevant_chunks = search_similar_chunks(query)

    # 4. Call LLM
    answer = generate_response_with_groq(
        query=query, context_chunks=relevant_chunks, history=list(chat_history),
        live_data=live_data, intent=intent # Pass determined intent
    )

    # 5. Update History
    chat_history.append({"role": "user", "content": query}); chat_history.append({"role": "assistant", "content": answer})
    return jsonify({"answer": answer})

@app.route('/clear', methods=['POST'])
def clear_history():
    global chat_history; chat_history.clear(); logging.info("Chat history cleared."); return jsonify({"message": "Chat history cleared."})

# --- Main Execution ---
if __name__ == '__main__':
    initialize_rag()
    logging.info(f"Starting Flask server on port {SERVER_PORT}... Allowing requests from origin: {ALLOWED_ORIGIN}")
    app.run(host='0.0.0.0', port=SERVER_PORT, debug=False)