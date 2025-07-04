import requests
from bs4 import BeautifulSoup
import json
import time
import os
import logging
from urllib.parse import urljoin
from newspaper import Article # type: ignore # For better article extraction
from langchain_groq import ChatGroq # Use Groq for consistency
# Removed: from dotenv import load_dotenv # No longer using .env

# --- Configuration ---
# Removed: load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# !!! IMPORTANT: Replace with REAL URLs and CSS selectors after checking permissions !!!
CASE_STUDY_SOURCES = [
    {
        "name": "Example Edu Site", # Replace
        "list_url": "https://www.example-edu.com/finance/case-studies/", # Replace
        "link_selector": "div.case-study-list-item > a", # CSS selector for links on the list page
        "content_selector": "article.main-content", # CSS selector for main text container (Newspaper might override)
        "link_base_url": "https://www.example-edu.com" # Base URL if links are relative
    },
    # { # Add another source if desired and permitted
    #     "name": "Another Example Source",
    #     "list_url": "https://another-site.org/cases",
    #     "link_selector": "h3.case-title a",
    #     "content_selector": "div.entry-content",
    #     "link_base_url": "https://another-site.org"
    # },
]
OUTPUT_FILE = "case_studies.json" # Output file in the same directory as the script
MAX_SUMMARIES_OVERALL = 15 # Total number of summaries to generate
MAX_SUMMARIES_PER_SOURCE = 8 # Max summaries from a single source
REQUEST_DELAY_SECONDS = 3 # Seconds between requests
MAX_CONTENT_LENGTH_FOR_LLM = 4000 # Approx character limit for LLM input

# --- Hardcoded Credentials (!!! SECURITY RISK !!!) ---
HARDCODED_GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE" # Replace! <<<<<<<<<<<<<<<<<<<<<<<
# --- ---

# --- Initialize Groq LLM ---
llm = None # Initialize as None
if not HARDCODED_GROQ_API_KEY or "YOUR_" in HARDCODED_GROQ_API_KEY:
    logging.error("Hardcoded GROQ_API_KEY not found or is placeholder. Cannot initialize LLM.")
else:
    try:
        llm = ChatGroq(
            temperature=0.4,
            groq_api_key=HARDCODED_GROQ_API_KEY, # Use hardcoded key
            model_name="llama3-8b-8192"
        )
        logging.info("Groq LLM initialized for summarization using hardcoded key.")
    except Exception as e:
        logging.error(f"Failed to initialize Groq LLM with hardcoded key: {e}", exc_info=True)
        llm = None # Ensure llm is None on failure

# Exit if LLM failed to initialize
if llm is None:
     logging.error("LLM initialization failed. Exiting script.")
     exit()

# --- Helper Functions ---

def get_page_soup(url):
    """Fetches and parses HTML content from a URL, returns BeautifulSoup object."""
    try:
        headers = {'User-Agent': 'FinEduCaseStudyScraper/1.0 (Educational Use; Respecting robots.txt)'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.exceptions.Timeout:
        logging.error(f"Timeout fetching {url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
         logging.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
         return None

def extract_article_text_with_newspaper(url):
    """Extracts main article text using the newspaper3k library."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        logging.warning(f"Newspaper3k failed for {url}: {e}. Will fallback if possible.")
        return None

def get_llm_summary(text_content):
    """Gets a summary from the initialized LLM."""
    if not text_content or llm is None: # Check if llm is initialized
        return None
    try:
        truncated_content = text_content[:MAX_CONTENT_LENGTH_FOR_LLM]
        if len(text_content) > MAX_CONTENT_LENGTH_FOR_LLM:
             logging.warning("Content truncated before sending to LLM for summarization.")

        prompt = f"""Please provide a concise summary (around 150-200 words) of the following financial case study or article for an educational website. Focus on the key problem/situation, the analysis or actions taken, and the main outcome or lesson learned.

        Text:
        {truncated_content}

        Summary:"""

        response = llm.invoke(prompt)
        summary = response.content.strip()
        return summary
    except Exception as e:
        logging.error(f"Error getting summary from LLM: {e}", exc_info=True)
        return None

# --- Main Processing Logic ---
if __name__ == "__main__":
    logging.info("Starting case study processing...")
    processed_cases = []
    processed_urls = set()

    for source in CASE_STUDY_SOURCES:
        logging.info(f"\n--- Processing Source: {source['name']} ---")
        list_soup = get_page_soup(source['list_url'])
        if not list_soup:
            logging.warning(f"Could not retrieve list page for {source['name']}. Skipping.")
            continue

        link_elements = list_soup.select(source['link_selector'])
        logging.info(f"Found {len(link_elements)} potential link elements.")
        source_summary_count = 0

        for link_element in link_elements:
            if len(processed_cases) >= MAX_SUMMARIES_OVERALL:
                logging.info("Reached overall maximum summaries.")
                break
            if source_summary_count >= MAX_SUMMARIES_PER_SOURCE:
                logging.info(f"Reached maximum summaries for source: {source['name']}.")
                break

            case_url = link_element.get('href')
            if not case_url: logging.debug("Skipping element with no href."); continue

            if not case_url.startswith('http'):
                base = source.get("link_base_url", source['list_url'])
                case_url = urljoin(base, case_url)
                if not case_url.startswith('http'): logging.warning(f"Could not resolve relative URL: {link_element.get('href')}"); continue

            if case_url in processed_urls or not case_url.startswith(('http://', 'https://')):
                 logging.debug(f"Skipping duplicate or non-http URL: {case_url}")
                 continue

            logging.info(f"  Processing URL: {case_url}")
            processed_urls.add(case_url)

            content_text = extract_article_text_with_newspaper(case_url)
            time.sleep(REQUEST_DELAY_SECONDS)

            if not content_text or len(content_text) < 100:
                logging.warning(f"    Insufficient content extracted from {case_url}. Skipping summary.")
                continue

            title = link_element.get_text(separator=' ', strip=True)
            if not title or len(title) < 10: title = generate_title_from_url(case_url) # type: ignore

            logging.info("    Requesting summary from LLM...")
            summary = get_llm_summary(content_text)

            if summary:
                processed_cases.append({
                    "title": title, "link": case_url, "source": source['name'],
                    "summary": summary, "scraped_date": datetime.now().strftime("%Y-%m-%d") }) # type: ignore
                source_summary_count += 1
                logging.info(f"    Successfully processed: {title}")
            else:
                logging.warning(f"    Failed to get summary for: {title}")

        if len(processed_cases) >= MAX_SUMMARIES_OVERALL: break

    logging.info(f"\n--- Saving {len(processed_cases)} summaries to {OUTPUT_FILE} ---")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(processed_cases, f, indent=2, ensure_ascii=False)
        logging.info("Processing complete. Summaries saved.")
    except IOError as e:
        logging.error(f"Error writing summaries to file {OUTPUT_FILE}: {e}")