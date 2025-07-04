import requests
from bs4 import BeautifulSoup
import json
import random
import time
import os # Added for path handling
# from flask import Flask, jsonify
# from flask_cors import CORS # Import CORS

def generate_title_from_url(url):
    """Generates a somewhat readable title from a URL slug."""
    try:
        # Get the last part of the URL path
        last_part = url.split('/')[-1]
        # Remove common extensions like .html, .cms, etc.
        last_part = os.path.splitext(last_part)[0]
        # Split by hyphens or underscores
        words = last_part.replace('-', ' ').replace('_', ' ').split()
        # Remove trailing numbers if they likely represent IDs
        if words and words[-1].isdigit() and len(words[-1]) > 4: # Avoid removing years like 2024
            words.pop()
        # Capitalize and join
        title = ' '.join(word.capitalize() for word in words if word) # Ensure empty strings aren't capitalized
        return title if title else "Untitled Article" # Return "Untitled" if empty
    except Exception as e:
        print(f"⚠️ Error generating title for {url}: {e}")
        return "Untitled Article"

def scrape_zeebusiness():
    """Scrapes latest news from Zee Business."""
    url = "https://www.zeebiz.com/latest-news"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"} # More common user agent
    articles = []
    seen_urls = set()
    print("Attempting to scrape ZeeBusiness...")
    try:
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        print(f"ZeeBusiness status: {response.status_code}")
        soup = BeautifulSoup(response.text, "html.parser")
        # Adjust selector based on current ZeeBusiness structure (inspect element)
        # This selector might need updating if the site changes
        candidates = soup.select("div.section-news-story h3 a")
        if not candidates:
             candidates = soup.select("div.news_listing a[href*='/news/']") # Alternative selector

        print(f"Found {len(candidates)} potential article links on ZeeBusiness.")

        for item in candidates:
            if len(articles) >= random.randint(7, 9): # Limit articles per source
                break

            link = item.get("href")
            if not link: continue

            # Ensure link is absolute
            if link.startswith('/'):
                link = "https://www.zeebiz.com" + link

            if link in seen_urls or not link.startswith("https://www.zeebiz.com"):
                continue

            seen_urls.add(link)
            title = item.text.strip()

            if title and len(title) > 10: # Basic check for valid title
                articles.append({"title": title, "url": link})
            # else:
            #     print(f"Skipping ZeeBusiness item with missing/short title: {link}")

    except requests.exceptions.Timeout:
        print(f"❌ Timeout error fetching ZeeBusiness: {url}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error fetching ZeeBusiness: {e}")
    except Exception as e:
        print(f"❌ Unexpected error scraping ZeeBusiness: {e}")

    print(f"Scraped {len(articles)} articles from ZeeBusiness.")
    return articles

def scrape_moneycontrol():
    """Scrapes latest news from Moneycontrol Business section."""
    url = "https://www.moneycontrol.com/news/business/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    articles = []
    seen_urls = set()
    print("Attempting to scrape Moneycontrol...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        print(f"Moneycontrol status: {response.status_code}")
        soup = BeautifulSoup(response.text, "html.parser")
        # Find the main news list container (inspect element)
        news_list = soup.find('ul', id='cagetory') or soup.find('div', class_='posts-listing') # Adjust selector based on inspection
        if not news_list:
            print("❌ Could not find main news list container on Moneycontrol.")
            return []

        # Selector for links within the list items
        candidates = news_list.select("li .entry-title a, li h2 a") # Common patterns
        print(f"Found {len(candidates)} potential article links on Moneycontrol.")

        for item in candidates:
            if len(articles) >= random.randint(7, 9): # Limit articles per source
                break

            link = item.get("href")
            if not link: continue

            # Filter out non-article links and duplicates
            if link in seen_urls or not link.startswith("https://www.moneycontrol.com/news/"):
                continue

            seen_urls.add(link)

            # Try to get title directly from tag, fallback to generating from URL
            title_raw = item.get('title') or item.text
            title = title_raw.strip() if title_raw else None

            if not title or len(title) < 10: # If title is missing or too short, generate
                 title = generate_title_from_url(link)

            articles.append({"title": title, "url": link})

    except requests.exceptions.Timeout:
        print(f"❌ Timeout error fetching Moneycontrol: {url}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error fetching Moneycontrol: {e}")
    except Exception as e:
        print(f"❌ Unexpected error scraping Moneycontrol: {e}")

    print(f"Scraped {len(articles)} articles from Moneycontrol.")
    return articles

def main():
    """Main function to scrape and save news data."""
    print("🔄 Scraping started...")

    # Scrape from different sources
    zeebiz_articles = scrape_zeebusiness()
    moneycontrol_articles = scrape_moneycontrol()

    # Combine all articles into a single list
    all_articles = zeebiz_articles + moneycontrol_articles

    # Optional: Shuffle the combined list for variety
    random.shuffle(all_articles)

    # Define the output filename
    output_filename = "news.json"

    # Save the combined list to JSON
    try:
        with open(output_filename, "w", encoding='utf-8') as f: # Use utf-8 encoding
            json.dump(all_articles, f, indent=2, ensure_ascii=False) # ensure_ascii=False for non-latin chars
        print(f"✅ Combined news data saved to '{output_filename}' ({len(all_articles)} articles)")
    except IOError as e:
        print(f"❌ Error writing JSON file: {e}")
    except Exception as e:
         print(f"❌ Unexpected error during JSON serialization: {e}")

if __name__ == "__main__":
    main()
    # Removed the while True loop