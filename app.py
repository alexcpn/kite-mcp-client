# Full `app.py` with Caching, Brave Search API, sentiment analysis, and duplicate filtering

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import csv
import yfinance as yf
from openai import OpenAI
import re
from dotenv import load_dotenv
import os
import json
import logging
from datetime import datetime, timedelta

load_dotenv()  # Load variables from .env into the environment
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"
CACHE_FILE = "cache.json"
CACHE_DURATION_HOURS = 4  # How long to keep cache data before refreshing

client = OpenAI(api_key=openai_api_key)

# --- Caching Functions ---
def load_cache():
    """Loads the cache from a JSON file."""
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(cache_data):
    """Saves the cache to a JSON file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=4)

# --- Core Functions ---
def load_holdings(path="holdings.csv"):
    with open(path, newline="", encoding='utf-8') as f:
        return list(csv.DictReader(f))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def expand_query_with_gpt(company_name, sector, description):
    # This function remains the same
    prompt = (
        f"You are a financial news search expert creating queries for the Brave Search API.\n"
        f"Company: \"{company_name}\" (sector: {sector})\n"
        "Task: Generate 3 to 4 diverse, professional-grade search phrases to find relevant news.\n"
        "Guidelines:\n"
        "- Phrase 1: The company's full, official name.\n"
        "- Phrase 2-4: Focus on key business activities, major announcements, partnerships, or market news.\n"
        "Output a valid JSON object with a single key 'phrases' which contains an array of strings."
    )
    try:
        logging.info(f"[GPT] Generating search phrases for {company_name!r}")
        resp = client.chat.completions.create(
            model="gpt-4o-mini", response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}], temperature=0.5,
        )
        result = json.loads(resp.choices[0].message.content)
        phrases = result.get("phrases", [])
        logging.info(f"[GPT] Generated phrases: {phrases}")
        return phrases
    except Exception as e:
        logging.error(f"⚠️ Phrase generation failed for {company_name!r}: {e}")
        return []

def is_article_relevant(article, company_name):
    # This function remains the same
    title = article.get("title", "")
    description = article.get("description", "")
    prompt = (
        f"You are a news classifier for an investor in '{company_name}'.\n\n"
        f"News Article:\nTitle: {title}\nDescription: {description}\n\n"
        "Task: Is this article directly relevant to '{company_name}'s business operations, financial performance, stock, or major partnerships/products? Answer 'YES' or 'NO'."
    )
    try:
        logging.info(f"[GPT] Relevance check for: '{title}'")
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        logging.info(f"[GPT] Relevance answer: {answer}")
        return "YES" in answer
    except Exception as e:
        logging.error(f"⚠️ relevance GPT error on {title!r}: {e}")
    return False

def get_sentiment(article):
    # This function remains the same
    title, description = article.get("title", ""), article.get("description", "")
    prompt = (
        f"You are a financial news sentiment analyst.\nAnalyze the following news article:\n"
        f"Title: {title}\nDescription: {description}\n\n"
        "Task: Rate the sentiment from an investor's perspective. Return ONLY a single float between -1.0 (very negative) and 1.0 (very positive)."
    )
    try:
        logging.info(f"[GPT] Sentiment check for: '{title}'")
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        score = float(resp.choices[0].message.content.strip())
        logging.info(f"[GPT] Sentiment score: {score}")
        return max(-1.0, min(1.0, score))
    except Exception as e:
        logging.error(f"⚠️ sentiment GPT error on {title!r}: {e}")
    return 0.0

@app.get("/api/top-headlines", response_class=JSONResponse)
async def api_top_headlines():
    """API endpoint to fetch only the top headlines."""
    top_headlines = []
    try:
        logging.info("Fetching general business news from Brave API for API endpoint...")
        headers = {"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"}
        params = {"q": "India business finance market", "country": "in", "search_lang": "en", "count": 10}
        
        top_headlines_resp = requests.get(BRAVE_NEWS_ENDPOINT, headers=headers, params=params)
        top_headlines_resp.raise_for_status()
        
        brave_results = top_headlines_resp.json().get("results", [])
        for item in brave_results:
            top_headlines.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "source": {"name": item.get("source")},
                "publishedAt": item.get("page_age")
            })
        logging.info(f"Successfully fetched {len(top_headlines)} general news articles for API.")
    except Exception as e:
        logging.error(f"Error fetching general business news from Brave for API: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    return JSONResponse(content=top_headlines)


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    cache = load_cache()
    holdings = load_holdings()

    holdings_with_news = []
    for h in holdings:
        symbol = h.get("tradingsymbol", "").strip()
        if not symbol: continue
        logging.info(f"Processing {symbol}")

        is_cache_fresh = False
        if symbol in cache:
            cached_at = datetime.fromisoformat(cache[symbol]['timestamp'])
            if datetime.now() - cached_at < timedelta(hours=CACHE_DURATION_HOURS):
                is_cache_fresh = True

        if is_cache_fresh:
            logging.info(f"Using fresh cache for {symbol}")
            articles_with_sentiment = cache[symbol]['articles']
        else:
            logging.info(f"Cache stale or not found for {symbol}. Fetching new data.")
            try:
                info = yf.Ticker(f"{symbol}.NS").info
                name, sector, desc = info.get("longName", symbol), info.get("sector", ""), info.get("longBusinessSummary", "")
                search_phrases = expand_query_with_gpt(name, sector, desc)
                if not search_phrases: search_phrases = [f'"{name}"']

                all_articles = []
                seen_urls = set()
                headers = {"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"}
                for phrase in search_phrases:
                    query = f'{phrase} -site:simplywall.st'
                    params = {"q": query, "count": 5, "freshness": "pd7"}
                    resp = requests.get(BRAVE_NEWS_ENDPOINT, headers=headers, params=params)
                    resp.raise_for_status()
                    for item in resp.json().get("results", []):
                        if item.get('url') and item['url'] not in seen_urls:
                            seen_urls.add(item['url'])
                            all_articles.append({
                                "title": item.get("title"), "description": item.get("description"),
                                "url": item.get("url"), "source": {"name": item.get("source")},
                                "publishedAt": item.get("page_age")
                            })
                
                articles_with_sentiment = []
                for art in all_articles:
                    if is_article_relevant(art, name):
                        art_copy = art.copy()
                        art_copy['sentiment'] = get_sentiment(art)
                        articles_with_sentiment.append(art_copy)
                
                cache[symbol] = {
                    "timestamp": datetime.now().isoformat(),
                    "articles": articles_with_sentiment
                }
            except Exception as e:
                logging.error(f"Failed to fetch new data for {symbol}: {e}")
                articles_with_sentiment = cache.get(symbol, {}).get('articles', [])

        try:
            info = yf.Ticker(f"{symbol}.NS").info
            hist = yf.Ticker(f"{symbol}.NS").history(period="1mo")
            
            last_price = hist["Close"].iloc[-1] if not hist.empty else None
            prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else last_price
            week_ago = hist["Close"].iloc[-5] if len(hist) >= 5 else hist["Close"].iloc[0] if not hist.empty else None
            month_ago = hist["Close"].iloc[0] if not hist.empty else None

            day_change_pct = ((last_price - prev_close) / prev_close * 100) if last_price and prev_close else None
            week_change_pct = ((last_price - week_ago) / week_ago * 100) if last_price and week_ago and week_ago != 0 else None
            month_change_pct = ((last_price - month_ago) / month_ago * 100) if last_price and month_ago and month_ago != 0 else None
            
            pe_ratio = info.get("trailingPE")
            eps = info.get("trailingEps")
            roce = info.get("returnOnEquity")

            holdings_with_news.append({
                "symbol": symbol, "quantity": h.get("quantity"), "avg_price": h.get("average_price"),
                "last_price": last_price, "pe_ratio": pe_ratio, "eps": eps, "roce": roce,
                "day_change_pct": day_change_pct, "week_change_pct": week_change_pct, "month_change_pct": month_change_pct,
                "articles": articles_with_sentiment
            })
        except Exception as e:
            logging.error(f"Failed to process yfinance data for {symbol}: {e}")
            
    save_cache(cache)
    return templates.TemplateResponse("holdings.html", {"request": request, "top_headlines": [], "data": holdings_with_news})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
