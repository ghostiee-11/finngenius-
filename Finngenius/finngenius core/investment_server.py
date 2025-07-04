# investment_server.py
import os
import logging
import asyncio
import time
import re
import math # For FV calculations
import random
from typing import Dict, Optional, List, Any

# --- FastAPI & Related ---
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError

# --- Data Handling & Finance ---
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import httpx

# --- LLM (Groq) ---
try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    AsyncGroq = None
    logging.warning("Groq library not found. LLM features disabled.")

# --- Configuration ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s') # Use DEBUG for detailed logs
logger = logging.getLogger(__name__)

SERVER_PORT = 5002
ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://127.0.0.1:3001"]

# --- Asset Definitions ---
INDEX_SYMBOLS_TO_FETCH = ["^NSEI", "^BSESN", "^NSEBANK"]
CRYPTO_IDS = ["bitcoin", "ethereum"]
EQUITY_TREND_SYMBOL = "^NSEI"
CRYPTO_TREND_SYMBOL = "BTC-USD"
GOLD_TREND_SYMBOL = "GC=F" # Using Futures for trend context

# --- Hardcoded Data ---
HARDCODED_GOLD_PRICE_INR_PER_GRAM = 9567.20 # << UPDATE AS NEEDED
HARDCODED_GOLD_TREND = "Upward" # << UPDATE AS NEEDED
HARDCODED_GROQ_API_KEY = "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" # <<< REPLACE OR USE ENV VAR!

# --- Estimated Annualized Returns (CRUCIAL ASSUMPTIONS) ---
ESTIMATED_RETURNS = {
    "Equity": 0.12,   # 12% p.a.
    "Crypto": 0.18,   # 18% p.a. (High assumption, very volatile)
    "Gold":   0.07,   # 7% p.a. (Illustrative, often lower/hedge focus)
    "FD":     0.06,   # 6% p.a. (Fixed Deposit - Simulated)
    "DebtMF": 0.065,  # 6.5% p.a. (Debt Mutual Fund - Simulated)
}

# --- Initialize Groq Client ---
async_groq_client = None
if GROQ_AVAILABLE:
    if HARDCODED_GROQ_API_KEY and "YOUR_GROQ_API_KEY_HERE" not in HARDCODED_GROQ_API_KEY and "gsk_" in HARDCODED_GROQ_API_KEY:
        try: async_groq_client = AsyncGroq(api_key=HARDCODED_GROQ_API_KEY); logger.info("AsyncGroq client initialized.")
        except Exception as e: logger.error(f"Failed AsyncGroq init: {e}"); GROQ_AVAILABLE = False; async_groq_client = None
    else: logger.warning("Groq key placeholder/invalid. LLM disabled."); GROQ_AVAILABLE = False

# --- Global Cache ---
market_context_cache = {"timestamp": 0, "data": None}
CACHE_DURATION_SECONDS = 300

# --- Pydantic Models ---

class GoalInfoRequest(BaseModel):
    """ Input for goal-based recommendation. """
    risk_profile: str = Field(..., pattern="^(Low|Medium|High)$", description="User's risk tolerance (Low, Medium, High)")
    goal_duration_years: int = Field(..., gt=0, description="Investment duration for the goal in years")
    goal_name: str = Field(..., min_length=1, description="Name of the investment goal")
    goal_target_amount: float = Field(..., gt=0, description="Target amount user wants to achieve for the goal")
    preferred_frequency: str = Field("monthly", pattern="^(monthly|quarterly|yearly)$", description="User's preferred investment cadence (for display focus)") # Default to monthly

    # **** CORRECTED INDENTATION and FORMATTING ****
    @field_validator('goal_name')
    @classmethod # Good practice for validators
    def goal_name_strip_and_validate(cls, v: str) -> str:
        """ Ensure goal name is not empty or just whitespace. """
        if not isinstance(v, str):
             # Added type check for robustness, though Pydantic usually handles this
             raise ValueError('Goal name must be a string.')
        # 1. Strip whitespace
        name = v.strip()
        # 2. Check if empty after stripping
        if not name:
            # 3. Raise error if empty
            raise ValueError('Goal name cannot be empty or consist only of whitespace.')
        # 4. Return the cleaned name
        return name
    # **** END CORRECTION ****

class AssetData(BaseModel):
    # ... rest of the AssetData model ...
    symbol: str
    name: str
    current_price: Optional[float]=None
    currency: Optional[str]=None

class TrendAnalysis(BaseModel):
    # ... rest of the TrendAnalysis model ...
    rsi: Optional[float]=None
    macd_signal: Optional[str]=None
    volatility_atr: Optional[float]=None
    trend: Optional[str]=Field("N/A", pattern="^(Upward|Downward|Sideways|N/A)$")

class MarketContextResponse(BaseModel):
    # ... rest of the MarketContextResponse model ...
    assets: Dict[str, Dict[str, AssetData]]
    analysis: Dict[str, TrendAnalysis]

class InvestmentBreakdownPeriod(BaseModel):
    # ... rest of the InvestmentBreakdownPeriod model ...
    period: str
    required_total_investment: float = Field(...)
    breakdown: Dict[str, float]

class GoalRecommendationResponse(BaseModel):
    # ... rest of the GoalRecommendationResponse model ...
    goal_name: str
    goal_target_amount: float
    goal_duration_years: int
    estimated_portfolio_return: float = Field(...)
    allocation: Dict[str, float]
    required_investment_periods: List[InvestmentBreakdownPeriod]
    explanation: str


# --- Helper Functions (Data Fetching, Indicators - largely unchanged) ---
async def run_sync_in_thread(func, *args, **kwargs): loop = asyncio.get_running_loop(); return await loop.run_in_executor(None, func, *args, **kwargs)
def fetch_yfinance_sync(symbols: List[str]) -> Dict[str, AssetData]:
    data = {}; logger.info(f"[Sync] Fetching yfinance Index/Futures data for: {symbols}")
    if not symbols: return {}
    try:
        tickers = yf.Tickers(" ".join(symbols))
        for symbol in symbols:
            asset_data = AssetData(symbol=symbol, name=symbol, current_price=None, currency=None)
            try:
                ticker_obj = tickers.tickers.get(symbol);
                if ticker_obj is None: logger.warning(f"[Sync] Ticker object not found for '{symbol}'."); data[symbol] = asset_data; continue
                info = ticker_obj.info; price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                asset_data.name = info.get('shortName', symbol)
                # Refined currency logic
                is_indian_index = symbol.startswith(("^NS", "^BS"))
                is_gold_future = symbol == GOLD_TREND_SYMBOL # GC=F is typically USD
                asset_data.currency = 'INR' if is_indian_index else 'USD' if is_gold_future else info.get('currency', 'USD') # Default USD otherwise

                if price is None or not isinstance(price, (int, float)) or price <= 0:
                    hist = ticker_obj.history(period='2d', interval='1d');
                    if not hist.empty: price = hist['Close'].iloc[-1]; logger.debug(f"Using historical close for {symbol}: {price}")
                    else: price = None
                if price is not None and isinstance(price, (int, float)) and price > 0: asset_data.current_price = round(float(price), 2)
                else: logger.warning(f"[Sync] Could not retrieve valid > 0 price for {symbol}.")
            except Exception as e: error_str = str(e).lower(); logger.warning(f"[Sync] Error processing {symbol}: {e}", exc_info=not any(s in error_str for s in ["no data", "404", "decrypt"]))
            data[symbol] = asset_data
    except Exception as e: logger.error(f"[Sync] Major yfinance batch fetch error: {e}", exc_info=True)
    logger.info(f"[Sync] Completed yfinance fetch. Processed: {len(data)} symbols.")
    return data

async def fetch_coingecko_async(ids: List[str]) -> Dict[str, AssetData]:
    data = {};
    if not ids: return {}
    ids_str = ",".join(ids); url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_str}&vs_currencies=inr"; logger.info(f"Fetching CoinGecko: {ids} vs INR")
    try:
        await asyncio.sleep(random.uniform(0.1, 0.4))
        async with httpx.AsyncClient(timeout=15.0) as client: response = await client.get(url); response.raise_for_status(); api_data = response.json()
        for crypto_id, price_data in api_data.items():
            price_inr = price_data.get('inr'); symbol_upper = crypto_id.upper()
            data[symbol_upper] = AssetData(symbol=symbol_upper, name=crypto_id.capitalize(), current_price=round(float(price_inr), 2) if price_inr is not None and isinstance(price_inr, (int, float)) else None, currency='INR')
        logger.info(f"Successfully fetched CoinGecko: {list(data.keys())}")
    except httpx.HTTPStatusError as e: logger.warning(f"CoinGecko HTTP Status error: {e.response.status_code}") if e.response.status_code == 429 else logger.error(f"CoinGecko HTTP Status error: {e.response.status_code}", exc_info=False)
    except httpx.RequestError as e: logger.error(f"CoinGecko network error: {e}", exc_info=True)
    except Exception as e: logger.error(f"CoinGecko processing error: {e}", exc_info=True)
    return data

async def fetch_market_prices() -> Dict[str, Dict[str, AssetData]]:
    logger.info("Starting market price fetching...")
    indices_task = asyncio.create_task(run_sync_in_thread(fetch_yfinance_sync, INDEX_SYMBOLS_TO_FETCH))
    crypto_task = asyncio.create_task(fetch_coingecko_async(CRYPTO_IDS))
    results = await asyncio.gather(indices_task, crypto_task, return_exceptions=True)
    indices_data = results[0] if not isinstance(results[0], Exception) else {}; crypto_data = results[1] if not isinstance(results[1], Exception) else {}
    if isinstance(results[0], Exception): logger.error(f"Indices price fetch failed: {results[0]}", exc_info=False)
    if isinstance(results[1], Exception): logger.error(f"Crypto price fetch failed: {results[1]}", exc_info=False)
    logger.info(f"Adding hardcoded Gold reference price: ₹{HARDCODED_GOLD_PRICE_INR_PER_GRAM}/gram")
    gold_data_processed = {"GOLD": AssetData(symbol="GOLD", name="Gold (INR/gram, Ref.)", current_price=HARDCODED_GOLD_PRICE_INR_PER_GRAM, currency="INR")}
    # **** ADDED FD Placeholder ****
    logger.info("Adding placeholder FD reference.")
    fd_data_processed = {"FD": AssetData(symbol="FD", name="Fixed Deposit (Simulated)", current_price=None, currency="INR")} # No 'price' for FD
    price_data = {"Indices": indices_data or {}, "Crypto": crypto_data or {}, "Gold": gold_data_processed, "FD": fd_data_processed}
    logger.info("Market price construction complete.")
    return price_data

def get_historical_data_sync(symbol: str, period: str = "6mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    logger.info(f"[Sync] Attempting historical data fetch for {symbol} ({period}, {interval})")
    try:
        ticker = yf.Ticker(symbol); hist = ticker.history(period=period, interval=interval, progress=False)
        if hist.empty: logger.warning(f"[Sync] No historical data returned for {symbol}."); return None
        logger.debug(f"[Sync] Fetched {len(hist)} rows for {symbol}. Head:\n{hist.head()}")
        hist.columns = [col.capitalize() for col in hist.columns]; required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in hist.columns for col in required_cols): logger.warning(f"[Sync] Historical data for {symbol} missing OHLC: {hist.columns}"); return None
        if hist.index.tz is not None: hist.index = hist.index.tz_localize(None)
        min_required_rows = 30
        if len(hist) < min_required_rows: logger.warning(f"[Sync] Insufficient history ({len(hist)} rows) for {symbol}. Needed ~{min_required_rows}+."); return None
        logger.info(f"[Sync] Historical data VALIDATED for {symbol} ({len(hist)} rows)")
        return hist
    except Exception as e: logger.error(f"[Sync] Error fetching historical for {symbol}: {e}", exc_info=True); return None

def calculate_indicators(df: pd.DataFrame, symbol_name: str = "Unknown") -> TrendAnalysis:
    analysis = TrendAnalysis(); logger.debug(f"Calculating indicators for {symbol_name}...")
    if df is None or df.empty or 'Close' not in df.columns: logger.warning(f"Indicators ({symbol_name}): Input invalid."); return analysis
    min_rows_macd = 26 + 9;
    if len(df) < min_rows_macd: logger.warning(f"Indicators ({symbol_name}): Insufficient data ({len(df)} rows) for MACD.")
    elif len(df) < 15: logger.warning(f"Indicators ({symbol_name}): Insufficient data ({len(df)} rows) for RSI.")
    try:
        df.ta.rsi(append=True); df.ta.macd(append=True); df.ta.atr(append=True)
        if 'RSI_14' not in df.columns: logger.warning(f"RSI_14 missing ({symbol_name})")
        if 'MACD_12_26_9' not in df.columns: logger.warning(f"MACD_12_26_9 missing ({symbol_name})")
        if 'ATRr_14' not in df.columns: logger.warning(f"ATRr_14 missing ({symbol_name})")
        last_row = df.iloc[-1]
        rsi_val = last_row.get('RSI_14');
        if pd.notna(rsi_val):
            analysis.rsi = round(rsi_val, 2)
            if analysis.rsi > 70: analysis.trend = "Upward"; # Strong Upward
            elif analysis.rsi > 55: analysis.trend = "Upward"; # Moderate Upward
            elif analysis.rsi < 30: analysis.trend = "Downward"; # Strong Downward
            elif analysis.rsi < 45: analysis.trend = "Downward"; # Moderate Downward
            else: analysis.trend = "Sideways"
            logger.debug(f"RSI ({symbol_name}): {analysis.rsi:.2f} -> Trend: {analysis.trend}")
        else: logger.warning(f"RSI NaN for {symbol_name}. Trend N/A.") # Keep default N/A
        macd_line = last_row.get('MACD_12_26_9'); signal_line = last_row.get('MACDs_12_26_9')
        if pd.notna(macd_line) and pd.notna(signal_line) and len(df) >= 2:
            prev_row = df.iloc[-2]; prev_macd = prev_row.get('MACD_12_26_9'); prev_signal = prev_row.get('MACDs_12_26_9')
            if pd.notna(prev_macd) and pd.notna(prev_signal): analysis.macd_signal = "Bullish Cross" if macd_line > signal_line and prev_macd <= prev_signal else "Bearish Cross" if macd_line < signal_line and prev_macd >= prev_signal else "Neutral"; logger.debug(f"MACD Signal ({symbol_name}): {analysis.macd_signal}")
            else: analysis.macd_signal = "Neutral"; logger.debug(f"MACD ({symbol_name}): Prev NaN.")
        else: analysis.macd_signal = "N/A"; logger.debug(f"MACD ({symbol_name}): Curr/Prev NaN or < 2 rows.")
        atr_val = last_row.get('ATRr_14');
        if pd.notna(atr_val): analysis.volatility_atr = round(atr_val, 4); logger.debug(f"ATR ({symbol_name}): {analysis.volatility_atr:.4f}")
        else: logger.debug(f"ATR NaN for {symbol_name}.")
    except Exception as e: logger.error(f"Indicator calculation error ({symbol_name}): {e}", exc_info=True); analysis = TrendAnalysis() # Reset on error
    finally: logger.info(f"Indicator Result ({symbol_name}): {analysis.model_dump()}")
    return analysis

async def analyze_market_trends() -> Dict[str, TrendAnalysis]:
    """Analyzes trends for Equity, Crypto, adds hardcoded Gold trend, placeholder for FD."""
    all_analysis = {}; logger.info("Starting trend analysis (Equity, Crypto + Hardcoded Gold + Placeholder FD)...")
    # Fetch historical for Equity & Crypto
    analysis_symbols = {"Equity": EQUITY_TREND_SYMBOL, "Crypto": CRYPTO_TREND_SYMBOL}
    tasks = []; categories = []
    for category, symbol in analysis_symbols.items(): categories.append(category); tasks.append(asyncio.create_task(run_sync_in_thread(get_historical_data_sync, symbol)))
    historical_data_results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, category in enumerate(categories):
        symbol = analysis_symbols[category]; hist_df = historical_data_results[i]
        if isinstance(hist_df, Exception) or hist_df is None: logger.error(f"Failed HISTORICAL for {category} trend ({symbol}). Error: {hist_df}", exc_info=False); all_analysis[category] = TrendAnalysis()
        else: logger.info(f"Calculating indicators for {category} trend ({symbol})"); all_analysis[category] = calculate_indicators(hist_df, symbol_name=symbol);
        if all_analysis[category].trend == "N/A": logger.warning(f"Indicator calc for {category} ({symbol}) resulted in Trend: N/A.")

    # Add hardcoded Gold trend context
    logger.info(f"Adding hardcoded Gold trend context: {HARDCODED_GOLD_TREND}"); all_analysis["Gold"] = TrendAnalysis(trend=HARDCODED_GOLD_TREND)
    # **** ADDED FD Placeholder Trend ****
    logger.info("Adding placeholder FD trend: N/A (Not Applicable)")
    all_analysis["FD"] = TrendAnalysis(trend="N/A") # FD doesn't have a 'trend' in this context

    logger.info(f"Market trend analysis complete. Final Analysis: {all_analysis}")
    return all_analysis

# --- Allocation & Goal Calculations ---

# **** UPDATED **** Allocation Logic for 5 Asset Classes
def calculate_portfolio_allocation(
    risk_profile: str,
    goal_duration: int,
    market_analysis: Dict[str, TrendAnalysis]
) -> Dict[str, float]:
    """ Calculates allocation percentages across Equity, Crypto, Gold, FD, and DebtMF. """
    logger.info(f"Calculating 5-asset allocation: Risk={risk_profile}, Duration={goal_duration} years")
    base_allocation = {}
    # Define the 5 asset classes
    asset_classes = ["Equity", "Crypto", "Gold", "FD", "DebtMF"]

    # --- Step 1: Base Allocation based on Risk Profile ---
    # ** These percentages are ILLUSTRATIVE and need careful tuning **
    if risk_profile == 'Low':
        # Conservative: High safety (DebtMF, FD), some Gold hedge, minimal Equity
        base_allocation = {"Equity": 0.15, "Crypto": 0.00, "Gold": 0.15, "FD": 0.30, "DebtMF": 0.40}
    elif risk_profile == 'Medium':
        # Balanced: Good Equity exposure, some Crypto/Gold, moderate safety
        base_allocation = {"Equity": 0.45, "Crypto": 0.05, "Gold": 0.10, "FD": 0.15, "DebtMF": 0.25}
    else: # High risk
        # Aggressive: Max Equity/Crypto, minimal safety/Gold
        base_allocation = {"Equity": 0.65, "Crypto": 0.10, "Gold": 0.05, "FD": 0.05, "DebtMF": 0.15}
    logger.debug(f" Portfolio Alloc - Step 1 (Risk): {base_allocation}")

    # --- Step 2: Adjust based on Goal Duration ---
    if goal_duration <= 3: # Short-term: Prioritize safety & liquidity
        logger.debug(" Portfolio Alloc - Adjusting Short Duration (<= 3 yrs)")
        base_allocation["Equity"] *= 0.3 # Heavily reduce equity
        base_allocation["Crypto"] = 0.0    # Eliminate crypto
        # Increase safer components (FD, DebtMF slightly more than Gold)
        base_allocation["FD"] = min(0.50, base_allocation.get("FD", 0) + 0.15)
        base_allocation["DebtMF"] = min(0.60, base_allocation.get("DebtMF", 0) + 0.10)
        base_allocation["Gold"] = min(0.25, base_allocation.get("Gold", 0) + 0.05)
    elif goal_duration >= 10: # Long-term: Favor growth
        logger.debug(" Portfolio Alloc - Adjusting Long Duration (>= 10 yrs)")
        # Reduce safer fixed income significantly
        base_allocation["FD"] *= 0.2
        base_allocation["DebtMF"] *= 0.4
        # Increase growth if risk allows
        if risk_profile != 'Low': base_allocation["Equity"] = min(0.85, base_allocation.get("Equity", 0) + 0.10)
        if risk_profile == 'High': base_allocation["Crypto"] = min(0.20, base_allocation.get("Crypto", 0) + 0.05)
        # Maybe slightly reduce Gold for very long term if high risk
        if risk_profile == 'High': base_allocation["Gold"] *= 0.7

    logger.debug(f" Portfolio Alloc - Step 2 (Duration Adj): {base_allocation}")

    # --- Step 3: Adjust based on Market Trends (Equity, Crypto, Gold) ---
    logger.debug(f" Portfolio Alloc - Step 3 (Trend Adj). Analysis: {market_analysis}")
    adjustment_factor = 0.05 # Tactical shift percentage

    for category in ["Equity", "Crypto", "Gold"]: # Adjust these 3 based on trends
        analysis = market_analysis.get(category)
        if not analysis or category not in base_allocation or base_allocation.get(category, 0) < 0.01: continue
        trend = analysis.trend; logger.debug(f"Considering trend '{trend}' for '{category}'")

        # Downward Trend: Reduce allocation, shift primarily to DebtMF, slightly to FD
        if trend == "Downward":
            if base_allocation[category] > adjustment_factor:
                reduced_amount = min(base_allocation[category], adjustment_factor); base_allocation[category] -= reduced_amount;
                # Split the reduced amount between DebtMF and FD (e.g., 70/30 split)
                base_allocation["DebtMF"] = base_allocation.get("DebtMF", 0) + (reduced_amount * 0.7)
                base_allocation["FD"] = base_allocation.get("FD", 0) + (reduced_amount * 0.3)
                logger.info(f" Trend Adj: Reducing '{category}' by {reduced_amount*100:.0f}% (Shifted to Debt/FD)")
            else: logger.debug(f" Skipping downward trend adj for {category} (low).")

        # Upward Trend: Increase allocation if risk/cap allows, fund from DebtMF/FD
        elif trend == "Upward":
             increase_allowed = True # Check conditions
             if category == "Crypto" and risk_profile == "Low": increase_allowed = False
             if category == "Equity" and base_allocation.get("Equity", 0) >= 0.90: increase_allowed = False # Cap equity
             if category == "Crypto" and base_allocation.get("Crypto", 0) >= 0.25: increase_allowed = False # Cap crypto
             if category == "Gold" and base_allocation.get("Gold", 0) >= 0.30: increase_allowed = False # Cap gold

             # Fund from DebtMF first, then FD if needed
             if increase_allowed:
                 funded = False
                 if base_allocation.get("DebtMF", 0) >= adjustment_factor:
                     base_allocation[category] += adjustment_factor; base_allocation["DebtMF"] -= adjustment_factor; funded = True; logger.info(f" Trend Adj: Increasing '{category}' by {adjustment_factor*100:.0f}% (from DebtMF).")
                 elif base_allocation.get("FD", 0) >= adjustment_factor:
                     base_allocation[category] += adjustment_factor; base_allocation["FD"] -= adjustment_factor; funded = True; logger.info(f" Trend Adj: Increasing '{category}' by {adjustment_factor*100:.0f}% (from FD).")

                 if not funded: logger.debug(f" Skipping upward trend adj for {category} (cannot fund).")
             else: logger.debug(f" Skipping upward trend adj for {category} (risk/cap limit).")

    logger.debug(f" Portfolio Alloc - Step 3 (Trend Adj Complete): {base_allocation}")

    # --- Step 4: Normalization ---
    for asset in asset_classes: base_allocation.setdefault(asset, 0.0) # Ensure all keys
    total = sum(base_allocation.values()); final_allocation = {}
    if total <= 0: logger.warning("Total alloc zero, default 100% DebtMF."); return {"DebtMF": 1.0}
    for asset in asset_classes: final_allocation[asset] = round(max(0, base_allocation[asset] / total), 3)
    diff = 1.0 - sum(final_allocation.values())
    if abs(diff) > 1e-9:
        # Prioritize adjusting DebtMF or FD, then largest component
        adjust_target = None
        if final_allocation.get("DebtMF", 0) > 0.01: adjust_target = "DebtMF"
        elif final_allocation.get("FD", 0) > 0.01: adjust_target = "FD"
        else: adjust_target = max(final_allocation, key=lambda k: final_allocation.get(k,0)) if any(v>0 for v in final_allocation.values()) else "DebtMF" # Fallback
        if adjust_target not in final_allocation: final_allocation[adjust_target] = 0.0 # Ensure key exists
        final_allocation[adjust_target] = round(final_allocation[adjust_target] + diff, 3); final_allocation[adjust_target] = max(0, final_allocation[adjust_target])
        logger.debug(f" Applied norm diff {diff:.5f} to '{adjust_target}'")
        if abs(1.0 - sum(final_allocation.values())) > 1e-9: logger.warning("Normalization failed!")
    final_allocation = {k: v for k, v in final_allocation.items() if v > 0.0001} # Filter near-zero
    logger.info(f"Final Portfolio Allocation Calculated (5 Assets): {final_allocation}")
    return final_allocation

# --- (calculate_required_periodic_investment unchanged) ---
def calculate_required_periodic_investment(future_value: float, annual_rate: float, years: int, periods_per_year: int) -> Optional[float]:
    """ Calculates the periodic investment needed using FV of annuity formula. """
    if annual_rate <= -1.0: logger.error("Annual rate cannot be -100% or less."); return None
    if future_value <= 0 or years <= 0 or periods_per_year <= 0: logger.warning("Invalid inputs for FV calculation."); return None
    try:
        periodic_rate = annual_rate / periods_per_year; total_periods = years * periods_per_year
        if abs(periodic_rate) < 1e-9: logger.info(f"Rate near zero. Using simple division."); required_investment = future_value / total_periods
        else:
            denominator = math.pow(1 + periodic_rate, total_periods) - 1
            if abs(denominator) < 1e-9: logger.warning("FV Annuity denominator near zero."); return None
            required_investment = future_value * periodic_rate / denominator
        logger.info(f"Calculated required investment (FV={future_value:.2f}, R={annual_rate:.3f}, Y={years}, F={periods_per_year}): {required_investment:.2f}")
        if required_investment <= 0: logger.warning(f"Required investment non-positive ({required_investment:.2f}).")
        return required_investment
    except OverflowError: logger.error("OverflowError in FV calc."); return None
    except Exception as e: logger.error(f"Error calculating required investment: {e}", exc_info=True); return None

# --- (calculate_investment_breakdown unchanged - works with any allocation dict) ---
def calculate_investment_breakdown(required_periodic_investment: float, allocation: Dict[str, float]) -> Dict[str, float]:
    """ Calculates the amount for each asset class for a single period's REQUIRED investment. """
    period_breakdown_dict: Dict[str, float] = {}; calculated_sum = 0.0
    if required_periodic_investment <= 0:
        logger.warning(f"Required investment {required_periodic_investment:.2f} <= 0. Returning zero breakdown.")
        for asset in allocation: period_breakdown_dict[asset] = 0.0
        return period_breakdown_dict
    for asset, percentage in allocation.items():
        if not isinstance(percentage, (int, float)) or percentage < 0: logger.warning(f"Invalid percentage {percentage} for {asset}."); continue
        amount = round(required_periodic_investment * percentage, 2); period_breakdown_dict[asset] = amount; calculated_sum += amount
    diff = round(required_periodic_investment - calculated_sum, 2)
    if abs(diff) > 0.001 and period_breakdown_dict:
         adjust_target = max(period_breakdown_dict, key=lambda k: period_breakdown_dict.get(k, 0))
         period_breakdown_dict[adjust_target] = round(period_breakdown_dict[adjust_target] + diff, 2)
         period_breakdown_dict[adjust_target] = max(0, period_breakdown_dict[adjust_target])
         logger.debug(f"Adjusted breakdown rounding diff: {diff:.2f} on '{adjust_target}'")
    return period_breakdown_dict

# --- Explanation Generation (Updated Prompts) ---
async def generate_llm_explanation_combined(allocation: Dict[str, float], goal_info: GoalInfoRequest, analysis: Dict[str, TrendAnalysis], required_investments: Dict[str, Optional[float]], estimated_portfolio_return: float) -> str:
    """ Generates combined explanation using Groq for the 5-asset allocation. """
    global async_groq_client, GROQ_AVAILABLE
    if not GROQ_AVAILABLE or async_groq_client is None: logger.warning("Groq N/A. Fallback explanation."); return generate_simple_explanation_fallback(allocation, goal_info, analysis, required_investments, estimated_portfolio_return)
    logger.info("Generating combined explanation using Groq LLM (5-Asset Goal-Based)...")
    # --- Context Preparation ---
    goal_context = f"**User Goal & Profile:** Goal={goal_info.goal_name}, Target=₹{goal_info.goal_target_amount:,.2f}, Duration={goal_info.goal_duration_years} yrs, Risk={goal_info.risk_profile}\n\n"
    market_context = f"**Market Analysis:** Equity Trend={analysis.get('Equity', TrendAnalysis()).trend or 'N/A'}, Crypto Trend={analysis.get('Crypto', TrendAnalysis()).trend or 'N/A'}, Gold Trend Context={analysis.get('Gold', TrendAnalysis()).trend or 'N/A'} (FD trend N/A)\n\n"
    allocation_context = "**Recommended Allocation (Equity/Crypto/Gold/FD/Debt):**\n" + "\n".join([f"- {asset}: {(percentage * 100):.0f}%" for asset, percentage in allocation.items()]) + "\n\n"
    investment_context = f"**Calculations:** Est. Portfolio Return={(estimated_portfolio_return * 100):.1f}% p.a.\n"
    req_monthly = required_investments.get('monthly'); req_quarterly = required_investments.get('quarterly'); req_yearly = required_investments.get('yearly')
    investment_context += f"- Req. Monthly Inv: {'₹{:,.2f}'.format(req_monthly) if req_monthly else 'Error'}\n"
    investment_context += f"- Req. Quarterly Inv: {'₹{:,.2f}'.format(req_quarterly) if req_quarterly else 'Error'}\n"
    investment_context += f"- Req. Yearly Inv: {'₹{:,.2f}'.format(req_yearly) if req_yearly else 'Error'}\n"
    full_context = goal_context + market_context + allocation_context + investment_context
    # --- Prompts ---
    # **** UPDATED SYSTEM PROMPT for 5 ASSETS ****
    alloc_system_prompt = """You are 'FinPlan Explainer'. Explain the reasoning behind the recommended **5-asset investment allocation** (Equity, Crypto, Gold, FD, DebtMF) and the **required investment amounts**.

**Instructions:**
1.  **Use ONLY Provided Context:** Base explanation STRICTLY on Goal, Profile, Market Analysis (Equity/Crypto/Gold trends), Allocation percentages, Estimated Return, and Required Investments.
2.  **Explain Allocation 'Why':** Explain *why* this specific mix (Equity for growth, Crypto for high-risk growth, Gold for hedge/stability, FD & DebtMF for safety/stability) is suitable for the user's **Risk Profile** and **Goal Duration**. Mention how **Equity/Crypto/Gold Trends** influenced their specific percentages.
3.  **Explain Investment Amount 'Why':** Explain *why* the **Required Periodic Investment** amounts are calculated. Link it to the **Goal Target**, **Duration**, and **Estimated Portfolio Return**. Note the relationship between risk, return estimates, and required savings.
4.  **Structure:** Two clear paragraphs: 1) Allocation Rationale (all 5 assets), 2) Required Investment Explanation.
5.  **Clarity & Conciseness:** Be clear and reasonably brief.
6.  **No External Info/Advice:** Stick to the context. No buy/sell advice.
"""
    alloc_user_prompt = f"Based *only* on the context below, explain the rationale for the recommended 5-asset allocation AND the required investment amounts:\n\n{full_context}\n\n**Explanation:**"
    # Gold Outlook Prompt remains the same (general outlook, not tied to specific allocation %)
    gold_system_prompt = f"You are 'FinCommentator'. Provide brief, general outlook on Gold investment in India (mid-2024), acknowledging Gold trend context ('{analysis.get('Gold', TrendAnalysis()).trend or 'N/A'}'). Mention pros (hedge, safety, cultural) & cons (returns vs equity). Balanced view (1 short paragraph). No specific advice."
    gold_user_prompt = "Provide brief, general outlook on Gold investment in India."
    try:
        logger.info("Sending requests to Groq LLM (5-Asset)...");
        alloc_task = async_groq_client.chat.completions.create(messages=[{"role": "system", "content": alloc_system_prompt},{"role": "user", "content": alloc_user_prompt}], model="llama3-8b-8192", temperature=0.6, max_tokens=400) # Increased tokens slightly
        gold_task = async_groq_client.chat.completions.create(messages=[{"role": "system", "content": gold_system_prompt},{"role": "user", "content": gold_user_prompt}], model="llama3-8b-8192", temperature=0.7, max_tokens=150)
        alloc_response, gold_response = await asyncio.gather(alloc_task, gold_task, return_exceptions=True); logger.info("Received responses from Groq LLM.")
        # Process Responses
        combined_explanation = "### Allocation & Investment Rationale (AI Generated):\n"
        if isinstance(alloc_response, Exception) or not alloc_response.choices: logger.error(f"Groq alloc explanation failed: {alloc_response}"); combined_explanation += generate_simple_explanation_fallback(allocation, goal_info, analysis, required_investments, estimated_portfolio_return).split("Disclaimer:")[0].strip() + "\n*(LLM rationale failed.)*"
        else: combined_explanation += alloc_response.choices[0].message.content.strip()
        combined_explanation += "\n\n### General Gold Outlook (AI Generated):\n"
        if isinstance(gold_response, Exception) or not gold_response.choices: logger.error(f"Groq gold outlook failed: {gold_response}"); combined_explanation += "(Could not retrieve gold outlook commentary.)"
        else: combined_explanation += gold_response.choices[0].message.content.strip()
        combined_explanation += "\n\n***\n**Disclaimer:** AI-generated content & calculations use estimates (esp. returns) & are informational only. **NOT financial advice.** Markets vary. Consult a qualified advisor. Tracking feature simulated."
        logger.info("Groq combined explanation assembled.")
        return combined_explanation
    except Exception as e: logger.error(f"Unexpected Groq explanation error: {e}", exc_info=True); logger.warning("Groq failed. Fallback."); return generate_simple_explanation_fallback(allocation, goal_info, analysis, required_investments, estimated_portfolio_return)

# **** UPDATED **** Fallback Explanation for 5 Assets
def generate_simple_explanation_fallback(allocation: Dict[str, float], goal_info: GoalInfoRequest, analysis: Dict[str, TrendAnalysis], required_investments: Dict[str, Optional[float]], estimated_portfolio_return: float) -> str:
    logger.info("Generating simple rule-based fallback explanation (5-Asset Goal-Based).")
    explanation = f"### Investment Rationale (Goal: {goal_info.goal_name}, Risk: {goal_info.risk_profile}, Duration: {goal_info.goal_duration_years} years):\n\n**Allocation:**\nThis mix balances growth potential with safety based on your profile:\n"
    for asset, percentage in sorted(allocation.items(), key=lambda item: item[1], reverse=True):
        percent_str = f"({(percentage * 100):.0f}%)"; reason = ""; trend_info = analysis.get(asset); trend_str = f"(Trend: {trend_info.trend})" if trend_info and trend_info.trend and trend_info.trend != "N/A" else ""
        if asset == "Equity": reason = "Aims for long-term growth."
        elif asset == "Crypto": reason = "High-risk/potential asset."
        elif asset == "Gold": reason = "Acts as a hedge and provides stability."
        elif asset == "FD": reason = "Offers capital safety and predictable returns."
        elif asset == "DebtMF": reason = "Provides stability and liquidity."
        if asset in ["Equity", "Crypto", "Gold"] and trend_str: reason += f" Market trend {trend_str} considered."
        explanation += f"- **{asset} {percent_str}:** {reason}\n"

    explanation += "\n**Required Investment:**\n"
    explanation += f"To reach ₹{goal_info.goal_target_amount:,.2f} in {goal_info.goal_duration_years} yrs (est. return {(estimated_portfolio_return * 100):.1f}% p.a.), suggested investments:\n"
    for period, amount in required_investments.items(): amount_str = f"₹{amount:,.2f}" if amount is not None else "Error"; explanation += f"- {period.capitalize()}: {amount_str}\n"
    explanation += "(Actual returns vary.)\n"
    explanation += f"\n**Analysis Time:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n**Disclaimer:** Rule-based suggestion. **NOT financial advice.** Consult advisor. Tracking simulated."
    return explanation

# --- FastAPI App Setup ---
app = FastAPI(title="Smart Investment Advisor API", version="0.9.0") # 5-Asset version
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API Endpoints ---
@app.get("/", summary="Health Check & API Info", tags=["General"])
async def read_root():
    # (No changes needed here)
    logger.info("Root endpoint '/' accessed."); llm_status = "Available" if GROQ_AVAILABLE and async_groq_client else "Disabled";
    if llm_status == "Available" and "YOUR_GROQ_API_KEY_HERE" in HARDCODED_GROQ_API_KEY: llm_status += " (Placeholder Key!)"
    elif llm_status == "Available": llm_status += " (Using Hardcoded Key - Dev Only!)"
    return {"message": "Investment Advisor API running!", "version": app.version, "llm_status": llm_status, "indices_fetched": INDEX_SYMBOLS_TO_FETCH, "crypto_fetched": CRYPTO_IDS, "hardcoded_gold_price_inr_per_gram": HARDCODED_GOLD_PRICE_INR_PER_GRAM, "hardcoded_gold_trend": HARDCODED_GOLD_TREND, "allowed_origins": ALLOWED_ORIGINS, "estimated_returns_used": ESTIMATED_RETURNS}

# Dependency function to get market context (unchanged)
async def get_current_market_context() -> MarketContextResponse:
    # (Same robust caching logic as before)
    logger.debug("Dependency: get_current_market_context called")
    now = time.time(); global market_context_cache
    if market_context_cache["data"] and (now - market_context_cache["timestamp"] < CACHE_DURATION_SECONDS):
        logger.info("Dependency: Returning cached market context.")
        try: return MarketContextResponse(**market_context_cache["data"])
        except Exception as e: logger.warning(f"Dependency: Cache invalid ({e}). Fetching fresh...")
    logger.info("Dependency: Cache miss/expired. Fetching fresh market context...")
    try:
        market_prices_task = asyncio.create_task(fetch_market_prices())
        market_analysis_task = asyncio.create_task(analyze_market_trends())
        market_assets, market_analysis = await asyncio.gather(market_prices_task, market_analysis_task)
        if not market_assets or not market_analysis: raise HTTPException(status_code=503, detail="Market context unavailable.")
        required_analysis = ["Equity", "Crypto", "Gold", "FD"]; required_assets = ["Indices", "Crypto", "Gold", "FD"] # Added FD
        if not all(k in market_analysis for k in required_analysis) or not all(k in market_assets for k in required_assets): logger.error(f"Context missing keys. Assets:{market_assets.keys()}, Analysis:{market_analysis.keys()}"); raise HTTPException(status_code=503, detail="Incomplete context data.")
        response = MarketContextResponse(assets=market_assets, analysis=market_analysis)
        market_context_cache["timestamp"] = now; market_context_cache["data"] = response.model_dump(); logger.info("Dependency: Cache updated.")
        return response
    except Exception as e: logger.error(f"Dependency Error: {e}", exc_info=True); market_context_cache["data"] = None; market_context_cache["timestamp"] = 0; raise HTTPException(status_code=500, detail="Internal error fetching market context.")

@app.get("/market/context", response_model=MarketContextResponse, summary="Get Market Context (Prices + Trends)", tags=["Market"])
async def get_market_context_endpoint(market_context: MarketContextResponse = Depends(get_current_market_context)):
    logger.info("API Endpoint: /market/context accessed")
    return market_context

# **** UPDATED ENDPOINT ****
@app.post("/goal-recommendation", response_model=GoalRecommendationResponse, summary="Get Goal-Based Recommendation (5 Assets)", tags=["Recommendation"])
async def get_goal_recommendation(goal_request: GoalInfoRequest, market_context: MarketContextResponse = Depends(get_current_market_context)):
    """ Calculates required investment & 5-asset allocation, provides breakdown & explanation. """
    logger.info(f"API Request: /goal-recommendation - Goal='{goal_request.goal_name}', Risk={goal_request.risk_profile}, Duration={goal_request.goal_duration_years}, Target=₹{goal_request.goal_target_amount:.2f}")
    try:
        if not market_context or not market_context.analysis: raise HTTPException(status_code=503, detail="Market analysis unavailable.")
        required_analysis_keys = ["Equity", "Crypto", "Gold", "FD"]; # Need all for context/allocation
        if not all(key in market_context.analysis for key in required_analysis_keys): raise HTTPException(status_code=503, detail="Incomplete market analysis data.")

        # 1. Calculate 5-Asset Allocation
        allocation = calculate_portfolio_allocation(goal_request.risk_profile, goal_request.goal_duration_years, market_context.analysis)
        if not allocation: raise HTTPException(status_code=500, detail="Failed to calculate allocation.")

        # 2. Calculate Estimated Portfolio Return (Weighted Average of 5 assets)
        estimated_portfolio_return = sum(percentage * ESTIMATED_RETURNS.get(asset, 0.0) for asset, percentage in allocation.items())
        logger.info(f"Estimated Portfolio Annual Return (5 Assets): {estimated_portfolio_return:.4f} ({(estimated_portfolio_return * 100):.2f}%)")
        if estimated_portfolio_return < -0.5: logger.warning("Estimated return highly negative.")

        # 3. Calculate REQUIRED periodic investments
        required_investments: Dict[str, Optional[float]] = {}
        periods_config = {'monthly': 12, 'quarterly': 4, 'yearly': 1}; calculation_successful = False
        for period_name, periods_per_year in periods_config.items():
            req_inv = calculate_required_periodic_investment(future_value=goal_request.goal_target_amount, annual_rate=estimated_portfolio_return, years=goal_request.goal_duration_years, periods_per_year=periods_per_year)
            required_investments[period_name] = req_inv
            if req_inv is not None: calculation_successful = True
        if not calculation_successful: raise HTTPException(status_code=400, detail="Could not calculate required investments.")

        # 4. Calculate Investment Breakdown for each period
        investment_periods_breakdown: List[InvestmentBreakdownPeriod] = []
        for period_name, req_total in required_investments.items():
            if req_total is not None and req_total > 0:
                breakdown_dict = calculate_investment_breakdown(required_periodic_investment=req_total, allocation=allocation)
                investment_periods_breakdown.append(InvestmentBreakdownPeriod(period=period_name, required_total_investment=req_total, breakdown=breakdown_dict))
            else:
                 investment_periods_breakdown.append(InvestmentBreakdownPeriod(period=period_name, required_total_investment=req_total if req_total is not None else 0, breakdown={}))
                 logger.warning(f"Required investment for {period_name} is {req_total}. Breakdown empty.")

        # 5. Generate Combined Explanation (LLM or fallback) - Pass 5-asset allocation
        explanation_text = await generate_llm_explanation_combined(allocation, goal_request, market_context.analysis, required_investments, estimated_portfolio_return)

        # 6. Construct final response
        response = GoalRecommendationResponse(
            goal_name=goal_request.goal_name, goal_target_amount=goal_request.goal_target_amount, goal_duration_years=goal_request.goal_duration_years,
            estimated_portfolio_return=estimated_portfolio_return, allocation=allocation,
            required_investment_periods=investment_periods_breakdown, explanation=explanation_text
        )
        return response
    except HTTPException as http_exc: raise http_exc
    except Exception as e: logger.error(f"Unexpected error in /goal-recommendation: {e}", exc_info=True); raise HTTPException(status_code=500, detail="Internal server error generating goal recommendation.")

# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    print(f"--- Smart Investment Advisor Server Starting (v{app.version}) ---")
    print(f" - Indices: {INDEX_SYMBOLS_TO_FETCH}, Crypto: {CRYPTO_IDS}")
    print(f" - Gold Price (INR/g): {HARDCODED_GOLD_PRICE_INR_PER_GRAM}, Gold Trend: {HARDCODED_GOLD_TREND}")
    print(f" - Allowed Origins: {ALLOWED_ORIGINS}")
    print(f" - Estimated Returns Used (Illustrative): {ESTIMATED_RETURNS}") # Show estimates used
    llm_stat_msg = "Available" if GROQ_AVAILABLE and async_groq_client else "Disabled"
    print(f" - Groq LLM Status: {llm_stat_msg}")
    if GROQ_AVAILABLE and async_groq_client: print("\n *** SECURITY WARNING: Using hardcoded Groq API key! ***\n")
    elif "YOUR_GROQ_API_KEY_HERE" in HARDCODED_GROQ_API_KEY: print("\n *** WARNING: Groq Key placeholder. LLM disabled. ***\n")
    elif not GROQ_AVAILABLE: print("\n *** WARNING: Groq lib not installed/failed init. LLM disabled. ***\n")
    print(f" -> Starting Uvicorn server on http://127.0.0.1:{SERVER_PORT}")
    print(f" -> API documentation at: http://127.0.0.1:{SERVER_PORT}/docs")
    print(" -> Press CTRL+C to stop.")
    uvicorn.run("investment_server:app", host="127.0.0.1", port=SERVER_PORT, reload=True)