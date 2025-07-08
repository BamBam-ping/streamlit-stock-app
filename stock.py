import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
import sys

# --- Configuration ---
TICKERS = [
    "MSFT",      # Microsoft (Tech/Growth, Software, Cloud)
    "AAPL",      # Apple (Tech/Growth, Hardware, Services)
    "NVDA",      # NVIDIA (Tech/Hyper-Growth, Semiconductor, AI)
    "GOOGL",     # Alphabet (Tech/Growth, Internet Services)
    "AMZN",      # Amazon (Tech/Growth, E-commerce, Cloud)
    "TSLA",      # Tesla (Tech/Growth, EV, AI)
    "PLTR",      # Palantir (Tech/Growth, AI)
    "AMD",       # AMD (Tech/Growth, Semiconductor)
    "TSM",       # TSMC (Semiconductor Foundry)
    "ORCL",      # Oracle (Software, Cloud)
    "ADBE",      # Adobe (Tech/Growth, Software)
    "LLY",       # Eli Lilly (Healthcare/Growth, Pharma)
    "UNH",       # UnitedHealth Group (Healthcare/Growth, Managed Health Services)
    "VRTX",      # Vertex Pharmaceuticals (Bio/Growth, Pharma)
    "REGN",      # Regeneron Pharmaceuticals (Bio/Growth, Pharma)
    "JPM",       # JPMorgan Chase (Finance/Value, Banking)
    "V",         # Visa (Tech/Growth, Payment Services)
    "MS",        # Morgan Stanley (Finance)
    "JNJ",       # Johnson & Johnson (Healthcare/Value, Consumer Staples, Dividend)
    "HOOD",      # Robinhood (Fintech)
    "SPY",       # SPDR S&P 500 ETF (US Large Cap Market)
    "QQQ",       # Invesco QQQ Trust (Nasdaq 100 Tech/Growth focused)
    "SCHD",      # Schwab U.S. Dividend Equity ETF (US High Dividend)
]

# Dictionary to store ticker descriptions
TICKER_DESCRIPTIONS = {
    "MSFT": "마이크로소프트 (기술/성장, 소프트웨어, 클라우드)",
    "AAPL": "애플 (기술/성장, 하드웨어, 서비스)",
    "NVDA": "엔비디아 (기술/초고성장, 반도체, AI)",
    "GOOGL": "알파벳 (기술/성장, 인터넷 서비스)",
    "AMZN": "아마존 (기술/성장, 이커머스, 클라우드)",
    "TSLA": "테슬라 (기술/성장, 전기차, AI)",
    "PLTR": "팔란티어 (기술/성장, AI)",
    "AMD": "AMD (기술/성장, 반도체)",
    "TSM": "TSMC (반도체 파운드리)",
    "ORCL": "오라클 (소프트웨어, 클라우드)",
    "ADBE": "어도비 (기술/성장, 소프트웨어)",
    "LLY": "일라이 릴리 (헬스케어/성장, 제약)",
    "UNH": "유나이티드헬스그룹 (헬스케어/성장, 관리형 건강 서비스)",
    "VRTX": "버텍스 파마슈티컬스 (바이오/성장, 제약)",
    "REGN": "리제네론 파마슈티컬스 (바이오/성장, 제약)",
    "JPM": "JP모건 체이스 (금융/가치, 은행)",
    "V": "비자 (기술/성장, 결제 서비스)",
    "MS": "모건 스탠리 (금융)",
    "JNJ": "존슨앤존슨 (헬스케어/가치, 필수 소비재, 배당)",
    "HOOD": "로빈후드 (핀테크)",
    "SPY": "SPDR S&P 500 ETF (미국 대형주 시장 전체)",
    "QQQ": "Invesco QQQ Trust (나스닥 100 기술/성장주 중심)",
    "SCHD": "Schwab U.S. Dividend Equity ETF (미국 고배당주)",
}

# Sector mapping for tickers (a ticker can belong to multiple sectors)
SECTOR_MAPPING = {
    "AI": ["NVDA", "PLTR", "MSFT", "GOOGL", "TSLA"],
    "반도체": ["NVDA", "AMD", "TSM"],
    "기술/성장": ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "TSLA", "PLTR", "AMD", "ORCL", "ADBE", "V", "QQQ"],
    "헬스케어/바이오": ["LLY", "UNH", "VRTX", "REGN", "JNJ"],
    "금융": ["JPM", "V", "MS", "HOOD"],
    "ETF": ["SPY", "QQQ", "SCHD"],
    "필수 소비재/방어": ["JNJ"]
}


END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d") # Approx. 2 years of data
MIN_DATA_REQUIRED_FOR_INDICATORS = 180 # Minimum daily data required for indicator calculation

# --- Helper Functions ---

@st.cache_data
def download_stock_data(ticker, start, end):
    """Downloads stock data."""
    return yf.download(ticker, start=start, end=end)

@st.cache_data
def download_macro_data(start, end):
    """Downloads VIX, 10-year Treasury yield, 3-month Treasury yield, S&P500, Nasdaq, and DXY data."""
    macro_tickers = {
        "VIX": "^VIX",
        "US10Y": "^TNX",
        "US3M": "^IRX",      # US 3-month Treasury yield
        "S&P500": "^GSPC",    # S&P 500 Index
        "NASDAQ": "^IXIC",    # Nasdaq Composite Index
        "DXY": "DX-Y.NYB"     # Dollar Index (Yahoo Finance ticker)
    }
    retrieved_data = {}
    
    # Fetch at least 2 days of data for daily change calculation
    fetch_start_date = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d") # Ensure enough data

    for name, ticker_symbol in macro_tickers.items():
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            data = ticker_obj.history(start=fetch_start_date, end=end) 
            
            if not data.empty and not data['Close'].dropna().empty:
                current_value = data['Close'].dropna().iloc[-1].item()
                previous_value = data['Close'].dropna().iloc[-2].item() if len(data['Close'].dropna()) >= 2 else np.nan
                
                change = current_value - previous_value if not np.isnan(previous_value) else np.nan
                
                retrieved_data[name] = {
                    "value": current_value,
                    "change": change
                }
            else:
                retrieved_data[name] = {"value": np.nan, "change": np.nan}
        except Exception as e:
            retrieved_data[name] = {"value": np.nan, "change": np.nan}
    
    # Calculate Yield Spread (10-year - 3-month)
    us10y_val = retrieved_data.get("US10Y", {}).get("value", np.nan)
    us3m_val = retrieved_data.get("US3M", {}).get("value", np.nan)
    
    if not np.isnan(us10y_val) and not np.isnan(us3m_val):
        retrieved_data["Yield_Spread_10Y_3M"] = {"value": us10y_val - us3m_val, "change": np.nan} # No daily change for spread
    else:
        retrieved_data["Yield_Spread_10Y_3M"] = {"value": np.nan, "change": np.nan}

    return retrieved_data

def macro_filter(macro_data):
    """Classifies market conditions based on macroeconomic indicators."""
    vix_val = macro_data.get("VIX", {}).get("value", np.nan)
    us10y_val = macro_data.get("US10Y", {}).get("value", np.nan)

    # Return 'Data Insufficient' if any data is NaN
    if np.isnan(vix_val) or np.isnan(us10y_val):
        return "데이터 부족"

    # Detailed market condition classification
    if vix_val < 18:  # Low volatility
        if us10y_val < 3.5:
            return "강세 (저변동)"
        elif us10y_val < 4.0:
            return "강세 (중변동)"
        else:
            return "강세 (고금리)"
    elif 18 <= vix_val <= 25: # Normal volatility
        if us10y_val < 4.0:
            return "중립 (저변동)"
        else:
            return "중립 (고금리)"
    else: # vix_val > 25 (High volatility)
        if us10y_val < 4.0:
            return "약세 (고변동)"
        else:
            return "약세 (극변동)"

def soften_signal(signal, market_condition):
    """Softens signals based on market conditions (excluding Strong Buy/Sell)."""
    if signal in ["강력 매수", "강력 매도"]:
        return signal

    # Adjust conditions based on market state
    if market_condition in ["약세 (극변동)", "약세 (고변동)", "고금리", "중립 (고금리)"]:
        if "매수" in signal:
            return "관망"
        if "매도" in signal:
            return signal
    return signal

def adjust_score(score, market_condition):
    """Adjusts recommendation scores based on market conditions."""
    # Adjust conditions based on market state
    if market_condition in ["약세 (극변동)", "약세 (고변동)"]:
        return max(0, score * 0.7)
    elif market_condition in ["고금리", "중립 (고금리)"]:
        return max(0, score * 0.8)
    elif market_condition in ["강세 (저변동)", "강세 (중변동)", "강세 (고금리)"]:
        return min(100, score * 1.1)
    return score

# --- Technical Indicator Calculation Function ---
def calc_indicators(df):
    """Calculates and adds technical indicators to the given stock dataframe."""
    if df.empty:
        return df

    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    n_stoch = 14
    m_stoch = 3
    df['Lowest_Low'] = df['Low'].rolling(window=n_stoch).min()
    df['Highest_High'] = df['High'].rolling(window=n_stoch).max()
    
    # Prevent division by zero when calculating %K
    denominator = df['Highest_High'] - df['Lowest_Low']
    df['%K'] = np.where(denominator != 0, ((df['Close'] - df['Lowest_Low']) / denominator) * 100, 50) # Set to 50 (neutral) if zero
    df['%D'] = df['%K'].rolling(window=m_stoch).mean()
    
    # CCI (Commodity Channel Index)
    n_cci = 20
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_MA'] = df['TP'].rolling(window=n_cci).mean()
    df['Mean_Deviation'] = df['TP'].rolling(window=n_cci).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (df['TP'] - df['TP_MA']) / (0.015 * df['Mean_Deviation'])

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_StdDev'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_prev_close = abs(df['High'] - df['Close'].shift())
    low_prev_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.DataFrame({'HL': high_low, 'HPC': high_prev_close, 'LPC': low_prev_close}).max(axis=1)
    df['ATR'] = tr.ewm(span=14, adjust=False).mean()

    # ADX (Average Directional Index)
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)

    df['TR_ADX'] = tr

    df['+DI14'] = (df['+DM'].rolling(window=14).sum() / df['TR_ADX'].rolling(window=14).sum()) * 100
    df['-DI14'] = (df['-DM'].rolling(window=14).sum() / df['TR_ADX'].rolling(window=14).sum()) * 100

    df['DX'] = abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14']) * 100
    df['ADX'] = df['DX'].ewm(span=14, adjust=False).mean()

    # Volume Moving Average
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # Bollinger Band Squeeze Breakout Flag
    window_squeeze = 20
    df['Band_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['Band_Width_MA'] = df['Band_Width'].rolling(window=window_squeeze).mean()

    df['BB_Squeeze_Up_Breakout'] = False
    df['BB_Squeeze_Down_Breakout'] = False

    for i in range(1, len(df)):
        if i < window_squeeze:
            continue

        is_squeeze = df['Band_Width_MA'].iloc[i-1] < (df['BB_Middle'].iloc[i-1] * 0.02)

        is_breakout_up = df['Close'].iloc[i] > df['BB_Upper'].iloc[i] and df['Close'].iloc[i-1] <= df['BB_Upper'].iloc[i-1]
        is_volume_confirm_up = df['Volume'].iloc[i] > df['Volume_MA20'].iloc[i] * 1.5
        if is_squeeze and is_breakout_up and is_volume_confirm_up:
            df.loc[df.index[i], 'BB_Squeeze_Up_Breakout'] = True

        is_breakout_down = df['Close'].iloc[i] < df['BB_Lower'].iloc[i] and df['Close'].iloc[i-1] >= df['BB_Lower'].iloc[i-1]
        is_volume_confirm_down = df['Volume'].iloc[i] > df['Volume_MA20'].iloc[i] * 1.5
        if is_squeeze and is_breakout_down and is_volume_confirm_down:
            df.loc[df.index[i], 'BB_Squeeze_Down_Breakout'] = True

    df = df.drop(columns=['Band_Width', 'Band_Width_MA'])

    # Check for minimum data required for indicators
    if len(df) < MIN_DATA_REQUIRED_FOR_INDICATORS:
        return pd.DataFrame()

    # Remove NaN values generated at the beginning of indicator calculations
    df = df.dropna(subset=['MA20', 'MACD', 'RSI', 'BB_Middle', 'ATR', 'ADX', 'Volume_MA20', '%K', '%D', 'CCI', 'BB_Squeeze_Up_Breakout', 'BB_Squeeze_Down_Breakout'])
    return df

# --- Buy/Sell Signal Detection Functions ---

def is_macd_golden_cross(prev_row, current_row):
    """Checks for MACD Golden Cross."""
    return prev_row['MACD'] < prev_row['Signal'] and current_row['MACD'] > current_row['Signal']

def is_macd_dead_cross(prev_row, current_row):
    """Checks for MACD Dead Cross."""
    return prev_row['MACD'] > prev_row['Signal'] and current_row['MACD'] < current_row['Signal']

def is_rsi_oversold(current_row):
    """Checks if RSI is in oversold territory."""
    return current_row['RSI'] < 35

def is_rsi_overbought(current_row):
    """Checks if RSI is in overbought territory."""
    return current_row['RSI'] > 65

def is_ma_cross_up(prev_row, current_row):
    """Checks if closing price crossed above moving average."""
    return prev_row['Close'] < prev_row['MA20'] and current_row['Close'] > current_row['MA20']

def is_ma_cross_down(prev_row, current_row):
    """Checks if closing price crossed below moving average."""
    return prev_row['Close'] > prev_row['MA20'] and current_row['Close'] < current_row['MA20']

def is_volume_surge(current_row):
    """Checks for a surge in trading volume."""
    return current_row['Volume'] > (current_row['Volume_MA20'] * 1.5)

def is_bullish_divergence(prev2_row, prev_row, current_row):
    """Checks for bullish divergence (price falling, RSI rising)."""
    # Ensure all rows are valid and not NaN for the relevant columns
    if any(pd.isna(r[col]) for r in [prev2_row, prev_row, current_row] for col in ['Low', 'RSI']):
        return False

    price_low_decreasing = current_row['Low'] < prev_row['Low'] and prev_row['Low'] < prev2_row['Low']
    rsi_low_increasing = current_row['RSI'] > prev_row['RSI'] and prev_row['RSI'] > prev2_row['RSI']
    # Higher reliability if RSI occurs near oversold territory
    return price_low_decreasing and rsi_low_increasing and current_row['RSI'] < 50

def is_bearish_divergence(prev2_row, prev_row, current_row):
    """Checks for bearish divergence (price rising, RSI falling)."""
    # Ensure all rows are valid and not NaN for the relevant columns
    if any(pd.isna(r[col]) for r in [prev2_row, prev_row, current_row] for col in ['High', 'RSI']):
        return False

    price_high_increasing = current_row['High'] > prev_row['High'] and prev_row['High'] > prev2_row['High']
    rsi_high_decreasing = current_row['RSI'] < prev_row['RSI'] and prev_row['RSI'] < prev2_row['RSI']
    # Higher reliability if RSI occurs near overbought territory
    return price_high_increasing and rsi_high_decreasing and current_row['RSI'] > 50

def is_hammer_candlestick(current_row, prev_row):
    """Checks for a Hammer candlestick pattern (reversal signal in a downtrend)."""
    if any(pd.isna(current_row[col]) for col in ['Open', 'Close', 'High', 'Low']):
        return False

    open_price = current_row['Open']
    close_price = current_row['Close']
    high_price = current_row['High']
    low_price = current_row['Low']

    body = abs(close_price - open_price)
    lower_shadow = min(open_price, close_price) - low_price
    upper_shadow = high_price - max(open_price, close_price)
    total_range = high_price - low_price

    if total_range == 0: return False

    # Stricter conditions: small body, long lower shadow, very short upper shadow
    is_small_body = body <= 0.3 * total_range
    has_long_lower_shadow = lower_shadow >= 2 * body
    has_small_upper_shadow = upper_shadow <= 0.1 * body

    return is_small_body and has_long_lower_shadow and has_small_upper_shadow

def is_stoch_oversold(current_row):
    """Checks if Stochastic is in oversold territory."""
    return current_row['%K'] < 20 and current_row['%D'] < 20

def is_stoch_overbought(current_row):
    """Checks if Stochastic is in overbought territory."""
    return current_row['%K'] > 80 and current_row['%D'] > 80

def is_stoch_golden_cross(prev_row, current_row):
    """Checks for Stochastic Golden Cross."""
    return prev_row['%K'] < prev_row['%D'] and current_row['%K'] > current_row['%D'] and current_row['%K'] < 80

def is_stoch_dead_cross(prev_row, current_row):
    """Checks for Stochastic Dead Cross."""
    return prev_row['%K'] > prev_row['%D'] and current_row['%K'] < current_row['%D'] and current_row['%K'] > 20

def is_cci_oversold(current_row):
    """Checks if CCI is in oversold territory."""
    return current_row['CCI'] < -100

def is_cci_overbought(current_row):
    """Checks if CCI is in overbought territory."""
    return current_row['CCI'] > 100

# --- Smart Signal Logic ---
def smart_signal_row(row, prev_row, prev2_row):
    """Generates smart trading signals for individual rows."""
    # Check for required indicators (NaN handling)
    required_indicators = ['MACD', 'Signal', 'MACD_Hist', 'RSI', 'MA20', 'Volume_MA20', 'ATR', 'ADX', '%K', '%D', 'CCI', 'BB_Squeeze_Up_Breakout', 'BB_Squeeze_Down_Breakout']
    if any(pd.isna(row[ind]) for ind in required_indicators):
        return "관망"

    current_close = row['Close']
    prev_close = prev_row['Close']
    macd_hist_direction = row['MACD_Hist'] - prev_row['MACD_Hist']

    # 1. Strong Buy/Sell Signals (Composite Indicators, Highest Priority)
    # Strong Buy: MACD Golden Cross + MA20 Cross Up + Volume Surge + Strong Uptrend (ADX > 25, +DI > -DI) + Stochastic Golden Cross + RSI Exiting Oversold
    if (is_macd_golden_cross(prev_row, row) and
        is_ma_cross_up(prev_row, row) and
        is_volume_surge(row) and
        row['ADX'] > 25 and row['+DI14'] > row['-DI14'] and
        is_stoch_golden_cross(prev_row, row) and
        prev_row['RSI'] <= 30 and row['RSI'] > 30):
        return "강력 매수"

    # Strong Sell: MACD Dead Cross + MA20 Cross Down + Volume Surge + Strong Downtrend (ADX > 25, +DI < -DI) + Stochastic Dead Cross + RSI Exiting Overbought
    if (is_macd_dead_cross(prev_row, row) and
        is_ma_cross_down(prev_row, row) and
        is_volume_surge(row) and
        row['ADX'] > 25 and row['+DI14'] < row['-DI14'] and
        is_stoch_dead_cross(prev_row, row) and
        prev_row['RSI'] >= 70 and row['RSI'] < 70):
        return "강력 매도"

    # 2. Bollinger Band Squeeze Breakout (Strong Trend Reversal/Start)
    if row['BB_Squeeze_Up_Breakout']:
        return "강력 매수"
    if row['BB_Squeeze_Down_Breakout']:
        return "강력 매도"

    # 3. RSI Extreme Values and Reversal Signals
    if row['RSI'] >= 80:
        return "익절 매도"
    if row['RSI'] <= 20:
        return "신규 매수"

    # 4. Stochastic Extreme Values and Reversal Signals
    if is_stoch_overbought(row) and is_stoch_dead_cross(prev_row, row):
        return "매도"
    if is_stoch_oversold(row) and is_stoch_golden_cross(prev_row, row):
        return "신규 매수"

    # 5. CCI Extreme Values and Reversal Signals
    if is_cci_overbought(row) and row['CCI'] < prev_row['CCI']: # Overbought and turning down
        return "매도 고려"
    if is_cci_oversold(row) and row['CCI'] > prev_row['CCI']: # Oversold and turning up
        return "매수 고려"

    # 6. Momentum Change (MACD Hist)
    if row['MACD_Hist'] < 0 and macd_hist_direction > 0: # MACD Histogram negative and turning up (strengthening buy momentum)
        return "매수 고려"
    if row['MACD_Hist'] > 0 and macd_hist_direction < 0: # MACD Histogram positive and turning down (strengthening sell momentum)
        return "매도 고려"

    # 7. General Buy/Sell Signals
    if is_macd_golden_cross(prev_row, row):
        return "신규 매수"
    if is_macd_dead_cross(prev_row, row):
        return "매도"

    if is_ma_cross_up(prev_row, row):
        return "신규 매수"
    if is_ma_cross_down(prev_row, row):
        return "매도"

    # 8. Auxiliary Signals
    # Divergence detection requires 3 candles, check if prev2_row is valid
    if prev2_row is not None:
        if is_bullish_divergence(prev2_row, prev_row, row):
            return "반등 가능성"
        if is_bearish_divergence(prev2_row, prev_row, row):
            return "하락 가능성"
    
    if is_hammer_candlestick(row, prev_row): # Hammer candlestick (reversal)
        return "반전 신호"
    
    # Bollinger Band Touch/Breakout (non-squeeze)
    if current_close > row['BB_Upper'] and prev_close <= prev_row['BB_Upper']: # Price breaks above upper Bollinger Band
        if row['RSI'] > 70: return "익절 매도"
        else: return "보유"
    if current_close < row['BB_Lower'] and prev_close >= prev_row['BB_Lower']: # Price breaks below lower Bollinger Band
        if row['RSI'] < 30: return "신규 매수"
        else: return "관망"

    # Otherwise
    return "관망"

def smart_signal(df_with_signals):
    """Returns the most recent valid trading signal from the dataframe."""
    if df_with_signals.empty:
        return "데이터 부족"
    # Return the most recent valid signal
    last_valid_signal = df_with_signals['TradeSignal'].iloc[-1]
    return last_valid_signal

# --- Calculate Recommendation Score ---
def compute_recommendation_score(last, prev_row, per, market_cap, forward_pe, debt_to_equity):
    """Calculates a stock's recommendation score based on given indicators and market conditions."""
    score = 50 # Base score

    # MACD
    if last['MACD'] > last['Signal']: # MACD Golden Cross (Uptrend)
        score += 15
        if last['MACD_Hist'] > 0 and (last['MACD_Hist'] - prev_row['MACD_Hist']) > 0: # MACD Histogram positive and increasing (strong upward momentum)
            score += 10
    else: # MACD Dead Cross (Downtrend)
        score -= 15
        if last['MACD_Hist'] < 0 and (last['MACD_Hist'] - prev_row['MACD_Hist']) < 0: # MACD Histogram negative and decreasing (strong downward momentum)
            score -= 10

    # RSI
    if last['RSI'] > 70: # Extreme overbought
        score -= 25
    elif last['RSI'] < 30: # Extreme oversold
        score += 25
    elif last['RSI'] > 60: # Overbought (caution)
        score -= 10
    elif last['RSI'] < 40: # Oversold (opportunity)
        score += 10
    elif last['RSI'] > 50: # Bullish zone
        score += 5
    elif last['RSI'] < 50: # Bearish zone
        score -= 5

    # Stochastic
    if last['%K'] > last['%D'] and last['%K'] < 80: # Golden Cross and not overbought
        score += 10
    elif last['%K'] < last['%D'] and last['%K'] > 20: # Dead Cross and not oversold
        score -= 10
    if is_stoch_oversold(last):
        score += 15
    if is_stoch_overbought(last):
        score -= 15

    # CCI
    if last['CCI'] > 100: # Overbought
        score -= 10
    elif last['CCI'] < -100: # Oversold
        score += 10
    
    # Moving Averages
    if last['Close'] > last['MA20'] and last['MA20'] > last['MA60'] and last['MA60'] > last['MA120']: # Perfect bullish alignment
        score += 15
    elif last['Close'] < last['MA20'] and last['MA20'] < last['MA60'] and last['MA60'] < last['MA120']: # Perfect bearish alignment
        score -= 15
    elif last['Close'] > last['MA20'] and last['MA20'] > last['MA60']: # Bullish alignment
        score += 10
    elif last['Close'] < last['MA20'] and last['MA20'] < last['MA60']: # Bearish alignment
        score -= 10

    # ADX (Trend Strength and Direction)
    if last['ADX'] > 30: # Strong trend
        if last['+DI14'] > last['-DI14']: # Uptrend
            score += 10
        else: # Downtrend
            score -= 10
    elif last['ADX'] > 20: # Normal trend
        if last['+DI14'] > last['-DI14']: # Uptrend
            score += 5
        else: # Downtrend
            score -= 5
    elif last['ADX'] < 20: # Weak trend (sideways)
        score -= 5

    # Volume
    if last['Volume'] > last['Volume_MA20'] * 2: # Very high volume surge (significantly increases signal reliability)
        score += 10
    elif last['Volume'] > last['Volume_MA20'] * 1.5: # High volume surge (increases signal reliability)
        score += 5
    elif last['Volume'] < last['Volume_MA20'] * 0.5: # Low volume (decreased interest or weakening trend)
        score -= 5

    # PER (Price-to-Earnings Ratio) score
    if not np.isnan(per):
        if per > 0: # Only consider positive PER (negative means loss)
            if per < 15: # Potential undervaluation
                score += 10
            elif per >= 15 and per < 25: # Fair value
                score += 5
            elif per >= 25 and per < 40: # Slightly overvalued
                score -= 5
            else: # Overvalued
                score -= 10
        else: # PER is 0 or negative (loss-making company)
            score -= 15 # Significant penalty

    # Market Cap score
    if not np.isnan(market_cap):
        # Large cap (>$100B) for stability
        if market_cap >= 100_000_000_000:
            score += 5
        # Mid cap ($10B - $100B) neutral
        elif market_cap >= 10_000_000_000 and market_cap < 100_000_000_000:
            pass # Neutral
        # Small cap ($1B - $10B) for volatility
        elif market_cap >= 1_000_000_000 and market_cap < 10_000_000_000:
            score -= 2 # Small penalty (higher volatility)
        # Micro cap (<$1B) for risk
        else:
            score -= 5 # Larger penalty

    # Forward PE score
    if not np.isnan(forward_pe) and not np.isnan(per):
        if forward_pe < per: # Forward PE lower than current PE (positive, growth expectation)
            score += 7
        elif forward_pe > per: # Forward PE higher than current PE (negative, slowed growth or overvaluation)
            score -= 7
        # Otherwise neutral

    # Debt to Equity ratio score (lower is better)
    if not np.isnan(debt_to_equity):
        if debt_to_equity < 0.5: # Very low debt (healthy)
            score += 8
        elif debt_to_equity >= 0.5 and debt_to_equity < 1.0: # Moderate debt
            score += 4
        elif debt_to_equity >= 1.0 and debt_to_equity < 2.0: # Somewhat high debt
            score -= 4
        else: # Very high debt (risky)
            score -= 8

    # Strong weighting for TradeSignal itself (final signal reflection)
    if "강력 매수" in last['TradeSignal']:
        score += 30
    elif "신규 매수" in last['TradeSignal'] or "매수 고려" in last['TradeSignal'] or "반등 가능성" in last['TradeSignal']:
        score += 15
    elif "강력 매도" in last['TradeSignal']:
        score -= 30
    elif "매도" in last['TradeSignal'] or "익절 매도" in last['TradeSignal'] or "매도 고려" in last['TradeSignal'] or "하락 가능성" in last['TradeSignal']:
        score -= 15
    elif "관망" in last['TradeSignal'] or "보유" in last['TradeSignal'] or "반전 신호" in last['TradeSignal']:
        # Limit score range for neutral/hold signals
        score = max(score, 40) if score > 50 else min(score, 60)

    # Bollinger Band Breakout/Squeeze
    if last['BB_Squeeze_Up_Breakout']:
        score += 20
    if last['BB_Squeeze_Down_Breakout']:
        score -= 20
    
    # Normalize score (0-100)
    score = max(0, min(100, score))
    return score

# --- Determine Recommended Action and Percentage ---
def get_action_and_percentage_by_score(signal, score):
    """Determines specific action and percentage based on recommendation score and signal."""
    action_base = "관망"
    percentage = 0

    if "강력 매수" in signal:
        action_base = "신규 매수"
        percentage = 80 + (score - 80) * 0.5 if score > 80 else 80
    elif "신규 매수" in signal:
        action_base = "신규 매수"
        percentage = 50 + (score - 50) * 0.5 if score > 50 else 50
    elif "매수 고려" in signal or "반등 가능성" in signal:
        action_base = "신규 매수"
        percentage = 20 + (score - 30) * 0.5 if score > 30 else 20
        percentage = max(10, percentage)
    elif "익절 매도" in signal:
        action_base = "익절 매도"
        percentage = 50 + (score - 50) * 0.5 if score > 50 else 50
    elif "매도" in signal:
        action_base = "매도"
        percentage = 50 + (50 - score) * 0.5 if score < 50 else 50
    elif "매도 고려" in signal or "하락 가능성" in signal:
        action_base = "매도"
        percentage = 20 + (70 - score) * 0.5 if score < 70 else 20
        percentage = max(10, percentage)
    elif "강력 매도" in signal:
        action_base = "전량 매도"
        percentage = 80 + (80 - score) * 0.5 if score < 80 else 80
    elif "보유" in signal or "관망" in signal or "반전 신호" in signal:
        action_base = "관망"
        percentage = 0

    rounded_pct = max(0, min(100, round(percentage)))
    
    # Refine action text
    if "매수" in action_base:
        action_text = f"포트폴리오 분배 자산의 {rounded_pct}% 매수"
    elif "매도" in action_base or "익절 매도" in action_base:
        action_text = f"보유분의 {rounded_pct}% 매도"
    elif action_base == "전량 매도":
        action_text = f"보유분의 {rounded_pct}% 전량 매도"
    else: # Watch
        action_text = action_base

    return action_text, rounded_pct

# --- Convert Signal to Visual Symbol ---
def get_signal_symbol(signal_text):
    """Returns a visual symbol corresponding to the signal text and applies color."""
    if "매수" in signal_text or "반등 가능성" in signal_text:
        return "<span style='color: green;'>▲</span>"  # Green upward arrow
    elif "매도" in signal_text or "하락 가능성" in signal_text or "익절 매도" in signal_text:
        return "<span style='color: red;'>▼</span>"  # Red downward arrow
    else:
        return "<span style='color: orange;'>●</span>"  # Orange circle (neutral/watch)

# --- Convert Signal Text for UI Display ---
def get_display_signal_text(signal_original): # BB breakout is now a separate column, so removed as argument
    """Converts original signal text for UI display."""
    display_text = signal_original
    if display_text == "강력 매수":
        display_text = "강력 상승추세 가능성"
    elif display_text == "강력 매도":
        display_text = "강력 하락추세 가능성"
        
    return display_text

# --- Generate ChatGPT Prompt ---
def generate_chatgpt_prompt(ticker, rsi, macd, macd_hist, signal_line, atr, adx, k_stoch, d_stoch, cci, per, market_cap, forward_pe, debt_to_equity):
    """Generates a technical indicator prompt string for ChatGPT."""
    rsi_str = f"RSI: {rsi:.2f}" if not np.isnan(rsi) else "RSI: N/A"
    macd_str = f"MACD: {macd:.2f}, Signal: {signal_line:.2f}, Hist: {macd_hist:.2f}" if not np.isnan(macd) else "MACD: N/A"
    atr_str = f"ATR: {atr:.2f}" if not np.isnan(atr) else "ATR: N/A"
    adx_str = f"ADX: {adx:.2f}" if not np.isnan(adx) else "ADX: N/A"
    stoch_str = f"Stoch %K: {k_stoch:.2f}, %D: {d_stoch:.2f}" if not np.isnan(k_stoch) else "Stoch: N/A"
    cci_str = f"CCI: {cci:.2f}" if not np.isnan(cci) else "CCI: N/A"
    per_str = f"PER: {per:.2f}" if not np.isnan(per) else "PER: N/A"
    
    market_cap_str = f"시가총액: {market_cap/1_000_000_000:.2f}B" if not np.isnan(market_cap) else "시가총액: N/A"
    forward_pe_str = f"선행PER: {forward_pe:.2f}" if not np.isnan(forward_pe) else "선행PER: N/A"
    debt_to_equity_str = f"부채비율: {debt_to_equity:.2f}" if not np.isnan(debt_to_equity) else "부채비율: N/A"

    return f"{ticker}: {rsi_str}, {macd_str}, {atr_str}, {adx_str}, {stoch_str}, {cci_str}, {per_str}, {market_cap_str}, {forward_pe_str}, {debt_to_equity_str}"

# --- Email Sending Function ---
def send_email(subject, body, to_email, from_email, password, attachments=None):
    """Sends an email."""
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email  
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))

    if attachments:
        for file_path, file_name in attachments:
            try:
                with open(file_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=file_name)
                part['Content-Disposition'] = f'attachment; filename="{file_name}"'
                msg.attach(part)
            except FileNotFoundError:
                print(f"Warning: Attachment {file_path} not found. Skipping.")
            except Exception as e:
                print(f"Warning: Error processing attachment {file_path}: {e}. Skipping.")

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(from_email, password)
        server.send_message(msg)  
        server.quit()
        print("✅ Email sent successfully!")
    except smtplib.SMTPAuthenticationError:
        print("❌ Email sending failed: Authentication error. Check sender email or password (app password).")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")

# --- Main Execution Logic ---
if __name__ == '__main__':
    # Check if in email sending mode
    send_email_mode = "--send-email" in sys.argv

    # Set environment variables or default values
    SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'parkib63@naver.com')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'bdnj dicf dzea wdrq') # Recommended to use Google App Password
    RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL', 'inbeom.park@samsung.com')
    STREAMLIT_APP_URL = os.getenv('STREAMLIT_APP_URL', 'https://app-stock-app-bomipark.streamlit.app/')

    if send_email_mode:
        print("🚀 Starting in email report sending mode...")
        if SENDER_EMAIL == 'parkib63@naver.com' or \
           SENDER_PASSWORD == 'bdnj dicf dzea wdrq' or \
           RECEIVER_EMAIL == 'inbeom.park@samsung.com' or \
           STREAMLIT_APP_URL == 'https://app-stock-app-bomipark.streamlit.app/':
            print("🚨 Warning: Sender email info (SENDER_EMAIL, SENDER_PASSWORD) or receiver email (RECEIVER_EMAIL) or Streamlit app URL (STREAMLIT_APP_URL) not configured. Check environment variables or change values in code.")
            # sys.exit(1) # Commented out to prevent app termination when running in Streamlit

        email_summary_rows = []
        email_tech_summaries_text = []

        macro_start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        macro_data = download_macro_data(macro_start_date, END_DATE)
        market_condition = macro_filter(macro_data)

        email_body_parts = []
        email_body_parts.append(f"<h1>📈 US Stock Signal Dashboard - {datetime.now().strftime('%Y-%m-%d')}</h1>")
        email_body_parts.append(f"<h2>마켓 센티멘트 현황</h2>")
        
        # Macroeconomic data for email
        email_body_parts.append(f"<p>- VIX (변동성 지수): <b>{macro_data.get('VIX', {}).get('value', 'N/A'):.2f}</b> "
                                f"<span style='font-size: 0.8em; color: {'green' if macro_data.get('VIX', {}).get('change', 0) >= 0 else 'red'};'>"
                                f"({macro_data.get('VIX', {}).get('change', np.nan):+.2f})"
                                f"</span></p>")
        email_body_parts.append(f("<p>- 시장 상태: <b>{market_condition}</b></p>"))
        email_body_parts.append(f"<p>- 미 10년 금리: <b>{macro_data.get('US10Y', {}).get('value', 'N/A'):.2f}%</b> "
                                f"<span style='font-size: 0.8em; color: {'green' if macro_data.get('US10Y', {}).get('change', 0) >= 0 else 'red'};'>"
                                f"({macro_data.get('US10Y', {}).get('change', np.nan):+.2f}%)"
                                f"</span></p>")
        email_body_parts.append(f"<p>- 미 3개월 금리: <b>{macro_data.get('US3M', {}).get('value', 'N/A'):.2f}%</b> "
                                f"<span style='font-size: 0.8em; color: {'green' if macro_data.get('US3M', {}).get('change', 0) >= 0 else 'red'};'>"
                                f"({macro_data.get('US3M', {}).get('change', np.nan):+.2f})"
                                f"</span></p>")
        email_body_parts.append(f"<p>- S&P 500: <b>{macro_data.get('S&P500', {}).get('value', 'N/A'):.2f}</b> "
                                f"<span style='font-size: 0.8em; color: {'green' if macro_data.get('S&P500', {}).get('change', 0) >= 0 else 'red'};'>"
                                f"({macro_data.get('S&P500', {}).get('change', np.nan):+.2f})"
                                f"</span></p>")
        email_body_parts.append(f"<p>- NASDAQ: <b>{macro_data.get('NASDAQ', {}).get('value', 'N/A'):.2f}</b> "
                                f"<span style='font-size: 0.8em; color: {'green' if macro_data.get('NASDAQ', {}).get('change', 0) >= 0 else 'red'};'>"
                                f"({macro_data.get('NASDAQ', {}).get('change', np.nan):+.2f})"
                                f"</span></p>")
        email_body_parts.append(f"<p>- 달러인덱스 (DXY): <b>{macro_data.get('DXY', {}).get('value', 'N/A'):.2f}</b> "
                                f"<span style='font-size: 0.8em; color: {'green' if macro_data.get('DXY', {}).get('change', 0) >= 0 else 'red'};'>"
                                f"({macro_data.get('DXY', {}).get('change', np.nan):+.2f})"
                                f"</span></p>")
        email_body_parts.append(f"<p><b>자세한 분석 및 실시간 캔들스틱 차트를 보려면 아래 링크를 클릭하세요:</b> <a href='{STREAMLIT_APP_URL}'>{STREAMLIT_APP_URL}</a></p>")
        
        email_body_parts.append(f"<h2>개별 종목 분석 요약</h2>")
        email_body_parts.append("<table border='1' style='width:100%; border-collapse: collapse;'>")
        email_body_parts.append("<thead><tr>"
                                "<th>티커</th>"
                                "<th>종목명</th>"
                                "<th>현재가</th>"
                                "<th>일일변동</th>"
                                "<th>추천 시그널</th>"
                                "<th>추천 점수</th>"
                                "<th>권장 행동</th>"
                                "<th>BB 돌파</th>" # New column for BB Breakout
                                "</tr></thead><tbody>")

        for ticker in TICKERS:
            try:
                # Download stock data
                stock_data = download_stock_data(ticker, START_DATE, END_DATE)
                if stock_data.empty:
                    print(f"Warning: No stock data found for {ticker}. Skipping.")
                    continue

                # Calculate technical indicators
                df_with_indicators = calc_indicators(stock_data.copy())
                if df_with_indicators.empty:
                    print(f"Warning: Insufficient data for {ticker} to calculate indicators. Skipping.")
                    continue

                # Apply smart signals
                df_with_indicators['TradeSignal'] = "관망"
                for i in range(2, len(df_with_indicators)): # Requires at least 3 candles (current, prev, prev2)
                    current_row = df_with_indicators.iloc[i]
                    prev_row = df_with_indicators.iloc[i-1]
                    prev2_row = df_with_indicators.iloc[i-2]
                    df_with_indicators.loc[df_with_indicators.index[i], 'TradeSignal'] = smart_signal_row(current_row, prev_row, prev2_row)

                last_row = df_with_indicators.iloc[-1]
                prev_row_for_score = df_with_indicators.iloc[-2] if len(df_with_indicators) >= 2 else last_row # Fallback for prev_row

                current_price = last_row['Close']
                previous_close = df_with_indicators['Close'].iloc[-2] if len(df_with_indicators) >= 2 else np.nan
                daily_change = current_price - previous_close if not np.isnan(previous_close) else np.nan
                daily_change_pct = (daily_change / previous_close) * 100 if not np.isnan(previous_close) else np.nan

                # Fetch additional info from Yahoo Finance (PER, Market Cap, Forward PE, Debt to Equity)
                ticker_info = yf.Ticker(ticker).info
                per = ticker_info.get('trailingPE', np.nan)
                market_cap = ticker_info.get('marketCap', np.nan)
                forward_pe = ticker_info.get('forwardPE', np.nan)
                debt_to_equity = ticker_info.get('debtToEquity', np.nan) # Debt/Equity ratio

                # Soften signal and adjust score
                original_signal = smart_signal(df_with_indicators)
                softened_signal = soften_signal(original_signal, market_condition)
                
                recommendation_score = compute_recommendation_score(last_row, prev_row_for_score, per, market_cap, forward_pe, debt_to_equity)
                adjusted_recommendation_score = adjust_score(recommendation_score, market_condition)

                action_text, percentage_value = get_action_and_percentage_by_score(softened_signal, adjusted_recommendation_score)
                
                # Check BB Breakout status
                is_bb_up_breakout = last_row['BB_Squeeze_Up_Breakout']
                is_bb_down_breakout = last_row['BB_Squeeze_Down_Breakout']
                
                # Generate BB Breakout display text
                bb_breakout_text = ""
                if is_bb_up_breakout:
                    bb_breakout_text = "<span style='color: green;'>상향 돌파 ▲</span>"
                elif is_bb_down_breakout:
                    bb_breakout_text = "<span style='color: red;'>하향 돌파 ▼</span>"
                # If neither, bb_breakout_text remains an empty string as per request

                display_signal_text_base = get_display_signal_text(softened_signal) # BB indicator now in separate column

                # Add row to email summary table
                change_color = 'green' if daily_change >= 0 else 'red'
                email_summary_rows.append(f"<tr>"
                                          f"<td>{ticker}</td>"
                                          f"<td>{TICKER_DESCRIPTIONS.get(ticker, 'N/A')}</td>"
                                          f"<td>{current_price:.2f}</td>"
                                          f"<td style='color: {change_color};'>{daily_change:+.2f} ({daily_change_pct:+.2f}%)</td>"
                                          f"<td>{get_signal_symbol(softened_signal)} {display_signal_text_base}</td>" # BB indicator removed
                                          f"<td>{int(adjusted_recommendation_score)}</td>"
                                          f"<td>{action_text}</td>"
                                          f"<td>{bb_breakout_text}</td>" # Add BB Breakout column
                                          f"</tr>")
                
                # Technical indicator summary for ChatGPT prompt
                email_tech_summaries_text.append(
                    generate_chatgpt_prompt(
                        ticker,
                        last_row['RSI'],
                        last_row['MACD'],
                        last_row['MACD_Hist'],
                        last_row['Signal'],
                        last_row['ATR'],
                        last_row['ADX'],
                        last_row['%K'],
                        last_row['%D'],
                        last_row['CCI'],
                        per,
                        market_cap,
                        forward_pe,
                        debt_to_equity
                    )
                )

            except Exception as e:
                print(f"ERROR: Error processing {ticker}: {e}. Skipping.")
                email_summary_rows.append(f"<tr><td>{ticker}</td><td colspan='7'>데이터 처리 오류: {e}</td></tr>") # colspan 6 -> 7
                continue
        
        email_body_parts.append("".join(email_summary_rows))
        email_body_parts.append("</tbody></table>")

        email_body_parts.append(f"<h2>ChatGPT 분석용 기술적 지표 요약 (복사하여 사용)</h2>")
        email_body_parts.append("<pre style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>")
        email_body_parts.append("\n".join(email_tech_summaries_text))
        email_body_parts.append("</pre>")

        email_subject = f"📈 US 주식 시그널 보고서 - {datetime.now().strftime('%Y-%m-%d')}"
        full_email_body = "".join(email_body_parts)

        send_email(email_subject, full_email_body, RECEIVER_EMAIL, SENDER_EMAIL, SENDER_PASSWORD)

    else: # Streamlit app execution mode
        st.set_page_config(layout="wide", page_title="US 주식 시그널 대시보드")

        st.title("📈 US 주식 시그널 대시보드")
        st.markdown("미국 주식 시장의 주요 기술적 지표와 거시경제 데이터를 기반으로 매매 시그널 및 추천 점수를 제공합니다.")

        # --- Macroeconomic Indicators Section ---
        st.subheader("📊 마켓 센티멘트 현황")
        
        # Load cached macroeconomic data
        macro_start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d") # Macroeconomic data for 2 months
        macro_data = download_macro_data(macro_start_date, END_DATE)
        market_condition = macro_filter(macro_data)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="VIX (변동성 지수)", 
                      value=f"{macro_data.get('VIX', {}).get('value', np.nan):.2f}" if not np.isnan(macro_data.get('VIX', {}).get('value', np.nan)) else "N/A",
                      delta=f"{macro_data.get('VIX', {}).get('change', np.nan):.2f}" if not np.isnan(macro_data.get('VIX', {}).get('change', np.nan)) else None,
                      delta_color="inverse")
            st.metric(label="미 10년 금리", 
                      value=f"{macro_data.get('US10Y', {}).get('value', np.nan):.2f}%" if not np.isnan(macro_data.get('US10Y', {}).get('value', np.nan)) else "N/A",
                      delta=f"{macro_data.get('US10Y', {}).get('change', np.nan):.2f}%" if not np.isnan(macro_data.get('US10Y', {}).get('change', np.nan)) else None)
        with col2:
            st.metric(label="시장 상태", value=market_condition)
            st.metric(label="미 3개월 금리", 
                      value=f"{macro_data.get('US3M', {}).get('value', np.nan):.2f}%" if not np.isnan(macro_data.get('US3M', {}).get('value', np.nan)) else "N/A",
                      delta=f"{macro_data.get('US3M', {}).get('change', np.nan):.2f}%" if not np.isnan(macro_data.get('US3M', {}).get('change', np.nan)) else None)
        with col3:
            st.metric(label="S&P 500", 
                      value=f"{macro_data.get('S&P500', {}).get('value', np.nan):.2f}" if not np.isnan(macro_data.get('S&P500', {}).get('value', np.nan)) else "N/A",
                      delta=f"{macro_data.get('S&P500', {}).get('change', np.nan):.2f}" if not np.isnan(macro_data.get('S&P500', {}).get('change', np.nan)) else None)
            st.metric(label="NASDAQ", 
                      value=f"{macro_data.get('NASDAQ', {}).get('value', np.nan):.2f}" if not np.isnan(macro_data.get('NASDAQ', {}).get('value', np.nan)) else "N/A",
                      delta=f"{macro_data.get('NASDAQ', {}).get('change', np.nan):.2f}" if not np.isnan(macro_data.get('NASDAQ', {}).get('change', np.nan)) else None)
            st.metric(label="달러인덱스 (DXY)", 
                      value=f"{macro_data.get('DXY', {}).get('value', np.nan):.2f}" if not np.isnan(macro_data.get('DXY', {}).get('value', np.nan)) else "N/A",
                      delta=f"{macro_data.get('DXY', {}).get('change', np.nan):.2f}" if not np.isnan(macro_data.get('DXY', {}).get('change', np.nan)) else None)

        st.markdown("---")

        # --- Stock Selection Section ---
        st.sidebar.header("종목 선택")
        # Multiselect for individual tickers
        individual_selected_tickers = st.sidebar.multiselect(
            "개별 티커 선택 (다중 선택 가능):", 
            TICKERS, 
            default=TICKERS[0] if TICKERS else [], # Default to first ticker
            format_func=lambda x: f"{x} - {TICKER_DESCRIPTIONS.get(x, '')}"
        )

        # Multiselect for sectors
        sector_choices = list(SECTOR_MAPPING.keys())
        selected_sectors = st.sidebar.multiselect("분야 선택 (다중 선택 가능):", sector_choices, default=[])

        # Determine which tickers to display in the detailed analysis section
        tickers_for_detailed_analysis = []
        if selected_sectors:
            for sector in selected_sectors:
                tickers_for_detailed_analysis.extend(SECTOR_MAPPING.get(sector, []))
            tickers_for_detailed_analysis = sorted(list(set(tickers_for_detailed_analysis))) # Remove duplicates and sort
        else:
            tickers_for_detailed_analysis = individual_selected_tickers


        if not tickers_for_detailed_analysis and not selected_sectors: # Only show message if no individual tickers selected AND no sectors selected
            st.info("왼쪽 사이드바에서 분석할 종목 또는 분야를 하나 이상 선택해주세요.")
        else:
            # Store analysis results for all tickers (for summary table)
            all_ticker_analysis_results = []

            for ticker in TICKERS: # Iterate through ALL TICKERS to process data for the summary table
                try:
                    # Download stock data
                    stock_data = download_stock_data(ticker, START_DATE, END_DATE)
                    if stock_data.empty:
                        continue

                    # Calculate technical indicators
                    df_with_indicators = calc_indicators(stock_data.copy())
                    if df_with_indicators.empty:
                        continue

                    # Apply smart signals
                    df_with_indicators['TradeSignal'] = "관망"
                    for i in range(2, len(df_with_indicators)): # Requires at least 3 candles (current, prev, prev2)
                        current_row = df_with_indicators.iloc[i]
                        prev_row = df_with_indicators.iloc[i-1]
                        prev2_row = df_with_indicators.iloc[i-2]
                        df_with_indicators.loc[df_with_indicators.index[i], 'TradeSignal'] = smart_signal_row(current_row, prev_row, prev2_row)

                    last_row = df_with_indicators.iloc[-1]
                    prev_row_for_score = df_with_indicators.iloc[-2] if len(df_with_indicators) >= 2 else last_row # Fallback for prev_row

                    current_price = last_row['Close']
                    previous_close = df_with_indicators['Close'].iloc[-2] if len(df_with_indicators) >= 2 else np.nan
                    daily_change = current_price - previous_close if not np.isnan(previous_close) else np.nan
                    daily_change_pct = (daily_change / previous_close) * 100 if not np.isnan(previous_close) else np.nan

                    # Fetch additional info from Yahoo Finance (PER, Market Cap, Forward PE, Debt to Equity)
                    ticker_info = yf.Ticker(ticker).info
                    per = ticker_info.get('trailingPE', np.nan)
                    market_cap = ticker_info.get('marketCap', np.nan)
                    forward_pe = ticker_info.get('forwardPE', np.nan)
                    debt_to_equity = ticker_info.get('debtToEquity', np.nan) # Debt/Equity ratio

                    # Soften signal and adjust score
                    original_signal = smart_signal(df_with_indicators)
                    softened_signal = soften_signal(original_signal, market_condition)
                    
                    recommendation_score = compute_recommendation_score(last_row, prev_row_for_score, per, market_cap, forward_pe, debt_to_equity)
                    adjusted_recommendation_score = adjust_score(recommendation_score, market_condition)

                    action_text, percentage_value = get_action_and_percentage_by_score(softened_signal, adjusted_recommendation_score)
                    
                    # Check BB Breakout status
                    is_bb_up_breakout = last_row['BB_Squeeze_Up_Breakout']
                    is_bb_down_breakout = last_row['BB_Squeeze_Down_Breakout']

                    all_ticker_analysis_results.append({
                        'ticker': ticker,
                        'description': TICKER_DESCRIPTIONS.get(ticker, 'N/A'),
                        'current_price': current_price,
                        'daily_change': daily_change,
                        'daily_change_pct': daily_change_pct,
                        'softened_signal': softened_signal,
                        'is_bb_up_breakout': is_bb_up_breakout, # Store BB breakout status
                        'is_bb_down_breakout': is_bb_down_breakout, # Store BB breakout status
                        'adjusted_recommendation_score': adjusted_recommendation_score,
                        'action_text': action_text,
                        'last_row': last_row,
                        'prev_row_for_score': prev_row_for_score,
                        'per': per,
                        'market_cap': market_cap,
                        'forward_pe': forward_pe,
                        'debt_to_equity': debt_to_equity,
                        'df_with_indicators': df_with_indicators # Store full dataframe for charts
                    })

                except Exception as e:
                    # st.error(f"ERROR: Error processing {ticker}: {e}. Skipping this ticker.")
                    continue 

            # --- Overall Stock Trading Signal Status Table ---
            st.subheader("📋 전체 종목별 매매 시그널 현황")
            if all_ticker_analysis_results:
                summary_data = []
                for result in all_ticker_analysis_results:
                    change_color = 'green' if result['daily_change'] >= 0 else 'red'
                    
                    # Generate BB Breakout display text
                    bb_breakout_text = ""
                    if result['is_bb_up_breakout']:
                        bb_breakout_text = "<span style='color: green;'>상향 돌파 ▲</span>"
                    elif result['is_bb_down_breakout']:
                        bb_breakout_text = "<span style='color: red;'>하향 돌파 ▼</span>"
                    # If neither, bb_breakout_text remains an empty string as per request

                    summary_data.append({
                        "티커": result['ticker'],
                        "종목명": result['description'],
                        "현재가": f"{result['current_price']:.2f}",
                        "일일변동": f"<span style='color: {change_color};'>{result['daily_change']:+.2f} ({result['daily_change_pct']:+.2f}%)</span>",
                        "추천 시그널": f"{get_signal_symbol(result['softened_signal'])} {get_display_signal_text(result['softened_signal'])}",
                        "추천 점수": int(result['adjusted_recommendation_score']),
                        "권장 행동": result['action_text'],
                        "BB 돌파": bb_breakout_text # Add BB Breakout column
                    })
                
                # Render table with HTML for rich text
                html_table = "<table style='width:100%; border-collapse: collapse;'>"
                html_table += "<thead><tr style='background-color:#f0f0f0;'>"
                for col in summary_data[0].keys():
                    html_table += f"<th style='padding: 8px; border: 1px solid #ddd; text-align: left;'>{col}</th>"
                html_table += "</tr></thead><tbody>"
                for row in summary_data:
                    html_table += "<tr>"
                    for col_name, value in row.items():
                        html_table += f"<td style='padding: 8px; border: 1px solid #ddd;'>{value}</td>"
                    html_table += "</tr>"
                html_table += "</tbody></table>"
                st.markdown(html_table, unsafe_allow_html=True)
            else:
                st.info("유효한 데이터를 가져온 종목이 없습니다.")

            st.markdown("---")
            st.subheader("📊 개별 종목 상세 분석")

            # Filter analysis results based on selected tickers for detailed view
            filtered_analysis_results = [res for res in all_ticker_analysis_results if res['ticker'] in tickers_for_detailed_analysis]

            if not filtered_analysis_results:
                st.info("선택된 종목 또는 분야에 해당하는 상세 데이터를 표시할 수 없습니다.")
            else:
                for result in filtered_analysis_results:
                    ticker = result['ticker']
                    st.markdown(f"### {ticker} - {result['description']}")

                    change_color = "green" if result['daily_change'] >= 0 else "red"
                    st.write(f"**현재가:** ${result['current_price']:.2f}")
                    st.markdown(f"**일일 변동:** <span style='color:{change_color}'>{result['daily_change']:+.2f} ({result['daily_change_pct']:+.2f}%)</span>", unsafe_allow_html=True)
                    
                    # BB Breakout display text for detailed analysis (separate line)
                    bb_breakout_text_detail = ""
                    if result['is_bb_up_breakout']:
                        bb_breakout_text_detail = "<span style='color: green;'>**BB 스퀴즈 상향 돌파 발생! ▲**</span>"
                    elif result['is_bb_down_breakout']:
                        bb_breakout_text_detail = "<span style='color: red;'>**BB 스퀴즈 하향 돌파 발생! ▼**</span>"
                    
                    st.markdown(f"**최근 추천 시그널:** {get_signal_symbol(result['softened_signal'])} {get_display_signal_text(result['softened_signal'])}", unsafe_allow_html=True)
                    if bb_breakout_text_detail:
                        st.markdown(bb_breakout_text_detail, unsafe_allow_html=True)

                    st.write(f"**추천 점수 (0-100):** {int(result['adjusted_recommendation_score'])}")
                    st.write(f"**권장 행동:** {result['action_text']}")

                    st.markdown("---")
                    st.subheader(f"📈 {ticker} 캔들스틱 차트 및 기술적 지표")

                    df_with_indicators = result['df_with_indicators']

                    # Candlestick Chart
                    fig = go.Figure(data=[go.Candlestick(x=df_with_indicators.index,
                                                        open=df_with_indicators['Open'],
                                                        high=df_with_indicators['High'],
                                                        low=df_with_indicators['Low'],
                                                        close=df_with_indicators['Close'],
                                                        name='Candlestick')])
                    # Add Moving Averages
                    fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1)))
                    fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=1)))
                    fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA120'], mode='lines', name='MA120', line=dict(color='purple', width=1)))

                    # Add Bollinger Bands
                    fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dash')))
                    fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BB_Middle'], mode='lines', name='BB Middle', line=dict(color='gray', width=1)))
                    fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash')))

                    fig.update_layout(xaxis_rangeslider_visible=False, title=f'{ticker} 주가 차트', height=600)
                    st.plotly_chart(fig, use_container_width=True)

                    # MACD Chart
                    st.subheader(f"{ticker} MACD")
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['Signal'], mode='lines', name='Signal Line', line=dict(color='red')))
                    fig_macd.add_trace(go.Bar(x=df_with_indicators.index, y=df_with_indicators['MACD_Hist'], name='MACD Histogram', marker_color='green'))
                    fig_macd.update_layout(height=300)
                    st.plotly_chart(fig_macd, use_container_width=True)

                    # RSI Chart
                    st.subheader(f"{ticker} RSI")
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="과매도")
                    fig_rsi.update_layout(height=300)
                    st.plotly_chart(fig_rsi, use_container_width=True)

                    # Stochastic Chart
                    st.subheader(f"{ticker} Stochastic Oscillator")
                    fig_stoch = go.Figure()
                    fig_stoch.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['%K'], mode='lines', name='%K', line=dict(color='blue')))
                    fig_stoch.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['%D'], mode='lines', name='%D', line=dict(color='red')))
                    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="과매수")
                    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="과매도")
                    fig_stoch.update_layout(height=300)
                    st.plotly_chart(fig_stoch, use_container_width=True)

                    # CCI Chart
                    st.subheader(f"{ticker} Commodity Channel Index (CCI)")
                    fig_cci = go.Figure()
                    fig_cci.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['CCI'], mode='lines', name='CCI', line=dict(color='teal')))
                    fig_cci.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="과매수")
                    fig_cci.add_hline(y=-100, line_dash="dash", line_color="green", annotation_text="과매도")
                    fig_cci.update_layout(height=300)
                    st.plotly_chart(fig_cci, use_container_width=True)

                    # ADX Chart
                    st.subheader(f"{ticker} Average Directional Index (ADX)")
                    fig_adx = go.Figure()
                    fig_adx.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['ADX'], mode='lines', name='ADX', line=dict(color='black')))
                    fig_adx.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['+DI14'], mode='lines', name='+DI14', line=dict(color='green')))
                    fig_adx.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['-DI14'], mode='lines', name='-DI14', line=dict(color='red')))
                    fig_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="강한 추세")
                    fig_adx.update_layout(height=300)
                    st.plotly_chart(fig_adx, use_container_width=True)

                    st.markdown("---")
                    st.subheader(f"📋 {ticker} 주요 지표 요약 (최근 값)")
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    with col_sum1:
                        st.write(f"**MACD:** {result['last_row']['MACD']:.2f}")
                        st.write(f"**MACD Signal:** {result['last_row']['Signal']:.2f}")
                        st.write(f"**MACD Hist:** {result['last_row']['MACD_Hist']:.2f}")
                        st.write(f"**RSI:** {result['last_row']['RSI']:.2f}")
                        st.write(f"**Stoch %K:** {result['last_row']['%K']:.2f}")
                    with col_sum2:
                        st.write(f"**Stoch %D:** {result['last_row']['%D']:.2f}")
                        st.write(f"**CCI:** {result['last_row']['CCI']:.2f}")
                        st.write(f"**ADX:** {result['last_row']['ADX']:.2f}")
                        st.write(f"**+DI14:** {result['last_row']['+DI14']:.2f}")
                        st.write(f"**-DI14:** {result['last_row']['-DI14']:.2f}")
                    with col_sum3:
                        st.write(f"**MA20:** {result['last_row']['MA20']:.2f}")
                        st.write(f"**MA60:** {result['last_row']['MA60']:.2f}")
                        st.write(f"**MA120:** {result['last_row']['MA120']:.2f}")
                        st.write(f"**PER:** {result['per']:.2f}" if not np.isnan(result['per']) else "**PER:** N/A")
                        st.write(f"**시가총액:** {result['market_cap']/1_000_000_000:.2f}B" if not np.isnan(result['market_cap']) else "**시가총액:** N/A")
                        st.write(f"**선행PER:** {result['forward_pe']:.2f}" if not np.isnan(result['forward_pe']) else "**선행PER:** N/A")
                        st.write(f"**부채비율:** {result['debt_to_equity']:.2f}" if not np.isnan(result['debt_to_equity']) else "**부채비율:** N/A")

                    # Generate and display ChatGPT prompt
                    st.markdown("---")
                    st.subheader(f"🤖 {ticker} ChatGPT 분석용 프롬프트")
                    chatgpt_prompt = generate_chatgpt_prompt(
                        ticker,
                        result['last_row']['RSI'],
                        result['last_row']['MACD'],
                        result['last_row']['MACD_Hist'],
                        result['last_row']['Signal'],
                        result['last_row']['ATR'],
                        result['last_row']['ADX'],
                        result['last_row']['%K'],
                        result['last_row']['%D'],
                        result['last_row']['CCI'],
                        result['per'],
                        result['market_cap'],
                        result['forward_pe'],
                        result['debt_to_equity']
                    )
                    st.code(chatgpt_prompt, language='text')
                    st.info("위 텍스트를 복사하여 ChatGPT에 붙여넣어 추가 분석을 요청할 수 있습니다.")
                    st.markdown("---") # Separator for each stock analysis section
            
