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

# --- 설정 (Configuration) ---
TICKERS = [
    "MSFT",      # 마이크로소프트 (기술/성장, 소프트웨어, 클라우드)
    "AAPL",      # 애플 (기술/성장, 하드웨어, 서비스)
    "NVDA",      # 엔비디아 (기술/초고성장, 반도체, AI)
    "GOOGL",     # 알파벳 (기술/성장, 인터넷 서비스)
    "AMZN",      # 아마존 (기술/성장, 이커머스, 클라우드)
    "TSLA",      # 테슬라 (기술/성장, 전기차, AI)
    "PLTR",      # 팔란티어 (기술/성장, AI)
    "AMD",       # AMD (기술/성장, 반도체)
    "TSM",       # TSMC (반도체 파운드리)
    "ORCL",      # 오라클 (소프트웨어, 클라우드)
    "ADBE",      # 어도비 (기술/성장, 소프트웨어)
    "LLY",       # 일라이 릴리 (헬스케어/성장, 제약)
    "UNH",       # 유나이티드헬스그룹 (헬스케어/성장, 관리형 건강 서비스)
    "VRTX",      # 버텍스 파마슈티컬스 (바이오/성장, 제약)
    "REGN",      # 리제네론 파마슈티컬스 (바이오/성장, 제약)
    "JPM",       # JP모건 체이스 (금융/가치, 은행)
    "V",         # 비자 (기술/성장, 결제 서비스)
    "MS",        # 모건 스탠리 (금융)
    "JNJ",       # 존슨앤존슨 (헬스케어/가치, 필수 소비재, 배당) - 수정된 부분
    "HOOD",      # 로빈후드 (핀테크)
    "SPY",       # SPDR S&P 500 ETF (미국 대형주 시장 전체)
    "QQQ",       # Invesco QQQ Trust (나스닥 100 기술/성장주 중심)
    "SCHD",      # Schwab U.S. Dividend Equity ETF (미국 고배당주)
]

# 티커별 설명을 저장하는 딕셔너리
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

END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d") # 약 2년치 데이터
MIN_DATA_REQUIRED_FOR_INDICATORS = 180 # 지표 계산에 필요한 최소 일봉 데이터 수

# --- 보조 함수들 ---

@st.cache_data
def download_macro_data(start, end):
    """VIX, 10년 국채 금리, 3개월 국채 금리, S&P500, Nasdaq, 달러인덱스 데이터를 다운로드합니다."""
    macro_tickers = {
        "VIX": "^VIX",
        "US10Y": "^TNX",
        "US3M": "^IRX",      # 미 3개월 국채 금리
        "S&P500": "^GSPC",    # S&P 500 지수
        "NASDAQ": "^IXIC",    # Nasdaq Composite 지수
        "DXY": "DX-Y.NYB"     # 달러인덱스 (Yahoo Finance 티커)
    }
    retrieved_data = {}
    
    # 일일 변화 계산을 위해 최소 2일치 데이터 가져오기
    fetch_start_date = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d") # 충분한 데이터 확보

    for name, ticker_symbol in macro_tickers.items():
        try:
            # print(f"DEBUG: {name} ({ticker_symbol}) 데이터 다운로드 시도 중...")
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
                # print(f"DEBUG: {name} 데이터 성공적으로 가져옴: {current_value}, 변화: {change:.2f}")
            else:
                retrieved_data[name] = {"value": np.nan, "change": np.nan}
                # print(f"DEBUG: {name} 데이터가 비어있거나 'Close' 컬럼이 NaN입니다. NaN으로 설정합니다.")
        except Exception as e:
            retrieved_data[name] = {"value": np.nan, "change": np.nan}
            # print(f"ERROR: {name} ({ticker_symbol}) 다운로드 실패. 이유: {e}. 건너뜁니다.")
    
    # 장단기 금리차 계산 (10년물 - 3개월물)
    us10y_val = retrieved_data.get("US10Y", {}).get("value", np.nan)
    us3m_val = retrieved_data.get("US3M", {}).get("value", np.nan)
    
    if not np.isnan(us10y_val) and not np.isnan(us3m_val):
        retrieved_data["Yield_Spread_10Y_3M"] = {"value": us10y_val - us3m_val, "change": np.nan} # 금리차는 일일 변화 계산 안함
    else:
        retrieved_data["Yield_Spread_10Y_3M"] = {"value": np.nan, "change": np.nan}

    return retrieved_data

def macro_filter(macro_data):
    """거시경제 지표에 따라 시장 상태를 분류합니다."""
    vix_val = macro_data.get("VIX", {}).get("value", np.nan)
    us10y_val = macro_data.get("US10Y", {}).get("value", np.nan)

    # 데이터가 하나라도 NaN이면 '데이터 부족' 반환
    if np.isnan(vix_val) or np.isnan(us10y_val):
        return "데이터 부족"

    # 세분화된 시장 상태 분류
    if vix_val < 18:  # 낮은 변동성
        if us10y_val < 3.5:
            return "강세 (저변동)"
        elif us10y_val < 4.0:
            return "강세 (중변동)"
        else:
            return "강세 (고금리)"
    elif 18 <= vix_val <= 25: # 보통 변동성
        if us10y_val < 4.0:
            return "중립 (저변동)"
        else:
            return "중립 (고금리)"
    else: # vix_val > 25 (높은 변동성)
        if us10y_val < 4.0:
            return "약세 (고변동)"
        else:
            return "약세 (극변동)"

def soften_signal(signal, market_condition):
    """시장 상황에 따라 시그널을 완화합니다. (강력 매수/매도는 제외)"""
    if signal in ["강력 매수", "강력 매도"]:
        return signal

    # 시장 상태에 맞춰 조건 변경
    if market_condition in ["약세 (극변동)", "약세 (고변동)", "고금리", "중립 (고금리)"]:
        if "매수" in signal:
            return "관망"
        if "매도" in signal:
            return signal
    return signal

def adjust_score(score, market_condition):
    """시장 상황에 따라 추천 점수를 조정합니다."""
    # 시장 상태에 맞춰 조건 변경
    if market_condition in ["약세 (극변동)", "약세 (고변동)"]:
        return max(0, score * 0.7)
    elif market_condition in ["고금리", "중립 (고금리)"]:
        return max(0, score * 0.8)
    elif market_condition in ["강세 (저변동)", "강세 (중변동)", "강세 (고금리)"]:
        return min(100, score * 1.1)
    return score

# --- 기술적 지표 계산 함수 ---
def calc_indicators(df):
    """주어진 주식 데이터프레임에 기술적 지표를 계산하여 추가합니다."""
    if df.empty:
        return df

    # 이동평균선 (Moving Averages)
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
    df['%K'] = ((df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])) * 100
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

    # 거래량 이동평균 (Volume Average)
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # 볼린저 밴드 스퀴즈 돌파 플래그
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

    # 지표 계산에 필요한 최소 데이터 확인
    if len(df) < MIN_DATA_REQUIRED_FOR_INDICATORS:
        return pd.DataFrame()

    # 지표 계산 초반에 생기는 NaN 값 제거
    df = df.dropna(subset=['MA20', 'MACD', 'RSI', 'BB_Middle', 'ATR', 'ADX', 'Volume_MA20', '%K', '%D', 'CCI', 'BB_Squeeze_Up_Breakout', 'BB_Squeeze_Down_Breakout'])
    return df

# --- 매수/매도 시그널 감지 함수들 ---

def is_macd_golden_cross(prev_row, current_row):
    """MACD 골든 크로스 발생 여부를 확인합니다."""
    return prev_row['MACD'] < prev_row['Signal'] and current_row['MACD'] > current_row['Signal']

def is_macd_dead_cross(prev_row, current_row):
    """MACD 데드 크로스 발생 여부를 확인합니다."""
    return prev_row['MACD'] > prev_row['Signal'] and current_row['MACD'] < current_row['Signal']

def is_rsi_oversold(current_row):
    """RSI가 과매도 구간에 있는지 확인합니다."""
    return current_row['RSI'] < 35

def is_rsi_overbought(current_row):
    """RSI가 과매수 구간에 있는지 확인합니다."""
    return current_row['RSI'] > 65

def is_ma_cross_up(prev_row, current_row):
    """종가가 이동평균선을 상향 돌파했는지 확인합니다."""
    return prev_row['Close'] < prev_row['MA20'] and current_row['Close'] > current_row['MA20']

def is_ma_cross_down(prev_row, current_row):
    """종가가 이동평균선을 하향 돌파했는지 확인합니다."""
    return prev_row['Close'] > prev_row['MA20'] and current_row['Close'] < current_row['MA20']

def is_volume_surge(current_row):
    """거래량이 급증했는지 확인합니다."""
    return current_row['Volume'] > (current_row['Volume_MA20'] * 1.5)

def is_bullish_divergence(prev2_row, prev_row, current_row):
    """강세 다이버전스 발생 여부를 확인합니다. (가격은 하락, RSI는 상승)"""
    # Ensure all rows are valid and not NaN for the relevant columns
    if any(pd.isna(r[col]) for r in [prev2_row, prev_row, current_row] for col in ['Low', 'RSI']):
        return False

    price_low_decreasing = current_row['Low'] < prev_row['Low'] and prev_row['Low'] < prev2_row['Low']
    rsi_low_increasing = current_row['RSI'] > prev_row['RSI'] and prev_row['RSI'] > prev2_row['RSI']
    # RSI가 과매도권 근처에서 발생 시 신뢰도 증가
    return price_low_decreasing and rsi_low_increasing and current_row['RSI'] < 50

def is_bearish_divergence(prev2_row, prev_row, current_row):
    """약세 다이버전스 발생 여부를 확인합니다. (가격은 상승, RSI는 하락)"""
    # Ensure all rows are valid and not NaN for the relevant columns
    if any(pd.isna(r[col]) for r in [prev2_row, prev_row, current_row] for col in ['High', 'RSI']):
        return False

    price_high_increasing = current_row['High'] > prev_row['High'] and prev_row['High'] > prev2_row['High']
    rsi_high_decreasing = current_row['RSI'] < prev_row['RSI'] and prev_row['RSI'] < prev2_row['RSI']
    # RSI가 과매수권 근처에서 발생 시 신뢰도 증가
    return price_high_increasing and rsi_high_decreasing and current_row['RSI'] > 50

def is_hammer_candlestick(current_row, prev_row):
    """망치형 캔들스틱 발생 여부를 확인합니다. (하락 추세에서 반전 신호)"""
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

    # 조건 강화: 몸통이 전체 길이의 특정 비율 미만, 아래 꼬리가 몸통의 특정 배수 이상, 위 꼬리가 매우 짧음
    is_small_body = body <= 0.3 * total_range
    has_long_lower_shadow = lower_shadow >= 2 * body
    has_small_upper_shadow = upper_shadow <= 0.1 * body

    return is_small_body and has_long_lower_shadow and has_small_upper_shadow

def is_stoch_oversold(current_row):
    """스토캐스틱이 과매도 구간에 있는지 확인합니다."""
    return current_row['%K'] < 20 and current_row['%D'] < 20

def is_stoch_overbought(current_row):
    """스토캐스틱이 과매수 구간에 있는지 확인합니다."""
    return current_row['%K'] > 80 and current_row['%D'] > 80

def is_stoch_golden_cross(prev_row, current_row):
    """스토캐스틱 골든 크로스 발생 여부를 확인합니다."""
    return prev_row['%K'] < prev_row['%D'] and current_row['%K'] > current_row['%D'] and current_row['%K'] < 80

def is_stoch_dead_cross(prev_row, current_row):
    """스토캐스틱 데드 크로스 발생 여부를 확인합니다."""
    return prev_row['%K'] > prev_row['%D'] and current_row['%K'] < current_row['%D'] and current_row['%K'] > 20

def is_cci_oversold(current_row):
    """CCI가 과매도 구간에 있는지 확인합니다."""
    return current_row['CCI'] < -100

def is_cci_overbought(current_row):
    """CCI가 과매수 구간에 있는지 확인합니다."""
    return current_row['CCI'] > 100

# --- 스마트 시그널 로직 ---
def smart_signal_row(row, prev_row, prev2_row):
    """개별 행에 대한 스마트 매매 시그널을 생성합니다."""
    # 필수 지표 확인 (NaN 값 처리)
    required_indicators = ['MACD', 'Signal', 'MACD_Hist', 'RSI', 'MA20', 'Volume_MA20', 'ATR', 'ADX', '%K', '%D', 'CCI', 'BB_Squeeze_Up_Breakout', 'BB_Squeeze_Down_Breakout']
    if any(pd.isna(row[ind]) for ind in required_indicators):
        return "관망"

    current_close = row['Close']
    prev_close = prev_row['Close']
    macd_hist_direction = row['MACD_Hist'] - prev_row['MACD_Hist']

    # 1. 강력 매수/매도 시그널 (복합 지표, 최우선)
    # 강력 매수: MACD 골든크로스 + MA20 상향 돌파 + 거래량 급증 + 추세 강세 (ADX > 25, +DI > -DI) + 스토캐스틱 골든크로스 + RSI 과매도 탈출
    if (is_macd_golden_cross(prev_row, row) and
        is_ma_cross_up(prev_row, row) and
        is_volume_surge(row) and
        row['ADX'] > 25 and row['+DI14'] > row['-DI14'] and
        is_stoch_golden_cross(prev_row, row) and
        prev_row['RSI'] <= 30 and row['RSI'] > 30):
        return "강력 매수"

    # 강력 매도: MACD 데드크로스 + MA20 하향 돌파 + 거래량 급증 + 추세 약세 (ADX > 25, +DI < -DI) + 스토캐스틱 데드크로스 + RSI 과매수 탈출
    if (is_macd_dead_cross(prev_row, row) and
        is_ma_cross_down(prev_row, row) and
        is_volume_surge(row) and
        row['ADX'] > 25 and row['+DI14'] < row['-DI14'] and
        is_stoch_dead_cross(prev_row, row) and
        prev_row['RSI'] >= 70 and row['RSI'] < 70):
        return "강력 매도"

    # 2. 볼린저 밴드 스퀴즈 돌파 (강력한 추세 전환/시작)
    if row['BB_Squeeze_Up_Breakout']:
        return "강력 매수"
    if row['BB_Squeeze_Down_Breakout']:
        return "강력 매도"

    # 3. RSI 극단값 및 반전 시그널
    if row['RSI'] >= 80:
        return "익절 매도"
    if row['RSI'] <= 20:
        return "신규 매수"

    # 4. 스토캐스틱 극단값 및 반전 시그널
    if is_stoch_overbought(row) and is_stoch_dead_cross(prev_row, row):
        return "매도"
    if is_stoch_oversold(row) and is_stoch_golden_cross(prev_row, row):
        return "신규 매수"

    # 5. CCI 극단값 및 반전 시그널
    if is_cci_overbought(row) and row['CCI'] < prev_row['CCI']: # 과매수 후 하락 전환
        return "매도 고려"
    if is_cci_oversold(row) and row['CCI'] > prev_row['CCI']: # 과매도 후 상승 전환
        return "매수 고려"

    # 6. 모멘텀 변화 (MACD Hist)
    if row['MACD_Hist'] < 0 and macd_hist_direction > 0: # MACD 히스토그램 음수 구간에서 상승 전환 (매수 모멘텀 강화)
        return "매수 고려"
    if row['MACD_Hist'] > 0 and macd_hist_direction < 0: # MACD 히스토그램 양수 구간에서 하락 전환 (매도 모멘텀 강화)
        return "매도 고려"

    # 7. 일반적인 매수/매도 시그널
    if is_macd_golden_cross(prev_row, row):
        return "신규 매수"
    if is_macd_dead_cross(prev_row, row):
        return "매도"

    if is_ma_cross_up(prev_row, row):
        return "신규 매수"
    if is_ma_cross_down(prev_row, row):
        return "매도"

    # 8. 보조 시그널
    # 다이버전스 감지는 3개 봉 필요, prev2_row가 유효한지 확인
    if prev2_row is not None:
        if is_bullish_divergence(prev2_row, prev_row, row):
            return "반등 가능성"
        if is_bearish_divergence(prev2_row, prev_row, row):
            return "하락 가능성"
    
    if is_hammer_candlestick(row, prev_row): # 해머 캔들스틱 (반전)
        return "반전 신호"
    
    # 볼린저 밴드 터치/돌파 (스퀴즈 아닌 경우)
    if current_close > row['BB_Upper'] and prev_close <= prev_row['BB_Upper']: # 볼린저밴드 상단 돌파
        if row['RSI'] > 70: return "익절 매도"
        else: return "보유"
    if current_close < row['BB_Lower'] and prev_close >= prev_row['BB_Lower']: # 볼린저밴드 하단 돌파
        if row['RSI'] < 30: return "신규 매수"
        else: return "관망"

    # 그 외
    return "관망"

def smart_signal(df_with_signals):
    """데이터프레임에서 가장 최근의 유효한 매매 시그널을 반환합니다."""
    if df_with_signals.empty:
        return "데이터 부족"
    # 가장 최근의 유효한 시그널 반환
    last_valid_signal = df_with_signals['TradeSignal'].iloc[-1]
    return last_valid_signal

# --- 추천 정도 계산 ---
def compute_recommendation_score(last, prev_row, per, market_cap, forward_pe, debt_to_equity):
    """주어진 지표와 시장 상황에 따라 종목의 추천 점수를 계산합니다."""
    score = 50 # 기본 점수

    # MACD
    if last['MACD'] > last['Signal']: # MACD 골든 크로스 (추세 상승)
        score += 15
        if last['MACD_Hist'] > 0 and (last['MACD_Hist'] - prev_row['MACD_Hist']) > 0: # MACD 히스토그램 양수, 증가 (강한 상승 모멘텀)
            score += 10
    else: # MACD 데드 크로스 (추세 하락)
        score -= 15
        if last['MACD_Hist'] < 0 and (last['MACD_Hist'] - prev_row['MACD_Hist']) < 0: # MACD 히스토그램 음수, 감소 (강한 하락 모멘텀)
            score -= 10

    # RSI
    if last['RSI'] > 70: # 과매수 극단
        score -= 25
    elif last['RSI'] < 30: # 과매도 극단
        score += 25
    elif last['RSI'] > 60: # 과매수 (주의)
        score -= 10
    elif last['RSI'] < 40: # 과매도 (기회)
        score += 10
    elif last['RSI'] > 50: # 강세 영역
        score += 5
    elif last['RSI'] < 50: # 약세 영역
        score -= 5

    # Stochastic
    if last['%K'] > last['%D'] and last['%K'] < 80: # 골든 크로스 및 과매수 아님
        score += 10
    elif last['%K'] < last['%D'] and last['%K'] > 20: # 데드 크로스 및 과매도 아님
        score -= 10
    if is_stoch_oversold(last):
        score += 15
    if is_stoch_overbought(last):
        score -= 15

    # CCI
    if last['CCI'] > 100: # 과매수
        score -= 10
    elif last['CCI'] < -100: # 과매도
        score += 10
    
    # 이동평균선
    if last['Close'] > last['MA20'] and last['MA20'] > last['MA60'] and last['MA60'] > last['MA120']: # 완벽한 정배열
        score += 15
    elif last['Close'] < last['MA20'] and last['MA20'] < last['MA60'] and last['MA60'] < last['MA120']: # 완벽한 역배열
        score -= 15
    elif last['Close'] > last['MA20'] and last['MA20'] > last['MA60']: # 정배열
        score += 10
    elif last['Close'] < last['MA20'] and last['MA20'] < last['MA60']: # 역배열
        score -= 10

    # ADX (추세 강도 및 방향)
    if last['ADX'] > 30: # 강한 추세
        if last['+DI14'] > last['-DI14']: # 상승 추세
            score += 10
        else: # 하락 추세
            score -= 10
    elif last['ADX'] > 20: # 보통 추세
        if last['+DI14'] > last['-DI14']: # 상승 추세
            score += 5
        else: # 하락 추세
            score -= 5
    elif last['ADX'] < 20: # 추세가 약할 때 (횡보)
        score -= 5

    # 거래량
    if last['Volume'] > last['Volume_MA20'] * 2: # 거래량 매우 급증 (시그널 신뢰도 크게 증가)
        score += 10
    elif last['Volume'] > last['Volume_MA20'] * 1.5: # 거래량 급증 (시그널 신뢰도 증가)
        score += 5
    elif last['Volume'] < last['Volume_MA20'] * 0.5: # 거래량 급감 (관심 감소 또는 추세 약화)
        score -= 5

    # PER (주가수익비율) 점수 추가
    if not np.isnan(per):
        if per > 0: # PER이 양수인 경우만 고려 (음수는 적자 기업)
            if per < 15: # 저평가 가능성
                score += 10
            elif per >= 15 and per < 25: # 적정 가치
                score += 5
            elif per >= 25 and per < 40: # 약간 고평가
                score -= 5
            else: # 고평가
                score -= 10
        else: # PER이 0 이하 (적자)인 경우
            score -= 15 # 큰 감점

    # 시가총액 (Market Cap) 점수 추가
    if not np.isnan(market_cap):
        # 대형주 (1000억 달러 이상)는 안정성 점수
        if market_cap >= 100_000_000_000:
            score += 5
        # 중형주 (100억 ~ 1000억 달러)는 중립
        elif market_cap >= 10_000_000_000 and market_cap < 100_000_000_000:
            pass # 중립
        # 소형주 (10억 ~ 100억 달러)는 변동성 고려
        elif market_cap >= 1_000_000_000 and market_cap < 10_000_000_000:
            score -= 2 # 약간의 감점 (변동성 높음)
        # 초소형주 (10억 달러 미만)는 위험성 고려
        else:
            score -= 5 # 더 큰 감점

    # 선행 PER (Forward PE) 점수 추가
    if not np.isnan(forward_pe) and not np.isnan(per):
        if forward_pe < per: # 선행 PER이 현재 PER보다 낮으면 긍정적 (성장 기대)
            score += 7
        elif forward_pe > per: # 선행 PER이 현재 PER보다 높으면 부정적 (성장 둔화 또는 고평가)
            score -= 7
        # 그 외는 중립

    # 부채비율 (Debt to Equity) 점수 추가 (값이 작을수록 좋음)
    if not np.isnan(debt_to_equity):
        if debt_to_equity < 0.5: # 매우 낮은 부채 (건전)
            score += 8
        elif debt_to_equity >= 0.5 and debt_to_equity < 1.0: # 적정 부채
            score += 4
        elif debt_to_equity >= 1.0 and debt_to_equity < 2.0: # 다소 높은 부채
            score -= 4
        else: # 매우 높은 부채 (위험)
            score -= 8

    # TradeSignal 자체에 큰 가중치 부여 (최종 시그널 반영)
    if "강력 매수" in last['TradeSignal']:
        score += 30
    elif "신규 매수" in last['TradeSignal'] or "매수 고려" in last['TradeSignal'] or "반등 가능성" in last['TradeSignal']:
        score += 15
    elif "강력 매도" in last['TradeSignal']:
        score -= 30
    elif "매도" in last['TradeSignal'] or "익절 매도" in last['TradeSignal'] or "매도 고려" in last['TradeSignal'] or "하락 가능성" in last['TradeSignal']:
        score -= 15
    elif "관망" in last['TradeSignal'] or "보유" in last['TradeSignal'] or "반전 신호" in last['TradeSignal']:
        # 관망/보유 시그널일 때 점수 범위 제한
        score = max(score, 40) if score > 50 else min(score, 60)

    # 볼린저 밴드 돌파/스퀴즈
    if last['BB_Squeeze_Up_Breakout']:
        score += 20
    if last['BB_Squeeze_Down_Breakout']:
        score -= 20
    
    # 점수 정규화 (0-100)
    score = max(0, min(100, score))
    return score

# --- 추천 행동 및 비율 결정 ---
def get_action_and_percentage_by_score(signal, score):
    """추천 점수와 시그널에 따라 구체적인 행동과 비율을 결정합니다."""
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
    
    # 행동 텍스트 세부화
    if "매수" in action_base:
        action_text = f"포트폴리오 분배 자산의 {rounded_pct}% 매수"
    elif "매도" in action_base or "익절 매도" in action_base:
        action_text = f"보유분의 {rounded_pct}% 매도"
    elif action_base == "전량 매도":
        action_text = f"보유분의 {rounded_pct}% 전량 매도"
    else: # 관망
        action_text = action_base

    return action_text, rounded_pct

# --- 시그널을 시각적 기호로 변환 ---
def get_signal_symbol(signal_text):
    """시그널 텍스트에 해당하는 시각적 기호를 반환하고 색상을 적용합니다."""
    if "매수" in signal_text or "반등 가능성" in signal_text:
        return "<span style='color: green;'>▲</span>"  # 초록색 위쪽 화살표
    elif "매도" in signal_text or "하락 가능성" in signal_text or "익절 매도" in signal_text:
        return "<span style='color: red;'>▼</span>"  # 빨간색 아래쪽 화살표
    else:
        return "<span style='color: orange;'>●</span>"  # 주황색 원형 (중립/관망)

# --- 시그널 텍스트를 UI 표시용으로 변환 ---
def get_display_signal_text(signal_original, is_bb_squeeze_up=False, is_bb_squeeze_down=False):
    """원래 시그널 텍스트를 UI 표시를 위한 형태로 변환하고, BB 돌파 여부를 별도로 반환합니다."""
    display_text = signal_original
    if signal_original == "강력 매수":
        display_text = "강력 상승추세 가능성"
    elif signal_original == "강력 매도":
        display_text = "강력 하락추세 가능성"
    
    bb_indicator = ""
    if is_bb_squeeze_up:
        bb_indicator = " ↑(BB)" # Concise indicator
    elif is_bb_squeeze_down:
        bb_indicator = " ↓(BB)" # Concise indicator
        
    return display_text, bb_indicator

# --- ChatGPT 프롬프트 생성 ---
def generate_chatgpt_prompt(ticker, rsi, macd, macd_hist, signal_line, atr, adx, k_stoch, d_stoch, cci, per, market_cap, forward_pe, debt_to_equity):
    """ChatGPT에 보낼 기술적 지표 프롬프트 문자열을 생성합니다."""
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

# --- 이메일 전송 함수 ---
def send_email(subject, body, to_email, from_email, password, attachments=None):
    """이메일을 전송하는 함수입니다."""
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
                print(f"경고: 첨부 파일 {file_path}를 찾을 수 없습니다. 건너킵니다.")
            except Exception as e:
                print(f"경고: 첨부 파일 {file_path} 처리 중 오류 발생: {e}. 건너킵니다.")

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(from_email, password)
        server.send_message(msg)  
        server.quit()
        print("✅ 이메일 전송 성공!")
    except smtplib.SMTPAuthenticationError:
        print("❌ 이메일 전송 실패: 인증 오류. 발신자 이메일 또는 비밀번호(앱 비밀번호)를 확인하세요.")
    except Exception as e:
        print(f"❌ 이메일 전송 실패: {e}")

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    # 이메일 전송 모드인지 확인
    send_email_mode = "--send-email" in sys.argv

    # 환경 변수 또는 기본값 설정
    SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'parkib63@naver.com')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'bdnj dicf dzea wdrq') # Google 앱 비밀번호 사용 권장
    RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL', 'inbeom.park@samsung.com')
    STREAMLIT_APP_URL = os.getenv('STREAMLIT_APP_URL', 'https://app-stock-app-bomipark.streamlit.app/')

    if send_email_mode:
        print("🚀 이메일 보고서 전송 모드로 시작합니다...")
        if SENDER_EMAIL == 'parkib63@naver.com' or \
           SENDER_PASSWORD == 'bdnj dicf dzea wdrq' or \
           RECEIVER_EMAIL == 'inbeom.park@samsung.com' or \
           STREAMLIT_APP_URL == 'https://app-stock-app-bomipark.streamlit.app/':
            print("🚨 경고: 이메일 발신자 정보(SENDER_EMAIL, SENDER_PASSWORD) 또는 수신자 이메일(RECEIVER_EMAIL) 혹은 Streamlit 앱 URL(STREAMLIT_APP_URL)이 설정되지 않았습니다. 환경 변수를 확인하거나 코드 내 값을 변경하세요.")
            # sys.exit(1) # Streamlit 앱에서 실행될 경우 앱이 종료될 수 있으므로 주석 처리

        email_summary_rows = []
        email_tech_summaries_text = []

        macro_start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        macro_data = download_macro_data(macro_start_date, END_DATE)
        market_condition = macro_filter(macro_data)

        email_body_parts = []
        email_body_parts.append(f"<h1>📈 US Stock Signal Dashboard - {datetime.now().strftime('%Y-%m-%d')}</h1>")
        email_body_parts.append(f"<h2>마켓 센티멘트 현황</h2>")
        
        # 이메일용 거시경제 데이터
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
                                f"({macro_data.get('US3M', {}).get('change', np.nan):+.2f}%)"
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
                                "</tr></thead><tbody>")

        for ticker in TICKERS:
            try:
                # 주식 데이터 다운로드
                stock_data = yf.download(ticker, start=START_DATE, end=END_DATE)
                if stock_data.empty:
                    print(f"경고: {ticker} 주식 데이터를 찾을 수 없습니다. 건너킵니다.")
                    continue

                # 기술적 지표 계산
                df_with_indicators = calc_indicators(stock_data.copy())
                if df_with_indicators.empty:
                    print(f"경고: {ticker} 지표 계산에 필요한 데이터가 부족합니다. 건너킵니다.")
                    continue

                # 스마트 시그널 적용
                df_with_indicators['TradeSignal'] = "관망"
                for i in range(2, len(df_with_indicators)): # 최소 3개 봉 필요 (현재, 이전, 이전2)
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

                # Yahoo Finance에서 추가 정보 가져오기 (PER, 시가총액, 선행PER, 부채비율)
                ticker_info = yf.Ticker(ticker).info
                per = ticker_info.get('trailingPE', np.nan)
                market_cap = ticker_info.get('marketCap', np.nan)
                forward_pe = ticker_info.get('forwardPE', np.nan)
                debt_to_equity = ticker_info.get('debtToEquity', np.nan) # 부채비율 (Debt/Equity)

                # 시그널 완화 및 점수 조정
                original_signal = smart_signal(df_with_indicators)
                softened_signal = soften_signal(original_signal, market_condition)
                
                recommendation_score = compute_recommendation_score(last_row, prev_row_for_score, per, market_cap, forward_pe, debt_to_equity)
                adjusted_recommendation_score = adjust_score(recommendation_score, market_condition)

                action_text, percentage_value = get_action_and_percentage_by_score(softened_signal, adjusted_recommendation_score)
                
                # BB 돌파 여부 확인 및 시그널 텍스트에 추가
                is_bb_up_breakout = last_row['BB_Squeeze_Up_Breakout']
                is_bb_down_breakout = last_row['BB_Squeeze_Down_Breakout']
                display_signal_text_base, bb_indicator_text = get_display_signal_text(softened_signal, is_bb_up_breakout, is_bb_down_breakout)

                # 이메일 요약 테이블 행 추가
                change_color = 'green' if daily_change >= 0 else 'red'
                email_summary_rows.append(f"<tr>"
                                          f"<td>{ticker}</td>"
                                          f"<td>{TICKER_DESCRIPTIONS.get(ticker, 'N/A')}</td>"
                                          f"<td>{current_price:.2f}</td>"
                                          f"<td style='color: {change_color};'>{daily_change:+.2f} ({daily_change_pct:+.2f}%)</td>"
                                          f"<td>{get_signal_symbol(softened_signal)} {display_signal_text_base}{bb_indicator_text}</td>"
                                          f"<td>{int(adjusted_recommendation_score)}</td>"
                                          f"<td>{action_text}</td>"
                                          f"</tr>")
                
                # ChatGPT 프롬프트용 기술적 지표 요약
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
                print(f"ERROR: {ticker} 처리 중 오류 발생: {e}. 건너킵니다.")
                email_summary_rows.append(f"<tr><td>{ticker}</td><td colspan='6'>데이터 처리 오류: {e}</td></tr>")
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

    else: # Streamlit 앱 실행 모드
        st.set_page_config(layout="wide", page_title="US 주식 시그널 대시보드")

        st.title("📈 US 주식 시그널 대시보드")
        st.markdown("미국 주식 시장의 주요 기술적 지표와 거시경제 데이터를 기반으로 매매 시그널 및 추천 점수를 제공합니다.")

        # --- 거시경제 지표 섹션 ---
        st.subheader("📊 마켓 센티멘트 현황")
        
        # 캐시된 거시경제 데이터 로드
        macro_start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d") # 거시경제 데이터는 2개월치만
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

        # --- 종목 선택 섹션 ---
        st.sidebar.header("종목 선택")
        selected_ticker = st.sidebar.selectbox("티커를 선택하세요:", TICKERS, format_func=lambda x: f"{x} - {TICKER_DESCRIPTIONS.get(x, '')}")

        st.subheader(f"📈 {selected_ticker} - {TICKER_DESCRIPTIONS.get(selected_ticker, '')} 분석")

        # 주식 데이터 다운로드
        @st.cache_data
        def download_stock_data(ticker, start, end):
            try:
                data = yf.download(ticker, start=start, end=end)
                return data
            except Exception as e:
                st.error(f"'{ticker}' 주식 데이터를 다운로드하는 데 실패했습니다: {e}")
                return pd.DataFrame()

        stock_data = download_stock_data(selected_ticker, START_DATE, END_DATE)

        if not stock_data.empty:
            # 기술적 지표 계산
            df_with_indicators = calc_indicators(stock_data.copy())

            if not df_with_indicators.empty:
                # 스마트 시그널 적용
                df_with_indicators['TradeSignal'] = "관망"
                for i in range(2, len(df_with_indicators)): # 최소 3개 봉 필요 (현재, 이전, 이전2)
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

                # Yahoo Finance에서 추가 정보 가져오기 (PER, 시가총액, 선행PER, 부채비율)
                ticker_info = yf.Ticker(selected_ticker).info
                per = ticker_info.get('trailingPE', np.nan)
                market_cap = ticker_info.get('marketCap', np.nan)
                forward_pe = ticker_info.get('forwardPE', np.nan)
                debt_to_equity = ticker_info.get('debtToEquity', np.nan) # 부채비율 (Debt/Equity)

                # 시그널 완화 및 점수 조정
                original_signal = smart_signal(df_with_indicators)
                softened_signal = soften_signal(original_signal, market_condition)
                
                recommendation_score = compute_recommendation_score(last_row, prev_row_for_score, per, market_cap, forward_pe, debt_to_equity)
                adjusted_recommendation_score = adjust_score(recommendation_score, market_condition)

                action_text, percentage_value = get_action_and_percentage_by_score(softened_signal, adjusted_recommendation_score)

                # BB 돌파 여부 확인 및 시그널 텍스트에 추가
                is_bb_up_breakout = last_row['BB_Squeeze_Up_Breakout']
                is_bb_down_breakout = last_row['BB_Squeeze_Down_Breakout']
                display_signal_text_base, bb_indicator_text = get_display_signal_text(softened_signal, is_bb_up_breakout, is_bb_down_breakout)

                st.write(f"**현재가:** ${current_price:.2f}")
                change_color = "green" if daily_change >= 0 else "red"
                st.markdown(f"**일일 변동:** <span style='color:{change_color}'>{daily_change:+.2f} ({daily_change_pct:+.2f}%)</span>", unsafe_allow_html=True)
                st.markdown(f"**최근 추천 시그널:** {get_signal_symbol(softened_signal)} {display_signal_text_base}{bb_indicator_text}", unsafe_allow_html=True)
                st.write(f"**추천 점수 (0-100):** {int(adjusted_recommendation_score)}")
                st.write(f"**권장 행동:** {action_text}")

                st.markdown("---")
                st.subheader("📈 캔들스틱 차트 및 기술적 지표")

                # 캔들스틱 차트
                fig = go.Figure(data=[go.Candlestick(x=df_with_indicators.index,
                                                    open=df_with_indicators['Open'],
                                                    high=df_with_indicators['High'],
                                                    low=df_with_indicators['Low'],
                                                    close=df_with_indicators['Close'],
                                                    name='Candlestick')])
                # 이동평균선 추가
                fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA20'], mode='lines', name='MA20', line=dict(color='blue', width=1)))
                fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA60'], mode='lines', name='MA60', line=dict(color='orange', width=1)))
                fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MA120'], mode='lines', name='MA120', line=dict(color='purple', width=1)))

                # 볼린저 밴드 추가
                fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dash')))
                fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BB_Middle'], mode='lines', name='BB Middle', line=dict(color='gray', width=1)))
                fig.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash')))

                fig.update_layout(xaxis_rangeslider_visible=False, title=f'{selected_ticker} 주가 차트', height=600)
                st.plotly_chart(fig, use_container_width=True)

                # MACD 차트
                st.subheader("MACD")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')))
                fig_macd.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['Signal'], mode='lines', name='Signal Line', line=dict(color='red')))
                fig_macd.add_trace(go.Bar(x=df_with_indicators.index, y=df_with_indicators['MACD_Hist'], name='MACD Histogram', marker_color='green'))
                fig_macd.update_layout(height=300)
                st.plotly_chart(fig_macd, use_container_width=True)

                # RSI 차트
                st.subheader("RSI")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="과매도")
                fig_rsi.update_layout(height=300)
                st.plotly_chart(fig_rsi, use_container_width=True)

                # Stochastic 차트
                st.subheader("Stochastic Oscillator")
                fig_stoch = go.Figure()
                fig_stoch.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['%K'], mode='lines', name='%K', line=dict(color='blue')))
                fig_stoch.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['%D'], mode='lines', name='%D', line=dict(color='red')))
                fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="과매수")
                fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="과매도")
                fig_stoch.update_layout(height=300)
                st.plotly_chart(fig_stoch, use_container_width=True)

                # CCI 차트
                st.subheader("Commodity Channel Index (CCI)")
                fig_cci = go.Figure()
                fig_cci.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['CCI'], mode='lines', name='CCI', line=dict(color='teal')))
                fig_cci.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="과매수")
                fig_cci.add_hline(y=-100, line_dash="dash", line_color="green", annotation_text="과매도")
                fig_cci.update_layout(height=300)
                st.plotly_chart(fig_cci, use_container_width=True)

                # ADX 차트
                st.subheader("Average Directional Index (ADX)")
                fig_adx = go.Figure()
                fig_adx.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['ADX'], mode='lines', name='ADX', line=dict(color='black')))
                fig_adx.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['+DI14'], mode='lines', name='+DI14', line=dict(color='green')))
                fig_adx.add_trace(go.Scatter(x=df_with_indicators.index, y=df_with_indicators['-DI14'], mode='lines', name='-DI14', line=dict(color='red')))
                fig_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="강한 추세")
                fig_adx.update_layout(height=300)
                st.plotly_chart(fig_adx, use_container_width=True)

                st.markdown("---")
                st.subheader("📋 주요 지표 요약 (최근 값)")
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.write(f"**MACD:** {last_row['MACD']:.2f}")
                    st.write(f"**MACD Signal:** {last_row['Signal']:.2f}")
                    st.write(f"**MACD Hist:** {last_row['MACD_Hist']:.2f}")
                    st.write(f"**RSI:** {last_row['RSI']:.2f}")
                    st.write(f"**Stoch %K:** {last_row['%K']:.2f}")
                with col_sum2:
                    st.write(f"**Stoch %D:** {last_row['%D']:.2f}")
                    st.write(f"**CCI:** {last_row['CCI']:.2f}")
                    st.write(f"**ADX:** {last_row['ADX']:.2f}")
                    st.write(f"**+DI14:** {last_row['+DI14']:.2f}")
                    st.write(f"**-DI14:** {last_row['-DI14']:.2f}")
                with col_sum3:
                    st.write(f"**MA20:** {last_row['MA20']:.2f}")
                    st.write(f"**MA60:** {last_row['MA60']:.2f}")
                    st.write(f"**MA120:** {last_row['MA120']:.2f}")
                    st.write(f"**PER:** {per:.2f}" if not np.isnan(per) else "**PER:** N/A")
                    st.write(f"**시가총액:** {market_cap/1_000_000_000:.2f}B" if not np.isnan(market_cap) else "**시가총액:** N/A")
                    st.write(f"**선행PER:** {forward_pe:.2f}" if not np.isnan(forward_pe) else "**선행PER:** N/A")
                    st.write(f"**부채비율:** {debt_to_equity:.2f}" if not np.isnan(debt_to_equity) else "**부채비율:** N/A")

                # ChatGPT 프롬프트 생성 및 표시
                st.markdown("---")
                st.subheader("🤖 ChatGPT 분석용 프롬프트")
                chatgpt_prompt = generate_chatgpt_prompt(
                    selected_ticker,
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
                st.code(chatgpt_prompt, language='text')
                st.info("위 텍스트를 복사하여 ChatGPT에 붙여넣어 추가 분석을 요청할 수 있습니다.")

            else:
                st.warning("선택된 종목의 기술적 지표를 계산할 충분한 데이터가 없습니다. 다른 종목을 선택하거나 기간을 조정해 보세요.")
        else:
            st.info("선택된 종목의 데이터를 불러올 수 없습니다. 티커를 확인해 주세요.")

