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
    "MSFT",     # 마이크로소프트 (기술/성장, 소프트웨어, 클라우드)
    "AAPL",     # 애플 (기술/성장, 하드웨어, 서비스)
    "NVDA",     # 엔비디아 (기술/초고성장, 반도체, AI)
    "GOOGL",    # 알파벳 (기술/성장, 인터넷 서비스)
    "AMZN",     # 아마존 (기술/성장, 이커머스, 클라우드)
    "TSLA",     # 테슬라 (기술/성장, 전기차, AI)
    "PANW",     # 팔로알토 네트웍스 (기술/성장, AI 보안)
    "AMD",      # AMD (기술/성장, 반도체)
    "TSM",      # TSMC (반도체 파운드리)
    "ORCL",     # 오라클 (소프트웨어, 클라우드)
    "ADBE",     # 어도비 (기술/성장, 소프트웨어)
    "LLY",      # 일라이 릴리 (헬스케어/성장, 제약)
    "UNH",      # 유나이티드헬스그룹 (헬스케어/성장, 관리형 건강 서비스)
    "VRTX",     # 버텍스 파마슈티컬스 (바이오/성장, 제약)
    "REGN",     # 리제네론 파마슈티컬스 (바이오/성장, 제약)
    "JPM",      # JP모건 체이스 (금융/가치, 은행)
    "V",        # 비자 (기술/성장, 결제 서비스)
    "MS",       # 모건 스탠리 (금융)
    "JNJ",      # 존슨앤존슨 (헬스케어/가치, 필수 소비재, 배당)
    "HOOD",     # 로빈후드 (핀테크)
    "SPY",      # SPDR S&P 500 ETF (미국 대형주 시장 전체)
    "QQQ",      # Invesco QQQ Trust (나스닥 100 기술/성장주 중심)
    "SCHD",     # Schwab U.S. Dividend Equity ETF (미국 고배당주)
]

# 티커별 설명을 저장하는 딕셔너리
TICKER_DESCRIPTIONS = {
    "MSFT": "마이크로소프트 (기술/성장, 소프트웨어, 클라우드)",
    "AAPL": "애플 (기술/성장, 하드웨어, 서비스)",
    "NVDA": "엔비디아 (기술/초고성장, 반도체, AI)",
    "GOOGL": "알파벳 (기술/성장, 인터넷 서비스)",
    "AMZN": "아마존 (기술/성장, 이커머스, 클라우드)",
    "TSLA": "테슬라 (기술/성장, 전기차, AI)",
    "PANW": "팔로알토 네트웍스 (기술/성장, AI 보안)",
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
        "S&P500": "^GSPC",   # S&P 500 지수
        "NASDAQ": "^IXIC",   # Nasdaq Composite 지수
        "DXY": "DX-Y.NYB"    # 달러인덱스 (Yahoo Finance 티커)
    }
    retrieved_data = {}
    
    # 일일 변화 계산을 위해 최소 2일치 데이터 가져오기
    fetch_start_date = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d") # 충분한 데이터 확보

    for name, ticker_symbol in macro_tickers.items():
        try:
            print(f"DEBUG: {name} ({ticker_symbol}) 데이터 다운로드 시도 중...")
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
                print(f"DEBUG: {name} 데이터 성공적으로 가져옴: {current_value}, 변화: {change:.2f}")
            else:
                retrieved_data[name] = {"value": np.nan, "change": np.nan}
                print(f"DEBUG: {name} 데이터가 비어있거나 'Close' 컬럼이 NaN입니다. NaN으로 설정합니다.")
        except Exception as e:
            retrieved_data[name] = {"value": np.nan, "change": np.nan}
            print(f"ERROR: {name} ({ticker_symbol}) 다운로드 실패. 이유: {e}. 건너뜁니다.")
    
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

def is_bullish_divergence(prev_row, current_row, prev2_row):
    """강세 다이버전스 발생 여부를 확인합니다. (가격은 하락, RSI는 상승)"""
    price_low_decreasing = current_row['Low'] < prev_row['Low'] and prev_row['Low'] < prev2_row['Low']
    rsi_low_increasing = current_row['RSI'] > prev_row['RSI'] and prev_row['RSI'] > prev2_row['RSI']
    # RSI가 과매도권 근처에서 발생 시 신뢰도 증가
    return price_low_decreasing and rsi_low_increasing and current_row['RSI'] < 50

def is_bearish_divergence(prev_row, current_row, prev2_row):
    """약세 다이버전스 발생 여부를 확인합니다. (가격은 상승, RSI는 하락)"""
    price_high_increasing = current_row['High'] > prev_row['High'] and prev_row['High'] > prev2_row['High']
    rsi_high_decreasing = current_row['RSI'] < prev_row['RSI'] and prev_row['RSI'] < prev2_row['RSI']
    # RSI가 과매수권 근처에서 발생 시 신뢰도 증가
    return price_high_increasing and rsi_high_decreasing and current_row['RSI'] > 50

def is_hammer_candlestick(current_row, prev_row):
    """망치형 캔들스틱 발생 여부를 확인합니다. (하락 추세에서 반전 신호)"""
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
    if is_bullish_divergence(prev2_row, prev_row, row): # 다이버전스 감지는 3개 봉 필요
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
def get_display_signal_text(signal_original):
    """원래 시그널 텍스트를 UI 표시를 위한 형태로 변환합니다."""
    if signal_original == "강력 매수":
        return "강력 상승추세 가능성"
    elif signal_original == "강력 매도":
        return "강력 하락추세 가능성"
    return signal_original

# --- 추천 행동에 대한 확신 정도 점수 계산 ---
def get_conviction_score_for_display(signal, raw_score):
    """
    주어진 시그널과 원본 점수를 바탕으로 '추천 정도'를 계산합니다.
    매수 시그널일 때는 원본 점수를, 매도 시그널일 때는 원본 점수를 반전하여,
    해당 행동에 대한 '확신 정도'를 0-100 사이의 값으로 표현합니다.
    관망/보유 시그널일 경우, 원본 점수가 이미 해당 상태의 '질'을 반영하므로 그대로 사용합니다.
    """
    if "매수" in signal or "반등 가능성" in signal:
        # 매수 시그널: 점수가 높을수록 매수에 대한 확신이 높음
        return raw_score
    elif "매도" in signal or "익절 매도" in signal or "하락 가능성" in signal:
        # 매도 시그널: 점수가 낮을수록 매도에 대한 확신이 높으므로, 100에서 빼서 반전
        return 100 - raw_score
    else: # 관망, 보유, 반전 신호 등
        # 관망/보유 시그널: 원본 점수 자체가 해당 관망/보유의 '질'을 나타냄.
        # 예를 들어, 50이면 중립적 관망, 60이면 긍정적 보유, 40이면 부정적 관망.
        return raw_score

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
                print(f"경고: 첨부 파일 {file_path}를 찾을 수 없습니다. 건너뜁니다.")
            except Exception as e:
                print(f"경고: 첨부 파일 {file_path} 처리 중 오류 발생: {e}. 건너뜁니다.")

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

    SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'parkib63@naver.com')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'bdnj dicf dzea wdrq')
    RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL', 'inbeom.park@samsung.com')
    STREAMLIT_APP_URL = os.getenv('STREAMLIT_APP_URL', 'https://app-stock-app-bambam.streamlit.app/')


    if send_email_mode:
        print("🚀 이메일 보고서 전송 모드로 시작합니다...")
        if SENDER_EMAIL == 'censored' or \
           SENDER_PASSWORD == 'censored' or \
           RECEIVER_EMAIL == 'censored' or \
           STREAMLIT_APP_URL == 'https://app-stock-app-bomipark.streamlit.app/':
            print("🚨 경고: 이메일 발신자 정보(SENDER_EMAIL, SENDER_PASSWORD) 또는 수신자 이메일(RECEIVER_EMAIL) 혹은 Streamlit 앱 URL(STREAMLIT_APP_URL)이 설정되지 않았습니다. 환경 변수를 확인하거나 코드 내 값을 변경하세요.")
            sys.exit(1)


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
        email_body_parts.append(f"<p><b>자세한 분석 및 실시간 캔들스틱 차트를 보려면 아래 링크를 클릭하세요:</b></p>")
        email_body_parts.append(f"<p><a href='{STREAMLIT_APP_URL}'>👉 Streamlit 주식 시그널 대시보드 바로가기</a></p>")


        print(f"시장 상태: {market_condition}")

        for ticker in TICKERS:
            print(f"처리 중: {ticker}...")
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=START_DATE, end=END_DATE, interval="1d")

                if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    print(f"❌ {ticker} 데이터 누락 또는 형식 오류. 건너뜁니다.")
                    continue

                if 'Adj Close' in data.columns:
                    data = data.drop(columns=['Adj Close'])

                df = calc_indicators(data[['Open', 'High', 'Low', 'Close', 'Volume']].copy())

                if df.empty or len(df) < MIN_DATA_REQUIRED_FOR_INDICATORS:
                    print(f"❌ {ticker} 지표 계산 후 데이터 부족 ({len(df)}개). 시그널 생성을 건너뜁니다.")
                    continue

                df['TradeSignal'] = ["관망"] * len(df)
                for i in range(2, len(df)):
                    df.loc[df.index[i], 'TradeSignal'] = smart_signal_row(df.iloc[i], df.iloc[i-1], df.iloc[i-2])

                last = df.iloc[-1]
                prev_row = df.iloc[-2]

                # 기본 정보 가져오기
                info = ticker_obj.info
                per = info.get('trailingPE', np.nan)
                market_cap = info.get('marketCap', np.nan)
                forward_pe = info.get('forwardPE', np.nan)
                debt_to_equity = info.get('debtToEquity', np.nan)

                signal = last['TradeSignal']
                signal = soften_signal(signal, market_condition)
                df.loc[df.index[-1], 'TradeSignal'] = signal # 최종 시그널 업데이트

                score = compute_recommendation_score(last, prev_row, per, market_cap, forward_pe, debt_to_equity)
                score = adjust_score(score, market_condition) # 거시경제에 따른 점수 조정
                
                # '추천 정도'를 행동에 대한 확신 정도로 변환
                action, pct = get_action_and_percentage_by_score(signal, score)
                display_score = get_conviction_score_for_display(signal, score) # 수정된 함수 사용

                email_summary_rows.append({
                    "Ticker": ticker,
                    "Signal": signal,
                    "추천정도": f"{display_score:.1f}", # display_score 사용
                    "추천 행동": action,
                })

                rsi_val = float(last.get('RSI', np.nan))
                macd_val = float(last.get('MACD', np.nan))
                macd_hist_val = float(last.get('MACD_Hist', np.nan))
                signal_line_val = float(last.get('Signal', np.nan))
                atr_val = float(last.get('ATR', np.nan))
                adx_val = float(last.get('ADX', np.nan))
                k_stoch_val = float(last.get('%K', np.nan))
                d_stoch_val = float(last.get('%D', np.nan))
                cci_val = float(last.get('CCI', np.nan))


                email_tech_summaries_text.append(generate_chatgpt_prompt(ticker, rsi_val, macd_val, macd_hist_val, signal_line_val, atr_val, adx_val, k_stoch_val, d_stoch_val, cci_val, per, market_cap, forward_pe, debt_to_equity))

            except ValueError as ve:
                print(f"❌ {ticker} 지표 계산 중 오류 발생: {ve}. 건너뜁니다.")
            except Exception as e:
                print(f"❌ {ticker} 데이터 처리 중 알 수 없는 오류 발생: {e}. 건너뜁니다.")
                continue

        if email_summary_rows:
            email_body_parts.append("<h2>📋 오늘의 종목별 매매 전략 요약</h2>")
            email_body_parts.append(pd.DataFrame(email_summary_rows).to_html(index=False))

        if email_tech_summaries_text:
            ai_prompt_template = """
<br>
<h3>🧠 AI에게 물어보는 기술적 분석 프롬프트</h3>
<p>아래 각 종목의 기술적 지표만 보고, 
미국 주식 전문 트레이더처럼 매수/매도/익절/보유/관망 시그널과 
매수/매도/익절이 필요한 경우 “몇 % 정도” 하면 좋을지도 같이 구체적으로 알려줘.</p>
<p>- 한 종목당 한 줄씩,<br>- 신호와 추천 비율(%)만 간단명료하게<br>- 사유도 한 줄로 덧붙여줘.</p>
<p>아래 표로 정리해서 답변해줘.</p>
<pre><code>| 종목 | 추천액션 | 비율(%) | 근거 요약 |
|------|----------------|---------|-----------------------------|
"""
            full_ai_prompt_content = ai_prompt_template + "\n".join(email_tech_summaries_text) + "\n</code></pre>"
            email_body_parts.append(full_ai_prompt_content)

        final_email_body = "".join(email_body_parts)

        EMAIL_SUBJECT = f"US Stock Signal Dashboard - {datetime.now().strftime('%Y-%m-%d')}"
        email_attachments = []
        send_email(EMAIL_SUBJECT, final_email_body, RECEIVER_EMAIL, SENDER_EMAIL, SENDER_PASSWORD, attachments=email_attachments)


    else:
        # --- Streamlit 대시보드 UI ---
        st.set_page_config(layout="wide", page_title="US Stock Signal Dashboard")
        st.title("📈 US Stock Signal Dashboard")
        st.subheader(f"데이터 기준일: {END_DATE}")

        # 마켓 센티멘트
        st.markdown("---")
        st.subheader("마켓 센티멘트 현황")
        macro_start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        macro_data = download_macro_data(macro_start_date, END_DATE)
        market_condition = macro_filter(macro_data)

        # 컬럼 수 조정 및 순서 변경: VIX, 시장 상태, 미 10년 금리, 미 3개월 금리, S&P500, NASDAQ, 달러인덱스
        col_vix, col_market, col_us10y, col_us3m, col_sp500, col_nasdaq, col_dxy = st.columns(7) 
        
        def format_macro_metric(value_dict, unit=""):
            value = value_dict.get('value', np.nan)
            change = value_dict.get('change', np.nan)
            
            display_value = f"{value:.2f}{unit}" if not np.isnan(value) else "N/A"
            
            if not np.isnan(change):
                color = "green" if change >= 0 else "red"
                change_str = f"({change:+.2f}{unit})"
                return f"{display_value} <span style='font-size: 0.8em; color: {color};'>{change_str}</span>"
            return display_value

        with col_vix:
            st.markdown(f"**VIX**<br>{format_macro_metric(macro_data.get('VIX', {}))}", unsafe_allow_html=True)
        with col_market:
            st.markdown(f"**시장 상태**<br>{market_condition}", unsafe_allow_html=True)
        with col_us10y:
            st.markdown(f"**미 10년 금리**<br>{format_macro_metric(macro_data.get('US10Y', {}), '%')}", unsafe_allow_html=True)
        with col_us3m:
            st.markdown(f"**미 3개월 금리**<br>{format_macro_metric(macro_data.get('US3M', {}), '%')}", unsafe_allow_html=True)
        with col_sp500:
            st.markdown(f"**S&P 500**<br>{format_macro_metric(macro_data.get('S&P500', {}))}", unsafe_allow_html=True)
        with col_nasdaq:
            st.markdown(f"**NASDAQ**<br>{format_macro_metric(macro_data.get('NASDAQ', {}))}", unsafe_allow_html=True)
        with col_dxy:
            st.markdown(f"**달러인덱스 (DXY)**<br>{format_macro_metric(macro_data.get('DXY', {}))}", unsafe_allow_html=True)

        st.markdown("---")

        summary_rows = []
        all_tech_summaries_text = []
        
        all_ticker_data = {}
        # 모든 티커 데이터를 먼저 처리하여 all_ticker_data에 저장
        for ticker in TICKERS:
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=START_DATE, end=END_DATE, interval="1d")

                if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    st.warning(f"❌ **{ticker}** 데이터 누락 또는 형식 오류. 건너뜁니다.")
                    continue

                if 'Adj Close' in data.columns:
                    data = data.drop(columns=['Adj Close'])

                df = calc_indicators(data[['Open', 'High', 'Low', 'Close', 'Volume']].copy())

                if df.empty or len(df) < MIN_DATA_REQUIRED_FOR_INDICATORS:
                    st.warning(f"❌ **{ticker}** 지표 계산 후 데이터 부족 ({len(df)}개). 시그널 생성을 건너뛰었습니다.")
                    continue

                df['TradeSignal'] = ["관망"] * len(df)
                for i in range(2, len(df)):
                    df.loc[df.index[i], 'TradeSignal'] = smart_signal_row(df.iloc[i], df.iloc[i-1], df.iloc[i-2])

                last = df.iloc[-1]
                prev_row = df.iloc[-2]

                # 기본 정보 가져오기
                info = ticker_obj.info
                per = info.get('trailingPE', np.nan)
                market_cap = info.get('marketCap', np.nan)
                forward_pe = info.get('forwardPE', np.nan)
                debt_to_equity = info.get('debtToEquity', np.nan)

                signal = last['TradeSignal']
                signal = soften_signal(signal, market_condition) # 거시경제 필터링 적용
                df.loc[df.index[-1], 'TradeSignal'] = signal # 최종 시그널 업데이트

                score = compute_recommendation_score(last, prev_row, per, market_cap, forward_pe, debt_to_equity)
                score = adjust_score(score, market_condition) # 거시경제에 따른 점수 조정
                action, pct = get_action_and_percentage_by_score(signal, score)
                
                # '추천 정도'를 행동에 대한 확신 정도로 변환
                display_score = get_conviction_score_for_display(signal, score) # 수정된 함수 사용

                summary_rows.append({
                    "Ticker": ticker,
                    "Signal": signal,
                    "추천정도": f"{display_score:.1f}", # display_score 사용
                    "추천 행동": action,
                })
                all_ticker_data[ticker] = {
                    'df': df,
                    'last': last,
                    'prev_row': prev_row,
                    'signal': signal,
                    'score': score,
                    'action': action,
                    'pct': pct,
                    'per': per, # PER 저장
                    'market_cap': market_cap, # 시가총액 저장
                    'forward_pe': forward_pe, # 선행 PER 저장
                    'debt_to_equity': debt_to_equity # 부채비율 저장
                }

            except ValueError as ve:
                st.error(f"❌ **{ticker}** 지표 계산 중 오류 발생: **{ve}**")
                st.warning(f"**{ticker}** 시그널 생성을 건너뛰었습니다.")
            except Exception as e:
                st.error(f"❌ **{ticker}** 데이터 처리 중 알 수 없는 오류 발생: **{e}**")
                st.warning(f"**{ticker}** 시그널 생성을 건너뛰었습니다.")

        # --- 매수/매도/관망 종목 목록 표시 ---
        st.subheader("📊 전체 종목별 매매 시그널 현황")
        
        buy_tickers = []
        sell_tickers = []
        hold_tickers = []

        for ticker, data_for_ticker in all_ticker_data.items():
            signal = data_for_ticker['signal']
            last = data_for_ticker['last'] # last 행 데이터 가져오기

            # BB Squeeze Breakout 여부 확인
            is_bb_squeeze_breakout = last.get('BB_Squeeze_Up_Breakout', False) or last.get('BB_Squeeze_Down_Breakout', False)
            
            display_text = f"- {ticker} {get_signal_symbol(signal)} - {TICKER_DESCRIPTIONS.get(ticker, '설명 없음')}"
            if is_bb_squeeze_breakout:
                display_text += " ⭐" # BB Squeeze Breakout 발생 시 별 이모지 추가

            # 매수 또는 반등 가능성 시그널
            if "매수" in signal or "반등 가능성" in signal:
                buy_tickers.append(display_text)
            # 매도 또는 하락 가능성 또는 익절 매도 시그널
            elif "매도" in signal or "하락 가능성" in signal or "익절 매도" in signal:
                sell_tickers.append(display_text)
            # 그 외 시그널 (관망, 보유, 반전)
            else: 
                hold_tickers.append(display_text)

        total_stocks_processed = len(all_ticker_data)
        
        if total_stocks_processed > 0:
            col_buy, col_sell, col_hold = st.columns(3)
            with col_buy:
                st.markdown("#### ✅ 매수 시그널 종목")
                if buy_tickers:
                    for t_text in buy_tickers:
                        # get_signal_symbol 함수가 HTML을 반환하므로 unsafe_allow_html=True 설정
                        st.markdown(t_text, unsafe_allow_html=True)
                else:
                    st.write("없음")
            with col_sell:
                st.markdown("#### 🔻 매도 시그널 종목")
                if sell_tickers:
                    for t_text in sell_tickers:
                        # get_signal_symbol 함수가 HTML을 반환하므로 unsafe_allow_html=True 설정
                        st.markdown(t_text, unsafe_allow_html=True)
                else:
                    st.write("없음")
            with col_hold:
                st.markdown("#### 🟡 관망/보유 시그널 종목")
                if hold_tickers:
                    for t_text in hold_tickers:
                        # get_signal_symbol 함수가 HTML을 반환하므로 unsafe_allow_html=True 설정
                        st.markdown(t_text, unsafe_allow_html=True)
                else:
                    st.write("없음")
        else:
            st.info("시그널을 분석할 수 있는 종목 데이터가 부족합니다.")

        st.markdown("---")

        # --- 오늘의 종목별 매매 전략 요약 ---
        if summary_rows:
            st.subheader("📋 오늘의 종목별 매매 전략 요약")
            st.markdown("""
            - **'신규 매수'**: 포트폴리오 분배 자산의 권장 비율만큼 신규 진입을 고려합니다.
            - **'익절 매도'**: 보유 주식의 권장 비율만큼을 수익 실현을 위해 매도하는 것을 권장합니다.
            - **'매도'**: 보유 주식의 권장 비율만큼을 매도하여 리스크를 줄이는 것을 권장합니다.
            - **'전량 매도'**: 보유 주식 전체를 매도하여 포지션을 정리하는 것을 권장합니다.
            - **'관망'**: 현재 포지션을 유지하거나 시장의 추가적인 신호를 기다리는 것을 권장합니다.
            """)
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True)
            st.markdown("---")

        # --- 각 종목별 상세 지표 및 차트 ---
        for ticker in TICKERS:
            if ticker in all_ticker_data:
                data_for_ticker = all_ticker_data[ticker]
                df = data_for_ticker['df']
                last = data_for_ticker['last']
                signal = data_for_ticker['signal']
                action = data_for_ticker['action']
                pct = data_for_ticker['pct']
                score = data_for_ticker['score'] # 원본 score
                per = data_for_ticker['per'] # PER 가져오기
                market_cap = data_for_ticker['market_cap'] # 시가총액 가져오기
                forward_pe = data_for_ticker['forward_pe'] # 선행 PER 가져오기
                debt_to_equity = data_for_ticker['debt_to_equity'] # 부채비율 가져오기

                # '추천 정도'를 행동에 대한 확신 정도로 변환
                display_score = get_conviction_score_for_display(signal, score) # 수정된 함수 사용

                # 티커 설명 추가 (추천 행동 위에 위치)
                st.write(f"**{TICKER_DESCRIPTIONS.get(ticker, '설명 없음')}**")
                # 시그널 기호와 텍스트를 별도의 markdown으로 표시
                st.subheader(f"📊 {ticker} 시그널 (오늘 종가: **${last['Close']:.2f}**)")
                st.markdown(f"{get_signal_symbol(signal)} {get_display_signal_text(signal)}", unsafe_allow_html=True)
                
                st.write(f"**추천 행동**: **{action}**")
                st.write(f"**추천정도**: **{display_score:.1f}/100**") # display_score 사용

                st.markdown("---")
                st.subheader(f"{ticker} 최근 지표")
                
                # 추천 점수 아래에 기본 정보 별도 표시
                st.markdown(f"""
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
                        <div><strong>PER:</strong> {per:.2f}</div>
                        <div><strong>시가총액:</strong> {market_cap/1_000_000_000:.2f}B</div>
                        <div><strong>선행PER:</strong> {forward_pe:.2f}</div>
                        <div><strong>부채비율:</strong> {debt_to_equity:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)

                # 메인 데이터프레임 표시 컬럼 (기술적 지표만)
                display_cols = ['Close', 'MA20', 'MACD', 'Signal', 'MACD_Hist', 'RSI', 'ATR', 'ADX', '+DI14', '-DI14', 'Volume', 'Volume_MA20', 'TradeSignal', '%K', '%D', 'CCI', 'BB_Squeeze_Up_Breakout', 'BB_Squeeze_Down_Breakout']
                
                st.dataframe(df.tail(7)[display_cols])
                st.markdown("---")

                # 캔들스틱 차트
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']
                )])
                fig.update_layout(title=f'{ticker} 캔들스틱 차트', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                rsi_val = float(last.get('RSI', np.nan))
                macd_val = float(last.get('MACD', np.nan))
                macd_hist_val = float(last.get('MACD_Hist', np.nan))
                signal_line_val = float(last.get('Signal', np.nan))
                atr_val = float(last.get('ATR', np.nan))
                adx_val = float(last.get('ADX', np.nan))
                k_stoch_val = float(last.get('%K', np.nan))
                d_stoch_val = float(last.get('%D', np.nan))
                cci_val = float(last.get('CCI', np.nan))


                all_tech_summaries_text.append(generate_chatgpt_prompt(ticker, rsi_val, macd_val, macd_hist_val, signal_line_val, atr_val, adx_val, k_stoch_val, d_stoch_val, cci_val, per, market_cap, forward_pe, debt_to_equity))
            
        # --- AI 기술적 분석 프롬프트 ---
        if all_tech_summaries_text:
            st.subheader("🧠 AI에게 물어보는 기술적 분석 프롬프트")
            
            ai_prompt_template = """
아래 각 종목의 기술적 지표만 보고, 
미국 주식 전문 트레이더처럼 매수/매도/익절/보유/관망 시그널과 
매수/매도/익절이 필요한 경우 “몇 % 정도” 하면 좋을지도 같이 구체적으로 알려줘.

- 한 종목당 한 줄씩, 
- 신호와 추천 비율(%)만 간단명료하게 
- 사유도 한 줄로 덧붙여줘.

**[질문]**
- 각 종목별로 
  1) 매수/매도/익절/보유/관망 중 뭐가 적합한지 
  2) 추천 비율(%)은 얼마나 할지 (예: “익절 30%” “신규매수 50%” 등) 
  3) 근거 한 줄

아래 표로 정리해서 답변해줘.

| 종목 | 추천액션 | 비율(%) | 근거 요약 |
|------|----------------|---------|-----------------------------|
"""
            
            full_ai_prompt_content = ai_prompt_template + "\n" + "\n".join(all_tech_summaries_text)

            st.code(full_ai_prompt_content, language='markdown', line_numbers=False)
            st.markdown("👆 위 프롬프트 내용 옆 **'Copy to clipboard' 버튼**을 클릭하여 쉽게 복사할 수 있습니다.")
