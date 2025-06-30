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
    "TSLA",     # 테슬라 (기술/성장, 전기차, AI) - 추가
    "PANW",     # 팔로알토 (기술/성장, AI)
    "AMD",      # AMD (기술/성장, 반도체) - 추가
    "ASML",     # ASML (기술/성장, 반도체 장비) - 추가
    "CRM",      # 세일즈포스 (기술/성장, 클라우드 소프트웨어) - 추가
    "ADBE",     # 어도비 (기술/성장, 소프트웨어) - 추가
    "LLY",      # 일라이 릴리 (헬스케어/성장, 제약)
    "UNH",      # 유나이티드헬스그룹 (헬스케어/성장, 관리형 건강 서비스)
    "VRTX",     # 버텍스 파마슈티컬스 (바이오/성장, 제약) - 추가
    "REGN",     # 리제네론 파마슈티컬스 (바이오/성장, 제약) - 추가
    "JPM",      # JP모건 체이스 (금융/가치, 은행)
    "V",        # 비자 (기술/성장, 결제 서비스)
    "XOM",      # 엑손 모빌 (에너지/가치, 원유, 가스) - 유지
    "JNJ",      # 존슨앤존슨 (헬스케어/가치, 필수 소비재, 배당) - 유지 (헬스케어 측면)
    "SPY",      # SPDR S&P 500 ETF (미국 대형주 시장 전체)
    "QQQ",      # Invesco QQQ Trust (나스닥 100 기술/성장주 중심)
    "SCHD",     # Schwab U.S. Dividend Equity ETF (미국 고배당주)
]

END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d") # 약 2.5년치 데이터
MIN_DATA_REQUIRED_FOR_INDICATORS = 180 # 지표 계산에 필요한 최소 일봉 데이터 수

# --- 보조 함수들 (Helper Functions) ---

@st.cache_data
def download_macro_data(start, end):
    """VIX와 10년 국채 금리 데이터를 다운로드합니다."""
    macro_tickers = ["^VIX", "^TNX"] # TNX는 10년 국채 금리 티커
    data = yf.download(macro_tickers, start=start, end=end, progress=False)
    vix = data['Close']['^VIX'].iloc[-1] if '^VIX' in data['Close'].columns else np.nan
    us10y = data['Close']['^TNX'].iloc[-1] if '^TNX' in data['Close'].columns else np.nan
    return {"VIX": vix, "US10Y": us10y}

def macro_filter(macro_data):
    """거시경제 지표에 따라 시장 상태를 분류합니다."""
    vix_val = macro_data.get("VIX", np.nan)
    us10y_val = macro_data.get("US10Y", np.nan)

    if not np.isnan(vix_val) and vix_val > 25: # VIX 25 이상은 고변동성
        return "HIGH_VOLATILITY"
    if not np.isnan(us10y_val) and us10y_val > 4.5: # 10년물 4.5% 이상은 고금리
        return "HIGH_INTEREST_RATE"
    if not np.isnan(vix_val) and vix_val < 18 and not np.isnan(us10y_val) and us10y_val < 4.0: # VIX 18 미만, 10년물 4.0% 미만은 강세장
        return "BULLISH"
    return "NORMAL"

def soften_signal(signal, market_condition):
    """시장 상황에 따라 시그널을 완화합니다. (강력 매수/매도는 제외)"""
    if signal in ["강력 매수", "강력 매도"]:
        return signal # 강력 시그널은 거시경제 필터링을 받지 않음

    if market_condition in ["HIGH_VOLATILITY", "HIGH_INTEREST_RATE"]:
        if "매수" in signal:
            return "관망"
        if "매도" in signal: # 매도 시그널은 유지 또는 강화 (이미 매도면 그대로)
            return signal
    return signal

def adjust_score(score, market_condition):
    """시장 상황에 따라 추천 점수를 조정합니다."""
    if market_condition == "HIGH_VOLATILITY":
        return max(0, score * 0.7) # 변동성 높을 때는 점수 30% 하향
    elif market_condition == "HIGH_INTEREST_RATE":
        return max(0, score * 0.8) # 금리 높을 때는 점수 20% 하향
    elif market_condition == "BULLISH":
        return min(100, score * 1.1) # 강세장에서는 점수 10% 상향
    return score

# --- 기술적 지표 계산 함수 (calc_indicators) ---
def calc_indicators(df):
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

    # ADX (Average Directional Index) - TA-Lib 구현과 다를 수 있음
    # +DM, -DM
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)

    # TR (True Range for ADX) - ATR에서 이미 계산됨. 재사용
    df['TR_ADX'] = tr

    # +DI, -DI
    df['+DI14'] = (df['+DM'].rolling(window=14).sum() / df['TR_ADX'].rolling(window=14).sum()) * 100
    df['-DI14'] = (df['-DM'].rolling(window=14).sum() / df['TR_ADX'].rolling(window=14).sum()) * 100

    # DX
    df['DX'] = abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14']) * 100
    df['ADX'] = df['DX'].ewm(span=14, adjust=False).mean()

    # 거래량 이동평균 (Volume Average)
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()

    # 필요한 데이터가 충분히 있는지 확인
    if len(df) < MIN_DATA_REQUIRED_FOR_INDICATORS:
        return pd.DataFrame() # 데이터 부족 시 빈 DataFrame 반환

    # NaN 값 제거 (지표 계산 초반에 생기는 NaN)
    df = df.dropna(subset=['MA20', 'MACD', 'RSI', 'BB_Middle', 'ATR', 'ADX', 'Volume_MA20'])
    return df

# --- 매수/매도 시그널 감지 함수들 ---

def is_macd_golden_cross(prev_row, current_row):
    return prev_row['MACD'] < prev_row['Signal'] and current_row['MACD'] > current_row['Signal']

def is_macd_dead_cross(prev_row, current_row):
    return prev_row['MACD'] > prev_row['Signal'] and current_row['MACD'] < current_row['Signal']

def is_macd_hist_cross_up_zero(prev_row, current_row):
    return prev_row['MACD_Hist'] < 0 and current_row['MACD_Hist'] >= 0

def is_macd_hist_cross_down_zero(prev_row, current_row):
    return prev_row['MACD_Hist'] > 0 and current_row['MACD_Hist'] <= 0

def is_rsi_oversold(current_row):
    return current_row['RSI'] < 35 # 30에서 35로 완화

def is_rsi_overbought(current_row):
    return current_row['RSI'] > 65 # 70에서 65로 완화

def is_ma_cross_up(prev_row, current_row):
    return prev_row['Close'] < prev_row['MA20'] and current_row['Close'] > current_row['MA20']

def is_ma_cross_down(prev_row, current_row):
    return prev_row['Close'] > prev_row['MA20'] and current_row['Close'] < current_row['MA20']

def is_volume_surge(current_row):
    return current_row['Volume'] > (current_row['Volume_MA20'] * 1.5) # 1.3배에서 1.5배로 강화

def is_bullish_divergence(prev_row, current_row, prev2_row):
    # 가격은 하락 추세 (저점 낮아짐) but RSI는 상승 추세 (저점 높아짐)
    price_low_decreasing = current_row['Low'] < prev_row['Low'] and prev_row['Low'] < prev2_row['Low']
    rsi_low_increasing = current_row['RSI'] > prev_row['RSI'] and prev_row['RSI'] > prev2_row['RSI']
    # RSI가 과매도권 근처에서 발생 시 신뢰도 증가 (RSI 40 -> 50으로 변경)
    return price_low_decreasing and rsi_low_increasing and current_row['RSI'] < 50

def is_bearish_divergence(prev_row, current_row, prev2_row):
    # 가격은 상승 추세 (고점 높아짐) but RSI는 하락 추세 (고점 낮아짐)
    price_high_increasing = current_row['High'] > prev_row['High'] and prev_row['High'] > prev2_row['High']
    rsi_high_decreasing = current_row['RSI'] < prev_row['RSI'] and prev_row['RSI'] < prev2_row['RSI']
    # RSI가 과매수권 근처에서 발생 시 신뢰도 증가 (RSI 60 -> 50으로 변경)
    return price_high_increasing and rsi_high_decreasing and current_row['RSI'] > 50

def is_hammer_candlestick(current_row, prev_row):
    # 망치형 캔들스틱 (Hammer Candlestick)
    # 특징: 긴 아래 꼬리, 짧은 윗 꼬리, 작은 몸통 (하락 추세에서 반전 신호)
    open_price = current_row['Open']
    close_price = current_row['Close']
    high_price = current_row['High']
    low_price = current_row['Low']

    body = abs(close_price - open_price)
    lower_shadow = min(open_price, close_price) - low_price
    upper_shadow = high_price - max(open_price, close_price)
    total_range = high_price - low_price

    if total_range == 0: return False # 0으로 나누는 오류 방지

    # 조건 강화: 몸통이 전체 길이의 특정 비율 미만, 아래 꼬리가 몸통의 특정 배수 이상, 위 꼬리가 매우 짧음
    is_small_body = body <= 0.3 * total_range # 몸통이 전체 길이의 30% 이하
    has_long_lower_shadow = lower_shadow >= 2 * body # 아래 꼬리가 몸통의 2배 이상
    has_small_upper_shadow = upper_shadow <= 0.1 * body # 위 꼬리가 몸통의 10% 이하 (매우 짧음)

    return is_small_body and has_long_lower_shadow and has_small_upper_shadow

# --- 스마트 시그널 로직 (smart_signal_row) ---
def smart_signal_row(row, prev_row, prev2_row):
    # 필수 지표 확인 (NaN 값 처리)
    required_indicators = ['MACD', 'Signal', 'MACD_Hist', 'RSI', 'MA20', 'Volume_MA20', 'ATR', 'ADX']
    if any(pd.isna(row[ind]) for ind in required_indicators):
        return "관망"

    current_close = row['Close']
    prev_close = prev_row['Close']
    macd_hist_direction = row['MACD_Hist'] - prev_row['MACD_Hist']

    # 1. RSI 극단값 및 반전 시그널 (최우선)
    if row['RSI'] >= 80: # RSI 과매수 강력
        return "강력 매도"
    if row['RSI'] <= 20: # RSI 과매도 강력
        return "강력 매수"
    if row['RSI'] >= 70 and prev_row['RSI'] < 70: # RSI 70 진입 (익절)
        return "익절 매도"
    if row['RSI'] <= 30 and prev_row['RSI'] > 30: # RSI 30 진입 (매수)
        return "신규 매수"

    # 2. 강력 매수/매도 시그널 (복합 지표)
    # 강력 매수: MACD 골든크로스 + MA20 상향 돌파 + 거래량 급증 + 추세 강세 (+DI > -DI and ADX > 20)
    if (is_macd_golden_cross(prev_row, row) and
        is_ma_cross_up(prev_row, row) and
        is_volume_surge(row) and
        row['+DI14'] > row['-DI14'] and row['ADX'] > 25): # ADX 20에서 25로 강화
        return "강력 매수"

    # 강력 매도: MACD 데드크로스 + MA20 하향 돌파 + 거래량 급증 + 추세 약세 (+DI < -DI and ADX > 20)
    if (is_macd_dead_cross(prev_row, row) and
        is_ma_cross_down(prev_row, row) and
        is_volume_surge(row) and
        row['+DI14'] < row['-DI14'] and row['ADX'] > 25): # ADX 20에서 25로 강화
        return "강력 매도"

    # 3. 모멘텀 변화 (MACD Hist)
    if row['MACD_Hist'] < 0 and macd_hist_direction > 0: # MACD 히스토그램 음수 구간에서 상승 전환 (매수 모멘텀 강화)
        return "매수 고려"
    if row['MACD_Hist'] > 0 and macd_hist_direction < 0: # MACD 히스토그램 양수 구간에서 하락 전환 (매도 모멘텀 강화)
        return "매도 고려"

    # 4. 일반적인 매수/매도 시그널
    if is_macd_golden_cross(prev_row, row):
        return "신규 매수"
    if is_macd_dead_cross(prev_row, row):
        return "매도"

    if is_ma_cross_up(prev_row, row):
        return "신규 매수"
    if is_ma_cross_down(prev_row, row):
        return "매도"

    # 5. 보조 시그널
    if is_bullish_divergence(prev2_row, prev_row, row): # 다이버전스 감지는 3개 봉 필요
        return "반등 가능성"
    if is_bearish_divergence(prev2_row, prev_row, row):
        return "하락 가능성"
    if is_hammer_candlestick(row, prev_row): # 해머 캔들스틱 (반전)
        return "반전 신호"
    if current_close > row['BB_Upper'] and prev_close <= prev_row['BB_Upper']: # 볼린저밴드 상단 돌파 (과매수 또는 강한 추세)
        if row['RSI'] > 70: # RSI 70 이상이면 과매수로 인한 매도 가능성
            return "익절 매도"
        else: # RSI 70 미만이면 강한 추세 유지
            return "보유"
    if current_close < row['BB_Lower'] and prev_close >= prev_row['BB_Lower']: # 볼린저밴드 하단 돌파 (과매도 또는 강한 하락 추세)
        if row['RSI'] < 30: # RSI 30 미만이면 과매수로 인한 매수 가능성
            return "신규 매수"
        else: # RSI 30 이상이면 강한 하락 추세 유지
            return "관망"

    # 그 외
    return "관망"

def smart_signal(df_with_signals):
    if df_with_signals.empty:
        return "데이터 부족"
    # 가장 최근의 유효한 시그널 반환
    last_valid_signal = df_with_signals['TradeSignal'].iloc[-1]
    return last_valid_signal

# --- 추천 점수 계산 (compute_recommendation_score) ---
def compute_recommendation_score(last, prev_row):
    score = 50 # 기본 점수 50점

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
    if last['RSI'] > 65: # 과매수 (매도)
        score -= 20 # 가중치 강화
    elif last['RSI'] < 35: # 과매도 (매수)
        score += 20 # 가중치 강화
    elif last['RSI'] > 50: # 강세 영역
        score += 5
    elif last['RSI'] < 50: # 약세 영역
        score -= 5

    # 이동평균선
    if last['Close'] > last['MA20'] and last['MA20'] > last['MA60']: # 정배열
        score += 10
    elif last['Close'] < last['MA20'] and last['MA20'] < last['MA60']: # 역배열
        score -= 10

    # ADX (추세 강도 및 방향)
    if last['ADX'] > 25: # 추세가 강할 때
        if last['+DI14'] > last['-DI14']: # 상승 추세
            score += 7
        else: # 하락 추세
            score -= 7
    elif last['ADX'] < 20: # 추세가 약할 때 (횡보)
        score -= 5 # 횡보는 기회비용 발생 가능성

    # 거래량
    if last['Volume'] > last['Volume_MA20'] * 1.5: # 거래량 급증 (시그널 신뢰도 증가)
        score += 5
    elif last['Volume'] < last['Volume_MA20'] * 0.7: # 거래량 급감 (관심 감소 또는 추세 약화)
        score -= 3

    # TradeSignal 자체에 큰 가중치 부여 (최종 시그널 반영)
    if "강력 매수" in last['TradeSignal']:
        score += 30
    elif "신규 매수" in last['TradeSignal'] or "매수 고려" in last['TradeSignal'] or "반등 가능성" in last['TradeSignal']:
        score += 15
    elif "강력 매도" in last['TradeSignal']:
        score -= 30
    elif "매도" in last['TradeSignal'] or "익절 매도" in last['TradeSignal'] or "매도 고려" in last['TradeSignal'] or "하락 가능성" in last['TradeSignal']:
        score -= 15
    elif "관망" in last['TradeSignal']:
        score = max(score, 40) if score > 50 else min(score, 60) # 관망 시그널이면 40-60 범위로 조정

    # 점수 정규화 (0-100)
    score = max(0, min(100, score))
    return score

# --- 추천 행동 및 비율 결정 (get_action_and_percentage_by_score) ---
def get_action_and_percentage_by_score(signal, score):
    action = "관망"
    percentage = 0

    if "강력 매수" in signal:
        action = "신규 매수"
        percentage = min(100, 70 + (score - 50) * 0.6) # 점수가 높을수록 매수 비율 증가
    elif "신규 매수" in signal or "매수 고려" in signal or "반등 가능성" in signal:
        action = "신규 매수"
        percentage = min(70, 30 + (score - 50) * 0.5)
        if score < 50: percentage = max(10, percentage) # 최소 10%
    elif "익절 매도" in signal:
        action = "익절 매도"
        percentage = min(100, 30 + (score - 50) * 0.7) # 점수가 높을수록 매도 비율 증가 (음수 점수면 매도량 줄임)
        if score < 50: percentage = max(10, percentage) # 최소 10%
    elif "매도" in signal or "매도 고려" in signal or "하락 가능성" in signal:
        action = "매도"
        percentage = min(100, 50 + (50 - score) * 0.7) # 점수가 낮을수록 매도 비율 증가
    elif "강력 매도" in signal:
        action = "전량 매도"
        percentage = 100
    elif "보유" in signal or "관망" in signal or "반전 신호" in signal:
        action = "관망"
        percentage = 0

    return action, round(percentage)

# --- ChatGPT 프롬프트 생성 (generate_chatgpt_prompt) ---
def generate_chatgpt_prompt(ticker, rsi, macd, macd_hist, signal_line, atr, adx):
    rsi_str = f"RSI: {rsi:.2f}" if not np.isnan(rsi) else "RSI: N/A"
    macd_str = f"MACD: {macd:.2f}, Signal: {signal_line:.2f}, Hist: {macd_hist:.2f}" if not np.isnan(macd) else "MACD: N/A"
    atr_str = f"ATR: {atr:.2f}" if not np.isnan(atr) else "ATR: N/A"
    adx_str = f"ADX: {adx:.2f}" if not np.isnan(adx) else "ADX: N/A"

    return f"{ticker}: {rsi_str}, {macd_str}, {atr_str}, {adx_str}"

# --- 이메일 전송 함수 ---
def send_email(subject, body, to_email, from_email, password, attachments=None):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html')) # HTML 형식으로 본문을 보낼 경우 'html'

    if attachments:
        for file_path, file_name in attachments:
            try:
                with open(file_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=file_name)
                part['Content-Disposition'] = f'attachment; filename="{file_name}"'
                msg.attach(part)
            except FileNotFoundError:
                print(f"경고: 첨부 파일 {file_path}를 찾을 수 없습니다. 스킵합니다.")
            except Exception as e:
                print(f"경고: 첨부 파일 {file_path} 처리 중 오류 발생: {e}. 스킵합니다.")

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

    SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'parkib63@gmail.com') 
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'bdnj dicf dzea wdrq') 
    RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL', 'parkib63@naver.com') 
    STREAMLIT_APP_URL = os.getenv('STREAMLIT_APP_URL', 'https://app-stock-app-bomipark.streamlit.app/')


    if send_email_mode:
        print("🚀 이메일 보고서 전송 모드로 시작합니다...")
        email_summary_rows = []
        email_tech_summaries_text = []


        macro_start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        macro_data = download_macro_data(macro_start_date, END_DATE)
        market_condition = macro_filter(macro_data)

        email_body_parts = []
        email_body_parts.append(f"<h1>📈 미국 주식 시그널 대시보드 - {datetime.now().strftime('%Y년 %m월 %d일')}</h1>")
        email_body_parts.append(f"<h2>마켓 센티멘트 현황</h2>")
        email_body_parts.append(f"<p>- VIX (변동성 지수): <b>{macro_data.get('VIX', 'N/A'):.2f}</b></p>")
        email_body_parts.append(f"<p>- 미국 10년 국채 금리: <b>{macro_data.get('US10Y', 'N/A'):.2f}%</b></p>")
        email_body_parts.append(f"<p>- 시장 상태: <b>{market_condition}</b></p>")
        email_body_parts.append(f"<p><b>자세한 분석 및 실시간 캔들스틱 차트를 보려면 아래 링크를 클릭하세요:</b></p>") # 변경됨: 링크 안내 추가
        email_body_parts.append(f"<p><a href='{STREAMLIT_APP_URL}'>👉 Streamlit 주식 시그널 대시보드 바로가기</a></p>") # 변경됨: 링크 추가


        print(f"시장 상태: {market_condition}")

        for ticker in TICKERS:
            print(f"처리 중: {ticker}...")
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=START_DATE, end=END_DATE, interval="1d")

                if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    print(f"❌ {ticker} 데이터 누락 또는 형식 오류. 스킵합니다.")
                    continue

                if 'Adj Close' in data.columns:
                    data = data.drop(columns=['Adj Close'])

                df = calc_indicators(data[['Open', 'High', 'Low', 'Close', 'Volume']].copy())

                if df.empty or len(df) < MIN_DATA_REQUIRED_FOR_INDICATORS:
                    print(f"❌ {ticker} 지표 계산 후 데이터 부족 ({len(df)}개). 스킵합니다.")
                    continue

                # Streamlit UI 모드에서만 사용되는 임시 컬럼이므로 이메일 모드에서는 불필요하지만, 로직 일관성을 위해 유지
                df['TradeSignal'] = ["관망"] * len(df)
                for i in range(2, len(df)):
                    # prev2_row를 사용할 수 있도록 인덱스 확인
                    if i - 2 >= 0:
                        df.loc[df.index[i], 'TradeSignal'] = smart_signal_row(df.iloc[i], df.iloc[i-1], df.iloc[i-2])
                    elif i - 1 >= 0: # prev2_row가 없을 경우 prev_row까지만 전달
                        df.loc[df.index[i], 'TradeSignal'] = smart_signal_row(df.iloc[i], df.iloc[i-1], df.iloc[i-1]) # prev2_row 대신 prev_row 재사용
                    else: # 첫 두 행은 시그널 계산 불가
                        df.loc[df.index[i], 'TradeSignal'] = "관망"

                last = df.iloc[-1]
                prev_row = df.iloc[-2]

                signal = last['TradeSignal']
                signal = soften_signal(signal, market_condition)
                df.loc[df.index[-1], 'TradeSignal'] = signal # 최종 시그널 업데이트

                score = compute_recommendation_score(last, prev_row)
                score = adjust_score(score, market_condition)
                action, pct = get_action_and_percentage_by_score(signal, score)

                email_summary_rows.append({
                    "Ticker": ticker,
                    "Signal": signal,
                    "Score": f"{score:.1f}",
                    "추천 행동": action,
                    "비율(%)": f"{pct}%",
                })

                rsi_val = float(last.get('RSI', np.nan))
                macd_val = float(last.get('MACD', np.nan))
                macd_hist_val = float(last.get('MACD_Hist', np.nan))
                signal_line_val = float(last.get('Signal', np.nan))
                atr_val = float(last.get('ATR', np.nan))
                adx_val = float(last.get('ADX', np.nan))

                email_tech_summaries_text.append(generate_chatgpt_prompt(ticker, rsi_val, macd_val, macd_hist_val, signal_line_val, atr_val, adx_val))

                # 캔들스틱 차트 생성 및 이미지로 저장
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']
                )])
                fig.update_layout(title=f'{ticker} 캔들스틱 차트', xaxis_rangeslider_visible=False)
                chart_image_path = f"{ticker}_candlestick_chart.png"
                fig.write_image(chart_image_path, width=800, height=400) # 이미지 해상도 설정

            except ValueError as ve:
                print(f"❌ {ticker} 지표 계산 중 오류 발생: {ve}. 스킵합니다.")
            except Exception as e:
                print(f"❌ {ticker} 데이터 처리 중 알 수 없는 오류 발생: {e}. 스킵합니다.")
                continue

        if email_summary_rows:
            email_body_parts.append("<h2>📋 오늘의 종목별 매매 전략 요약</h2>")
            # DataFrame을 HTML 테이블로 변환
            email_body_parts.append(pd.DataFrame(email_summary_rows).to_html(index=False))

        if email_tech_summaries_text:
            ai_prompt_template = """
<br>
<h3>🧠 AI에게 물어보는 기술적 분석 프롬프트</h3>
<p>아래 각 종목의 기술적 지표만 보고, 미국 주식 전문 트레이더처럼 매수/매도/익절/보유/관망 시그널과 매수/매도/익절이 필요한 경우 “몇 % 정도” 하면 좋을지도 같이 구체적으로 알려줘.</p>
<p>- 한 종목당 한 줄씩,<br>- 신호와 추천 비율(%)만 간단명료하게<br>- 사유도 한 줄로 덧붙여줘.</p>
<p><b>[질문]</b></p>
<p>- 각 종목별로<br>  1) 매수/매도/익절/보유/관망 중 뭐가 적합한지<br>  2) 추천 비율(%)은 얼마나 할지 (예: “익절 30%” “신규매수 50%” 등)<br>  3) 근거 한 줄</p>
<p>아래 표로 정리해서 답변해줘.</p>
<pre><code>| 종목 | 추천액션 | 비율(%) | 근거 요약 |
|------|----------------|---------|-----------------------------|
"""
            full_ai_prompt_content = ai_prompt_template + "\n".join(email_tech_summaries_text) + "\n</code></pre>"
            email_body_parts.append(full_ai_prompt_content)

        final_email_body = "".join(email_body_parts)

        # 이메일 전송
        EMAIL_SUBJECT = f"미국 주식 시그널 대시보드 - {datetime.now().strftime('%Y-%m-%d')}"
        send_email(EMAIL_SUBJECT, final_email_body, RECEIVER_EMAIL, SENDER_EMAIL, SENDER_PASSWORD)


    else:
        # --- Streamlit 대시보드 UI ---
        st.set_page_config(layout="wide", page_title="미국 주식 시그널 대시보드")
        st.title("📈 미국 주식 시그널 대시보드")
        st.subheader(f"데이터 기준일: {END_DATE}")

        # 마켓 센티멘트
        st.markdown("---")
        st.subheader("마켓 센티멘트 현황")
        macro_start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        macro_data = download_macro_data(macro_start_date, END_DATE)
        market_condition = macro_filter(macro_data)

        col_vix, col_us10y, col_market = st.columns(3)
        with col_vix:
            st.metric("VIX (변동성 지수)", f"{macro_data.get('VIX', 'N/A'):.2f}")
        with col_us10y:
            st.metric("미국 10년 국채 금리", f"{macro_data.get('US10Y', 'N/A'):.2f}%")
        with col_market:
            st.metric("시장 상태", market_condition)

        st.markdown("---")

        summary_rows = []
        all_tech_summaries_text = []

        for ticker in TICKERS:
            st.markdown(f"### {ticker}")
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=START_DATE, end=END_DATE, interval="1d")

                if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    st.warning(f"❌ **{ticker}** 데이터 누락 또는 형식 오류. 스킵합니다.")
                    continue

                if 'Adj Close' in data.columns:
                    data = data.drop(columns=['Adj Close'])

                df = calc_indicators(data[['Open', 'High', 'Low', 'Close', 'Volume']].copy())

                if df.empty or len(df) < MIN_DATA_REQUIRED_FOR_INDICATORS:
                    st.warning(f"❌ **{ticker}** 지표 계산 후 데이터 부족 ({len(df)}개). 시그널 생성을 건너뛰었습니다.")
                    continue

                df['TradeSignal'] = ["관망"] * len(df)
                for i in range(2, len(df)):
                    if i - 2 >= 0:
                        df.loc[df.index[i], 'TradeSignal'] = smart_signal_row(df.iloc[i], df.iloc[i-1], df.iloc[i-2])
                    elif i - 1 >= 0:
                        df.loc[df.index[i], 'TradeSignal'] = smart_signal_row(df.iloc[i], df.iloc[i-1], df.iloc[i-1])
                    else:
                        df.loc[df.index[i], 'TradeSignal'] = "관망"

                last = df.iloc[-1]
                prev_row = df.iloc[-2]

                signal = last['TradeSignal']
                signal = soften_signal(signal, market_condition) # 거시경제 필터링 적용
                df.loc[df.index[-1], 'TradeSignal'] = signal # 최종 시그널 업데이트

                score = compute_recommendation_score(last, prev_row)
                score = adjust_score(score, market_condition) # 거시경제에 따른 점수 조정
                action, pct = get_action_and_percentage_by_score(signal, score)

                st.write(f"**현재 시그널**: **{signal}**")
                st.write(f"**추천 행동**: **{action} ({pct}%)**")
                st.write(f"**추천 점수**: **{score:.1f}/100**")

                summary_rows.append({
                    "Ticker": ticker,
                    "Signal": signal,
                    "Score": f"{score:.1f}",
                    "추천 행동": action,
                    "비율(%)": f"{pct}%",
                })

                st.markdown("---")
                st.subheader(f"{ticker} 최근 지표")
                st.dataframe(df.tail(7)[['Close', 'MA20', 'MACD', 'Signal', 'MACD_Hist', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR', 'ADX', '+DI14', '-DI14', 'Volume', 'Volume_MA20', 'TradeSignal']])
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

                all_tech_summaries_text.append(generate_chatgpt_prompt(ticker, rsi_val, macd_val, macd_hist_val, signal_line_val, atr_val, adx_val))

            except ValueError as ve:
                st.error(f"❌ **{ticker}** 지표 계산 중 오류 발생: **{ve}**")
                st.warning(f"**{ticker}**에 대한 시그널 생성을 건너뛰었습니다.")
            except Exception as e:
                st.error(f"❌ **{ticker}** 데이터 처리 중 알 수 없는 오류 발생: **{e}**")
                st.warning(f"**{ticker}**에 대한 시그널 생성을 건너뛰었습니다.")

        if summary_rows:
            st.subheader("📋 오늘의 종목별 매매 전략 요약")
            st.markdown("""
            - **'신규 매수'**: 신규 진입을 고려하는 투자자에게 권장합니다.
            - **'익절 매도'**: 보유 주식의 일부를 수익 실현을 위해 매도하는 것을 권장합니다.
            - **'매도'**: 보유 주식의 절반 이상을 매도하여 리스크를 줄이는 것을 권장합니다.
            - **'전량 매도'**: 보유 주식 전체를 매도하여 포지션을 정리하는 것을 권장합니다.
            - **'관망'**: 현재 포지션을 유지하거나 시장의 추가적인 신호를 기다리는 것을 권장합니다.
            """)
            st.dataframe(pd.DataFrame(summary_rows))

        # 변경된 AI 프롬프트 부분
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

| 종목 | 추천액션       | 비율(%) | 근거 요약                   |
|------|----------------|---------|-----------------------------|


"""

    
            full_ai_prompt_content = ai_prompt_template + "\n" + "\n".join(all_tech_summaries_text)

            st.code(full_ai_prompt_content, language='markdown', line_numbers=False)
            st.markdown("👆 위 프롬프트 내용 옆 **'Copy to clipboard' 버튼**을 클릭하여 쉽게 복사할 수 있습니다.")
