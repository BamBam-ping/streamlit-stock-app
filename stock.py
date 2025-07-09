import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np

# --- 1. 설정 (Configuration) 및 글로벌 변수 ---
TICKERS = [
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "TSLA", "PANW", "AMD", "TSM", "ORCL",
    "ADBE", "LLY", "UNH", "VRTX", "REGN", "JPM", "V", "MS", "JNJ", "HOOD", "SPY", "QQQ", "SCHD"
]

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
START_DATE = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
MIN_DATA_REQUIRED_FOR_INDICATORS = 180

# --- 기술적 지표 계산 ---
def calc_indicators(df):
    if df.empty:
        return df
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    n_stoch = 14
    m_stoch = 3
    df['Lowest_Low'] = df['Low'].rolling(window=n_stoch).min()
    df['Highest_High'] = df['High'].rolling(window=n_stoch).max()
    df['%K'] = ((df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])) * 100
    df['%D'] = df['%K'].rolling(window=m_stoch).mean()
    n_cci = 20
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_MA'] = df['TP'].rolling(window=n_cci).mean()
    df['Mean_Deviation'] = df['TP'].rolling(window=n_cci).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (df['TP'] - df['TP_MA']) / (0.015 * df['Mean_Deviation'])
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_StdDev'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)
    high_low = df['High'] - df['Low']
    high_prev_close = abs(df['High'] - df['Close'].shift())
    low_prev_close = abs(df['Low'] - df['Close'].shift())
    tr = pd.DataFrame({'HL': high_low, 'HPC': high_prev_close, 'LPC': low_prev_close}).max(axis=1)
    df['ATR'] = tr.ewm(span=14, adjust=False).mean()
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    df['TR_ADX'] = tr
    df['+DI14'] = (df['+DM'].rolling(window=14).sum() / df['TR_ADX'].rolling(window=14).sum()) * 100
    df['-DI14'] = (df['-DM'].rolling(window=14).sum() / df['TR_ADX'].rolling(window=14).sum()) * 100
    df['DX'] = abs(df['+DI14'] - df['-DI14']) / (df['+DI14'] + df['-DI14']) * 100
    df['ADX'] = df['DX'].ewm(span=14, adjust=False).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
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
    if len(df) < MIN_DATA_REQUIRED_FOR_INDICATORS:
        return pd.DataFrame()
    df = df.dropna(subset=['MA20', 'MACD', 'RSI', 'BB_Middle', 'ATR', 'ADX', 'Volume_MA20', '%K', '%D', 'CCI', 'BB_Squeeze_Up_Breakout', 'BB_Squeeze_Down_Breakout'])
    return df

# --- 시그널/액션/스코어 함수들 (생략 없이 붙여넣으세요. 기존 코드 활용) ---
# ... smart_signal_row, smart_signal, compute_recommendation_score, get_action_and_percentage_by_score 등...
# (이전 답변/기존파일 그대로)

# --- BB Squeeze Breakout 종목 표시 (⭐ 이모지) ---
def format_signal_label(ticker, signal, last_row):
    star = ""
    if last_row.get('BB_Squeeze_Up_Breakout', False) or last_row.get('BB_Squeeze_Down_Breakout', False):
        star = " ⭐"
    if "매수" in signal or "반등 가능성" in signal:
        symbol = f"<span style='color: green;'>▲</span>"
    elif "매도" in signal or "하락 가능성" in signal or "익절 매도" in signal:
        symbol = f"<span style='color: red;'>▼</span>"
    else:
        symbol = f"<span style='color: orange;'>●</span>"
    return f"- {ticker} {symbol}{star} - {TICKER_DESCRIPTIONS.get(ticker, '')}"

# --- Main Streamlit 엔트리포인트 ---
def main():
    st.set_page_config(layout="wide", page_title="US Stock Signal Dashboard")
    st.title("📈 US Stock Signal Dashboard")
    st.subheader(f"데이터 기준일: {END_DATE}")

    summary_rows = []
    all_ticker_data = {}
    for ticker in TICKERS:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=START_DATE, end=END_DATE, interval="1d")
        if data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            continue
        if 'Adj Close' in data.columns:
            data = data.drop(columns=['Adj Close'])
        df = calc_indicators(data[['Open', 'High', 'Low', 'Close', 'Volume']].copy())
        if df.empty or len(df) < MIN_DATA_REQUIRED_FOR_INDICATORS:
            continue
        df['TradeSignal'] = ["관망"] * len(df)
        for i in range(2, len(df)):
            df.loc[df.index[i], 'TradeSignal'] = smart_signal_row(df.iloc[i], df.iloc[i-1], df.iloc[i-2])
        last = df.iloc[-1]
        signal = last['TradeSignal']
        all_ticker_data[ticker] = {
            'last': last,
            'signal': signal
        }

    st.subheader("📊 전체 종목별 매매 시그널 현황")
    buy_tickers, sell_tickers, hold_tickers = [], [], []
    for ticker, data_for_ticker in all_ticker_data.items():
        signal = data_for_ticker['signal']
        last_row = data_for_ticker['last']
        if "매수" in signal or "반등 가능성" in signal:
            buy_tickers.append((ticker, signal, last_row))
        elif "매도" in signal or "하락 가능성" in signal or "익절 매도" in signal:
            sell_tickers.append((ticker, signal, last_row))
        else:
            hold_tickers.append((ticker, signal, last_row))
    if buy_tickers or sell_tickers or hold_tickers:
        col_buy, col_sell, col_hold = st.columns(3)
        with col_buy:
            st.markdown("#### ✅ 매수 시그널 종목")
            if buy_tickers:
                for t, s, l in buy_tickers:
                    st.markdown(format_signal_label(t, s, l), unsafe_allow_html=True)
            else:
                st.write("없음")
        with col_sell:
            st.markdown("#### 🔻 매도 시그널 종목")
            if sell_tickers:
                for t, s, l in sell_tickers:
                    st.markdown(format_signal_label(t, s, l), unsafe_allow_html=True)
            else:
                st.write("없음")
        with col_hold:
            st.markdown("#### 🟡 관망/보유 시그널 종목")
            if hold_tickers:
                for t, s, l in hold_tickers:
                    st.markdown(format_signal_label(t, s, l), unsafe_allow_html=True)
            else:
                st.write("없음")
    else:
        st.info("시그널을 분석할 수 있는 종목 데이터가 부족합니다.")
    # ... (아래로 상세 요약/차트 등 기존 로직 이어서 사용)

if __name__ == '__main__':
    main()
