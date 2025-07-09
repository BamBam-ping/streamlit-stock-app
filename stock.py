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

# ==============================
# 1. 설정/상수 영역 (Config)
# ==============================
TICKERS = [
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "TSLA", "PANW", "AMD",
    "TSM", "ORCL", "ADBE", "LLY", "UNH", "VRTX", "REGN", "JPM", "V",
    "MS", "JNJ", "HOOD", "SPY", "QQQ", "SCHD"
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
MIN_DATA_REQUIRED = 180

# ==============================
# 2. 데이터 수집/가공 함수
# ==============================
@st.cache_data
def download_macro_data(start, end):
    macro_tickers = {"VIX":"^VIX","US10Y":"^TNX","US3M":"^IRX","S&P500":"^GSPC","NASDAQ":"^IXIC","DXY":"DX-Y.NYB"}
    res = {}
    fetch_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
    for name, tkr in macro_tickers.items():
        try:
            data = yf.Ticker(tkr).history(start=fetch_start, end=end)
            if not data.empty and not data['Close'].dropna().empty:
                c = data['Close'].dropna()
                res[name] = {
                    "value": c.iloc[-1].item(),
                    "change": (c.iloc[-1]-c.iloc[-2]) if len(c)>=2 else np.nan
                }
            else:
                res[name] = {"value": np.nan, "change": np.nan}
        except:
            res[name] = {"value": np.nan, "change": np.nan}
    us10y, us3m = res.get("US10Y",{}).get("value",np.nan), res.get("US3M",{}).get("value",np.nan)
    res["Yield_Spread_10Y_3M"] = {"value": us10y-us3m if not(np.isnan(us10y) or np.isnan(us3m)) else np.nan, "change": np.nan}
    return res

def get_ticker_history(ticker, start, end):
    data = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    if data.empty or not all(col in data.columns for col in ['Open','High','Low','Close','Volume']):
        return pd.DataFrame()
    return data[['Open','High','Low','Close','Volume']].copy()

# ==============================
# 3. 기술적 지표 계산 함수
# ==============================
def calc_indicators(df):
    if df.empty: return df
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['MA120'] = df['Close'].rolling(120).mean()
    df['EMA12'] = df['Close'].ewm(span=12,adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26,adjust=False).mean()
    df['MACD'] = df['EMA12']-df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9,adjust=False).mean()
    df['MACD_Hist'] = df['MACD']-df['Signal']
    delta = df['Close'].diff()
    gain = (delta.where(delta>0,0)).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    rs = gain/loss
    df['RSI'] = 100 - (100/(1+rs))
    n,m = 14,3
    df['Lowest_Low'] = df['Low'].rolling(n).min()
    df['Highest_High'] = df['High'].rolling(n).max()
    df['%K'] = ((df['Close']-df['Lowest_Low'])/(df['Highest_High']-df['Lowest_Low']))*100
    df['%D'] = df['%K'].rolling(m).mean()
    n_cci = 20
    df['TP'] = (df['High']+df['Low']+df['Close'])/3
    df['TP_MA'] = df['TP'].rolling(n_cci).mean()
    df['Mean_Deviation'] = df['TP'].rolling(n_cci).apply(lambda x: np.abs(x-x.mean()).mean(),raw=True)
    df['CCI'] = (df['TP']-df['TP_MA'])/(0.015*df['Mean_Deviation'])
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_StdDev'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle']+(df['BB_StdDev']*2)
    df['BB_Lower'] = df['BB_Middle']-(df['BB_StdDev']*2)
    tr = pd.DataFrame({
        'HL': df['High']-df['Low'],
        'HPC': abs(df['High']-df['Close'].shift()),
        'LPC': abs(df['Low']-df['Close'].shift())
    }).max(axis=1)
    df['ATR'] = tr.ewm(span=14,adjust=False).mean()
    df['UpMove'] = df['High']-df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1)-df['Low']
    df['+DM'] = np.where((df['UpMove']>df['DownMove']) & (df['UpMove']>0),df['UpMove'],0)
    df['-DM'] = np.where((df['DownMove']>df['UpMove']) & (df['DownMove']>0),df['DownMove'],0)
    df['TR_ADX'] = tr
    df['+DI14'] = (df['+DM'].rolling(14).sum()/df['TR_ADX'].rolling(14).sum())*100
    df['-DI14'] = (df['-DM'].rolling(14).sum()/df['TR_ADX'].rolling(14).sum())*100
    df['DX'] = abs(df['+DI14']-df['-DI14'])/(df['+DI14']+df['-DI14'])*100
    df['ADX'] = df['DX'].ewm(span=14,adjust=False).mean()
    df['Volume_MA20'] = df['Volume'].rolling(20).mean()
    window_squeeze=20
    df['Band_Width'] = df['BB_Upper']-df['BB_Lower']
    df['Band_Width_MA'] = df['Band_Width'].rolling(window_squeeze).mean()
    df['BB_Squeeze_Up_Breakout'] = False
    df['BB_Squeeze_Down_Breakout'] = False
    for i in range(1, len(df)):
        if i < window_squeeze: continue
        is_squeeze = df['Band_Width_MA'].iloc[i-1] < (df['BB_Middle'].iloc[i-1]*0.02)
        if is_squeeze and df['Close'].iloc[i]>df['BB_Upper'].iloc[i] and df['Close'].iloc[i-1]<=df['BB_Upper'].iloc[i-1] and df['Volume'].iloc[i]>df['Volume_MA20'].iloc[i]*1.5:
            df.loc[df.index[i],'BB_Squeeze_Up_Breakout']=True
        if is_squeeze and df['Close'].iloc[i]<df['BB_Lower'].iloc[i] and df['Close'].iloc[i-1]>=df['BB_Lower'].iloc[i-1] and df['Volume'].iloc[i]>df['Volume_MA20'].iloc[i]*1.5:
            df.loc[df.index[i],'BB_Squeeze_Down_Breakout']=True
    df = df.drop(columns=['Band_Width','Band_Width_MA'])
    if len(df) < MIN_DATA_REQUIRED: return pd.DataFrame()
    return df.dropna(subset=['MA20','MACD','RSI','BB_Middle','ATR','ADX','Volume_MA20','%K','%D','CCI','BB_Squeeze_Up_Breakout','BB_Squeeze_Down_Breakout'])

# ==============================
# 4. 시그널/점수 보조 함수
# ==============================
def is_macd_golden_cross(prev, curr): return prev['MACD']<prev['Signal'] and curr['MACD']>curr['Signal']
def is_macd_dead_cross(prev, curr): return prev['MACD']>prev['Signal'] and curr['MACD']<curr['Signal']
def is_rsi_oversold(row): return row['RSI']<35
def is_rsi_overbought(row): return row['RSI']>65
def is_ma_cross_up(prev, curr): return prev['Close']<prev['MA20'] and curr['Close']>curr['MA20']
def is_ma_cross_down(prev, curr): return prev['Close']>prev['MA20'] and curr['Close']<curr['MA20']
def is_volume_surge(row): return row['Volume'] > (row['Volume_MA20']*1.5)
def is_stoch_oversold(row): return row['%K']<20 and row['%D']<20
def is_stoch_overbought(row): return row['%K']>80 and row['%D']>80
def is_stoch_golden_cross(prev, curr): return prev['%K']<prev['%D'] and curr['%K']>curr['%D'] and curr['%K']<80
def is_stoch_dead_cross(prev, curr): return prev['%K']>prev['%D'] and curr['%K']<curr['%D'] and curr['%K']>20
def is_cci_oversold(row): return row['CCI']<-100
def is_cci_overbought(row): return row['CCI']>100

def soften_signal(signal, market_condition):
    if signal in ["강력 매수","강력 매도"]: return signal
    if market_condition in ["약세 (극변동)","약세 (고변동)","고금리","중립 (고금리)"]:
        if "매수" in signal: return "관망"
    return signal

def adjust_score(score, market_condition):
    if market_condition in ["약세 (극변동)","약세 (고변동)"]: return max(0,score*0.7)
    if market_condition in ["고금리","중립 (고금리)"]: return max(0,score*0.8)
    if market_condition in ["강세 (저변동)","강세 (중변동)","강세 (고금리)"]: return min(100,score*1.1)
    return score

def smart_signal_row(row, prev_row, prev2_row):
    try:
        req = ['MACD','Signal','MACD_Hist','RSI','MA20','Volume_MA20','ATR','ADX','%K','%D','CCI','BB_Squeeze_Up_Breakout','BB_Squeeze_Down_Breakout']
        if any(pd.isna(row.get(ind,np.nan)) for ind in req): return "관망"
        if (is_macd_golden_cross(prev_row,row) and is_ma_cross_up(prev_row,row) and is_volume_surge(row)
            and row['ADX']>25 and row['+DI14']>row['-DI14'] and is_stoch_golden_cross(prev_row,row)
            and prev_row['RSI']<=30 and row['RSI']>30): return "강력 매수"
        if (is_macd_dead_cross(prev_row,row) and is_ma_cross_down(prev_row,row) and is_volume_surge(row)
            and row['ADX']>25 and row['+DI14']<row['-DI14'] and is_stoch_dead_cross(prev_row,row)
            and prev_row['RSI']>=70 and row['RSI']<70): return "강력 매도"
        if row['BB_Squeeze_Up_Breakout']: return "강력 매수"
        if row['BB_Squeeze_Down_Breakout']: return "강력 매도"
        if row['RSI']>=80: return "익절 매도"
        if row['RSI']<=20: return "신규 매수"
        if is_stoch_overbought(row) and is_stoch_dead_cross(prev_row,row): return "매도"
        if is_stoch_oversold(row) and is_stoch_golden_cross(prev_row,row): return "신규 매수"
        if is_cci_overbought(row) and row['CCI']<prev_row['CCI']: return "매도 고려"
        if is_cci_oversold(row) and row['CCI']>prev_row['CCI']: return "매수 고려"
        macd_hist_direction = row['MACD_Hist']-prev_row['MACD_Hist']
        if row['MACD_Hist']<0 and macd_hist_direction>0: return "매수 고려"
        if row['MACD_Hist']>0 and macd_hist_direction<0: return "매도 고려"
        if is_macd_golden_cross(prev_row,row): return "신규 매수"
        if is_macd_dead_cross(prev_row,row): return "매도"
        if is_ma_cross_up(prev_row,row): return "신규 매수"
        if is_ma_cross_down(prev_row,row): return "매도"
        return "관망"
    except: return "관망"

def compute_recommendation_score(last, prev_row, per, market_cap, forward_pe, debt_to_equity):
    score = 50
    if last['MACD']>last['Signal']: score += 15
    else: score -= 15
    if last['MACD_Hist']>0 and (last['MACD_Hist']-prev_row['MACD_Hist'])>0: score += 10
    elif last['MACD_Hist']<0 and (last['MACD_Hist']-prev_row['MACD_Hist'])<0: score -= 10
    if last['RSI']>70: score -= 25
    elif last['RSI']<30: score += 25
    elif last['RSI']>60: score -= 10
    elif last['RSI']<40: score += 10
    elif last['RSI']>50: score += 5
    elif last['RSI']<50: score -= 5
    if last['%K']>last['%D'] and last['%K']<80: score += 10
    elif last['%K']<last['%D'] and last['%K']>20: score -= 10
    if is_stoch_oversold(last): score += 15
    if is_stoch_overbought(last): score -= 15
    if last['CCI']>100: score -= 10
    elif last['CCI']<-100: score += 10
    if last['Close']>last['MA20'] and last['MA20']>last['MA60'] and last['MA60']>last['MA120']: score += 15
    elif last['Close']<last['MA20'] and last['MA20']<last['MA60'] and last['MA60']<last['MA120']: score -= 15
    elif last['Close']>last['MA20'] and last['MA20']>last['MA60']: score += 10
    elif last['Close']<last['MA20'] and last['MA20']<last['MA60']: score -= 10
    if last['ADX']>30:
        if last['+DI14']>last['-DI14']: score += 10
        else: score -= 10
    elif last['ADX']>20:
        if last['+DI14']>last['-DI14']: score += 5
        else: score -= 5
    elif last['ADX']<20: score -= 5
    if last['Volume']>last['Volume_MA20']*2: score += 10
    elif last['Volume']>last['Volume_MA20']*1.5: score += 5
    elif last['Volume']<last['Volume_MA20']*0.5: score -= 5
    if not np.isnan(per):
        if per>0:
            if per<15: score += 10
            elif per<25: score += 5
            elif per<40: score -= 5
            else: score -= 10
        else: score -= 15
    if not np.isnan(market_cap):
        if market_cap>=100_000_000_000: score += 5
        elif market_cap<1_000_000_000: score -= 5
        elif market_cap<10_000_000_000: score -= 2
    if not np.isnan(forward_pe) and not np.isnan(per):
        if forward_pe<per: score += 7
        elif forward_pe>per: score -= 7
    if not np.isnan(debt_to_equity):
        if debt_to_equity<0.5: score += 8
        elif debt_to_equity<1.0: score += 4
        elif debt_to_equity<2.0: score -= 4
        else: score -= 8
    if "강력 매수" in last['TradeSignal']: score += 30
    elif "신규 매수" in last['TradeSignal'] or "매수 고려" in last['TradeSignal']: score += 15
    elif "강력 매도" in last['TradeSignal']: score -= 30
    elif "매도" in last['TradeSignal'] or "익절 매도" in last['TradeSignal'] or "매도 고려" in last['TradeSignal']: score -= 15
    elif "관망" in last['TradeSignal'] or "보유" in last['TradeSignal'] or "반전 신호" in last['TradeSignal']:
        score = max(score,40) if score>50 else min(score,60)
    if last['BB_Squeeze_Up_Breakout']: score += 20
    if last['BB_Squeeze_Down_Breakout']: score -= 20
    return max(0,min(100,score))

def get_action_and_percentage_by_score(signal, score):
    action_base, percentage = "관망", 0
    if "강력 매수" in signal: action_base, percentage = "신규 매수", 80+((score-80)*0.5 if score>80 else 0)
    elif "신규 매수" in signal: action_base, percentage = "신규 매수", 50+((score-50)*0.5 if score>50 else 0)
    elif "매수 고려" in signal: action_base, percentage = "신규 매수", max(10,20+((score-30)*0.5 if score>30 else 0))
    elif "익절 매도" in signal: action_base, percentage = "익절 매도", 50+((score-50)*0.5 if score>50 else 0)
    elif "매도" in signal: action_base, percentage = "매도", 50+((50-score)*0.5 if score<50 else 0)
    elif "매도 고려" in signal: action_base, percentage = "매도", max(10,20+((70-score)*0.5 if score<70 else 0))
    elif "강력 매도" in signal: action_base, percentage = "전량 매도", 80+((80-score)*0.5 if score<80 else 0)
    rounded_pct = max(0,min(100,round(percentage)))
    if "매수" in action_base: action_text = f"포트폴리오 분배 자산의 {rounded_pct}% 매수"
    elif "매도" in action_base: action_text = f"보유분의 {rounded_pct}% 매도"
    elif action_base=="전량 매도": action_text = f"보유분의 {rounded_pct}% 전량 매도"
    else: action_text = action_base
    return action_text, rounded_pct

# ==============================
# 5. 시각화/이메일/프롬프트 보조
# ==============================
def get_signal_symbol(signal_text):
    if "매수" in signal_text or "반등 가능성" in signal_text: return "<span style='color: green;'>▲</span>"
    if "매도" in signal_text or "하락 가능성" in signal_text or "익절 매도" in signal_text: return "<span style='color: red;'>▼</span>"
    return "<span style='color: orange;'>●</span>"

def get_display_signal_text(signal_original):
    if signal_original=="강력 매수": return "강력 상승추세 가능성"
    if signal_original=="강력 매도": return "강력 하락추세 가능성"
    return signal_original

def generate_chatgpt_prompt(ticker, rsi, macd, macd_hist, signal_line, atr, adx, k_stoch, d_stoch, cci, per, market_cap, forward_pe, debt_to_equity):
    def fval(x, nd=2): return "N/A" if np.isnan(x) else f"{x:.{nd}f}"
    vals = [
        f"RSI: {fval(rsi)}", f"MACD: {fval(macd)}, Signal: {fval(signal_line)}, Hist: {fval(macd_hist)}",
        f"ATR: {fval(atr)}", f"ADX: {fval(adx)}", f"Stoch %K: {fval(k_stoch)}, %D: {fval(d_stoch)}",
        f"CCI: {fval(cci)}", f"PER: {fval(per)}", f"시가총액: {fval(market_cap/1_000_000_000)}B",
        f"선행PER: {fval(forward_pe)}", f"부채비율: {fval(debt_to_equity)}"
    ]
    return f"{ticker}: {', '.join(vals)}"

def send_email(subject, body, to_email, from_email, password, attachments=None):
    msg = MIMEMultipart()
    msg['From'], msg['To'], msg['Subject'] = from_email, to_email, subject
    msg.attach(MIMEText(body, 'html'))
    if attachments:
        for file_path, file_name in attachments:
            try:
                with open(file_path, "rb") as f:
                    part = MIMEApplication(f.read(), Name=file_name)
                part['Content-Disposition'] = f'attachment; filename="{file_name}"'
                msg.attach(part)
            except: continue
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
    except: pass

def macro_filter(macro_data):
    vix = macro_data.get("VIX", {}).get("value", np.nan)
    us10y = macro_data.get("US10Y", {}).get("value", np.nan)
    if np.isnan(vix) or np.isnan(us10y): return "데이터 부족"
    if vix < 18:
        if us10y < 3.5: return "강세 (저변동)"
        elif us10y < 4.0: return "강세 (중변동)"
        else: return "강세 (고금리)"
    elif 18 <= vix <= 25:
        if us10y < 4.0: return "중립 (저변동)"
        else: return "중립 (고금리)"
    else:
        if us10y < 4.0: return "약세 (고변동)"
        else: return "약세 (극변동)"

# ==============================
# 6. 메인 실행 (Streamlit/E-mail)
# ==============================
def main():
    send_email_mode = "--send-email" in sys.argv
    SENDER_EMAIL = os.getenv('SENDER_EMAIL', 'your@email.com')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', 'app_password')
    RECEIVER_EMAIL = os.getenv('RECEIVER_EMAIL', 'receiver@email.com')
    STREAMLIT_APP_URL = os.getenv('STREAMLIT_APP_URL', 'https://your-app-link.com/')

    if send_email_mode:
        macro_start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        macro_data = download_macro_data(macro_start, END_DATE)
        market_condition = macro_filter(macro_data)
        email_rows, email_tech = [], []

        for ticker in TICKERS:
            data = get_ticker_history(ticker, START_DATE, END_DATE)
            if data.empty: continue
            df = calc_indicators(data)
            if df.empty: continue
            df['TradeSignal'] = ["관망"]*len(df)
            for i in range(2, len(df)):
                df.loc[df.index[i], 'TradeSignal'] = smart_signal_row(df.iloc[i], df.iloc[i-1], df.iloc[i-2])
            last, prev = df.iloc[-1], df.iloc[-2]
            info = yf.Ticker(ticker).info
            per = info.get('trailingPE', np.nan)
            mcap = info.get('marketCap', np.nan)
            fpe = info.get('forwardPE', np.nan)
            debt = info.get('debtToEquity', np.nan)
            signal = soften_signal(last['TradeSignal'], market_condition)
            df.loc[df.index[-1], 'TradeSignal'] = signal
            score = compute_recommendation_score(last, prev, per, mcap, fpe, debt)
            score = adjust_score(score, market_condition)
            action, pct = get_action_and_percentage_by_score(signal, score)
            email_rows.append({"Ticker": ticker, "Signal": signal, "추천정도": f"{score:.1f}", "추천 행동": action})
            email_tech.append(generate_chatgpt_prompt(ticker, float(last.get('RSI',np.nan)), float(last.get('MACD',np.nan)),
                        float(last.get('MACD_Hist',np.nan)), float(last.get('Signal',np.nan)), float(last.get('ATR',np.nan)),
                        float(last.get('ADX',np.nan)), float(last.get('%K',np.nan)), float(last.get('%D',np.nan)), float(last.get('CCI',np.nan)),
                        per, mcap, fpe, debt))
        email_body = "<h1>US Stock Signal Dashboard</h1>"
        email_body += "<h2>마켓 센티멘트 현황</h2>"
        email_body += f"<p>- 시장 상태: <b>{market_condition}</b></p>"
        if email_rows: email_body += "<h2>오늘의 종목별 매매 전략 요약</h2>" + pd.DataFrame(email_rows).to_html(index=False)
        if email_tech:
            ai_prompt = "| 종목 | 추천액션 | 비율(%) | 근거 요약 |\n|------|----------------|---------|-----------------------------|\n"
            ai_prompt += "\n".join(email_tech)
            email_body += "<h3>AI 프롬프트</h3><pre><code>"+ai_prompt+"</code></pre>"
        send_email(f"US Stock Signal Dashboard - {END_DATE}", email_body, RECEIVER_EMAIL, SENDER_EMAIL, SENDER_PASSWORD)

    else:
        st.set_page_config(layout="wide", page_title="US Stock Signal Dashboard")
        st.title("📈 US Stock Signal Dashboard")
        st.subheader(f"데이터 기준일: {END_DATE}")
        macro_start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        macro_data = download_macro_data(macro_start_date, END_DATE)
        market_condition = macro_filter(macro_data)
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
        with col_vix: st.markdown(f"**VIX**<br>{format_macro_metric(macro_data.get('VIX', {}))}", unsafe_allow_html=True)
        with col_market: st.markdown(f"**시장 상태**<br>{market_condition}", unsafe_allow_html=True)
        with col_us10y: st.markdown(f"**미 10년 금리**<br>{format_macro_metric(macro_data.get('US10Y', {}),'%')}", unsafe_allow_html=True)
        with col_us3m: st.markdown(f"**미 3개월 금리**<br>{format_macro_metric(macro_data.get('US3M', {}),'%')}", unsafe_allow_html=True)
        with col_sp500: st.markdown(f"**S&P 500**<br>{format_macro_metric(macro_data.get('S&P500', {}))}", unsafe_allow_html=True)
        with col_nasdaq: st.markdown(f"**NASDAQ**<br>{format_macro_metric(macro_data.get('NASDAQ', {}))}", unsafe_allow_html=True)
        with col_dxy: st.markdown(f"**달러인덱스 (DXY)**<br>{format_macro_metric(macro_data.get('DXY', {}))}", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 전체 종목별 매매 시그널 현황")

        buy_tickers, sell_tickers, hold_tickers = [], [], []
        for ticker, d in all_ticker_data.items():
            sig = d['signal']
            if "매수" in sig or "반등 가능성" in sig: buy_tickers.append(ticker)
            elif "매도" in sig or "하락 가능성" in sig or "익절 매도" in sig: sell_tickers.append(ticker)
            else: hold_tickers.append(ticker)
        col_buy, col_sell, col_hold = st.columns(3)
        with col_buy:
            st.markdown("#### ✅ 매수 시그널 종목")
            if buy_tickers:
                for t in buy_tickers:
                    st.markdown(f"- {t} {get_signal_symbol('매수')} - {TICKER_DESCRIPTIONS.get(t, '설명 없음')}", unsafe_allow_html=True)
            else: st.write("없음")
        with col_sell:
            st.markdown("#### 🔻 매도 시그널 종목")
            if sell_tickers:
                for t in sell_tickers:
                    st.markdown(f"- {t} {get_signal_symbol('매도')} - {TICKER_DESCRIPTIONS.get(t, '설명 없음')}", unsafe_allow_html=True)
            else: st.write("없음")
        with col_hold:
            st.markdown("#### 🟡 관망/보유 시그널 종목")
            if hold_tickers:
                for t in hold_tickers:
                    st.markdown(f"- {t} {get_signal_symbol('관망')} - {TICKER_DESCRIPTIONS.get(t, '설명 없음')}", unsafe_allow_html=True)
            else: st.write("없음")


        st.markdown("---")

        


        
        summary_rows, all_tech_summaries_text, all_ticker_data = [], [], {}
        for ticker in TICKERS:
            data = get_ticker_history(ticker, START_DATE, END_DATE)
            if data.empty: continue
            df = calc_indicators(data)
            if df.empty: continue
            df['TradeSignal'] = ["관망"]*len(df)
            for i in range(2, len(df)):
                df.loc[df.index[i], 'TradeSignal'] = smart_signal_row(df.iloc[i], df.iloc[i-1], df.iloc[i-2])
            last, prev = df.iloc[-1], df.iloc[-2]
            info = yf.Ticker(ticker).info
            per = info.get('trailingPE', np.nan)
            mcap = info.get('marketCap', np.nan)
            fpe = info.get('forwardPE', np.nan)
            debt = info.get('debtToEquity', np.nan)
            signal = soften_signal(last['TradeSignal'], market_condition)
            df.loc[df.index[-1], 'TradeSignal'] = signal
            score = compute_recommendation_score(last, prev, per, mcap, fpe, debt)
            score = adjust_score(score, market_condition)
            action, pct = get_action_and_percentage_by_score(signal, score)
            summary_rows.append({"Ticker": ticker, "Signal": signal, "추천정도": f"{score:.1f}", "추천 행동": action})
            all_ticker_data[ticker] = {'df':df,'last':last,'signal':signal,'score':score,'action':action,'per':per,'market_cap':mcap,'forward_pe':fpe,'debt_to_equity':debt}
            all_tech_summaries_text.append(generate_chatgpt_prompt(
                ticker, float(last.get('RSI',np.nan)), float(last.get('MACD',np.nan)),
                float(last.get('MACD_Hist',np.nan)), float(last.get('Signal',np.nan)),
                float(last.get('ATR',np.nan)), float(last.get('ADX',np.nan)),
                float(last.get('%K',np.nan)), float(last.get('%D',np.nan)), float(last.get('CCI',np.nan)),
                per, mcap, fpe, debt))
        st.subheader("📋 오늘의 종목별 매매 전략 요약")
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True)
        st.markdown("---")
        for ticker in TICKERS:
            if ticker in all_ticker_data:
                d = all_ticker_data[ticker]
                df, last, signal, action, score = d['df'], d['last'], d['signal'], d['action'], d['score']
                per, mcap, fpe, debt = d['per'], d['market_cap'], d['forward_pe'], d['debt_to_equity']
                st.write(f"**{TICKER_DESCRIPTIONS.get(ticker,'설명 없음')}**")
                st.subheader(f"📊 {ticker} 시그널 (오늘 종가: **${last['Close']:.2f}**)")
                st.markdown(f"{get_signal_symbol(signal)} {get_display_signal_text(signal)}", unsafe_allow_html=True)
                st.write(f"**추천 행동**: **{action}**")
                st.write(f"**추천정도**: **{score:.1f}/100**")
                st.markdown("---")
                st.subheader(f"{ticker} 최근 지표")
                st.markdown(f"""
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
                        <div><strong>PER:</strong> {per:.2f}</div>
                        <div><strong>시가총액:</strong> {mcap/1_000_000_000:.2f}B</div>
                        <div><strong>선행PER:</strong> {fpe:.2f}</div>
                        <div><strong>부채비율:</strong> {debt:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                display_cols = ['Close','MA20','MACD','Signal','MACD_Hist','RSI','ATR','ADX','+DI14','-DI14','Volume','Volume_MA20','TradeSignal','%K','%D','CCI','BB_Squeeze_Up_Breakout','BB_Squeeze_Down_Breakout']
                st.dataframe(df.tail(7)[display_cols])
                st.markdown("---")
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index, open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close']
                )])
                fig.update_layout(title=f'{ticker} 캔들스틱 차트', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
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

if __name__ == "__main__": main()

