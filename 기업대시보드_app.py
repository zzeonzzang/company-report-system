import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
import json
import datetime
import re
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
import zipfile
import xml.etree.ElementTree as ET
import os
import openai
import ta
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import hashlib
from functools import lru_cache
import concurrent.futures
import matplotlib
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
import platform
import matplotlib.dates as mdates
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.font_manager as fm

# --- matplotlib 한글 폰트 설정 ---
# Windows 환경에서 한글 폰트 설정
if platform.system() == 'Windows':
    # 여러 한글 폰트 옵션 시도
    font_options = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'Dotum', 'Batang']
    font_found = False
    
    for font_name in font_options:
        try:
            plt.rcParams['font.family'] = font_name
            # 테스트용 텍스트로 폰트 확인
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '테스트', fontsize=12)
            plt.close(fig)
            font_found = True
            break
        except:
            continue
    
    if not font_found:
        # 폰트를 찾지 못한 경우 기본 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
else:
    # Linux/Mac 환경
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# --- 토스 스타일 CSS ---
st.markdown('''
    <style>
    html, body, [class*="css"]  {
        font-family: 'Pretendard', 'NanumGothic', 'Apple SD Gothic Neo', 'sans-serif';
        background-color: #f7fafd;
    }
    .stApp {
        background-color: #f7fafd;
    }
    .toss-card {
        background: white;
        border-radius: 18px;
        box-shadow: 0 2px 12px 0 rgba(0,0,0,0.04);
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .toss-title {
        color: #0064ff;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 0.7rem;
    }
    .toss-sub {
        color: #222;
        font-size: 1.05rem;
        margin-bottom: 0.5rem;
    }
    /* 뉴스 테이블 스타일 수정 */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background-color: white;
        border-radius: 8px;
        overflow: hidden;
        font-size: 0.9rem;
    }
    th {
        background-color: #f0f4f8;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
        color: #1a1a1a;
        border-bottom: 1px solid #e0e0e0;
        white-space: nowrap;
    }
    td {
        padding: 10px 15px;
        border-bottom: 1px solid #e0e0e0;
        color: #333;
        line-height: 1.4;
    }
    /* 감정 컬럼 너비 조정 */
    th:nth-child(2), td:nth-child(2) {
        min-width: 60px;
        max-width: 60px;
    }
    /* 날짜 컬럼 너비 조정 */
    th:nth-child(4), td:nth-child(4) {
        min-width: 90px;
        max-width: 90px;
        white-space: nowrap;
    }
    /* 분석 결과 스타일 */
    .analysis-result {
        background-color: #f8fafc;
        padding: 15px 20px;
        border-radius: 8px;
        margin-top: 15px;
        color: #333;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .analysis-highlight {
        color: #0064ff;
        font-weight: 500;
    }
    tr:hover {
        background-color: #f8fafc;
    }
    a {
        color: #0064ff;
        text-decoration: none;
        font-weight: 500;
    }
    a:hover {
        text-decoration: underline;
    }
    /* 감정분석 요약 스타일 */
    .sentiment-summary {
        display: flex;
        gap: 20px;
        margin-top: 15px;
        padding: 15px;
        background: #f8fafc;
        border-radius: 8px;
    }
    .sentiment-item {
        flex: 1;
        text-align: center;
    }
    .sentiment-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0064ff;
    }
    .sentiment-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 5px;
    }
    </style>
''', unsafe_allow_html=True)

# --- 함수: 종목코드 조회 ---
def get_stock_code_by_name(name):
    krx_list = fdr.StockListing('KRX')
    match = krx_list[krx_list['Name'] == name]
    if match.empty:
        return None
    return match['Code'].values[0]

# --- 함수: 네이버 뉴스 크롤링 및 감정분석 ---
def fetch_naver_news(company_name, max_results=50):
    client_id = st.secrets["naver_client_id"] if "naver_client_id" in st.secrets else ""
    client_secret = st.secrets["naver_client_secret"] if "naver_client_secret" in st.secrets else ""
    query = urllib.parse.quote(company_name)
    base_url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    collected = []
    for start in range(1, 1000, 100):
        if len(collected) >= max_results:
            break
        url = f"{base_url}?query={query}&display=100&start={start}&sort=date"
        req = urllib.request.Request(url, headers=headers)
        res = urllib.request.urlopen(req)
        if res.getcode() == 200:
            data = json.loads(res.read().decode('utf-8'))
            items = data.get('items', [])
            for item in items:
                title_raw = item['title']
                title_text = re.sub(r"<.*?>", "", title_raw)
                link = item['link']
                pub_date = datetime.datetime.strptime(item['pubDate'], '%a, %d %b %Y %H:%M:%S %z')
                now = datetime.datetime.now(datetime.timezone.utc)
                if (
                    company_name in title_text and
                    "news.naver.com" in link and
                    (now - pub_date).days <= 5 * 365
                ):
                    collected.append({
                        "제목": title_text,
                        "URL": link,
                        "날짜": pub_date.strftime('%Y-%m-%d')
                    })
                if len(collected) >= max_results:
                    break
    return pd.DataFrame(collected)

def crawl_naver_article(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        content_tag = soup.select_one("div#newsct_article")
        if not content_tag:
            return None
        content = content_tag.get_text(strip=True).replace("\n", " ")
        return content
    except:
        return None

def analyze_sentiment_gpt(title, content):
    client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
    prompt = f"""
다음 뉴스 기사의 감정을 분석해주세요:

제목: {title}
본문: {content[:1000]}  # 본문 길이 제한

형식:
감정: (긍정/중립/부정)
이유: (한 문장으로)
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # GPT-4에서 변경
        messages=[
            {"role": "system", "content": "뉴스 기사의 감정을 분석하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def parse_gpt_result(text):
    # 감정과 이유가 한 줄에 있을 때도 분리
    sentiment_match = re.search(r"감정\s*[:：]?\s*(긍정|중립|부정)", text)
    reason_match = re.search(r"이유\s*[:：]?\s*([^\n]*)", text)
    if not sentiment_match and "감정:" in text:
        # 감정:긍정 이유:~~~ 한 줄 케이스
        m = re.match(r"감정\s*[:：]?\s*(긍정|중립|부정)\s*이유\s*[:：]?\s*(.*)", text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    sentiment = sentiment_match.group(1).strip() if sentiment_match else "분류 실패"
    reason = reason_match.group(1).strip() if reason_match else text.strip()
    return sentiment, reason

# DART API 기반 최신 재무정보 수집 및 전처리

def safe_get(url, max_retries=3, sleep_time=1.0):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response
        except Exception as e:
            time.sleep(sleep_time)
    return None

def get_corp_code_dict(api_key):
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={api_key}"
    response = requests.get(url)
    with open("corp_code.zip", "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile("corp_code.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    tree = ET.parse("CORPCODE.xml")
    root = tree.getroot()
    corp_dict = {}
    for child in root.findall('list'):
        corp_name = child.find('corp_name').text.strip()
        stock_code = child.find('stock_code').text.strip()
        corp_code = child.find('corp_code').text.strip()
        if stock_code:
            corp_dict[stock_code] = (corp_code, corp_name)
    return corp_dict

def get_financials_by_stock_code(stock_code, start_year=2017, end_year=None):
    api_key = st.secrets["dart_api_key"]
    if end_year is None:
        end_year = datetime.datetime.now().year
    corp_code_dict = get_corp_code_dict(api_key)
    corp_info = corp_code_dict.get(stock_code)
    if not corp_info:
        st.error(f"[오류] 종목코드 '{stock_code}'에 해당하는 기업을 찾을 수 없습니다.")
        return None, None
    corp_code, corp_name = corp_info
    reprt_codes = {'Q1': '11013', 'Q2': '11012', 'Q3': '11014', 'Q4': '11011'}
    target_accounts = [
        '자산총계', '부채총계', '자본총계',
        '유동자산', '비유동자산',
        '유동부채', '비유동부채',
        '자본금', '매출액', '영업이익', '당기순이익'
    ]
    all_data = []
    current_year = pd.Timestamp.today().year
    current_month = pd.Timestamp.today().month
    for year in range(start_year, end_year + 1):
        for quarter, reprt_code in reprt_codes.items():
            # 미래 분기 데이터는 스킵
            if year == current_year and ((quarter == 'Q2' and current_month < 6) or
                                          (quarter == 'Q3' and current_month < 9) or
                                          (quarter == 'Q4' and current_month < 12)):
                continue
            url = f"https://opendart.fss.or.kr/api/fnlttSinglAcnt.json?crtfc_key={api_key}&corp_code={corp_code}&bsns_year={year}&reprt_code={reprt_code}"
            response = safe_get(url)
            if not response:
                continue
            res = response.json()
            if 'list' not in res or res.get('status') != '000':
                continue
            df = pd.DataFrame(res['list'])
            df = df[df['account_nm'].isin(target_accounts)]
            df = df[['account_nm', 'thstrm_amount']]
            df['year'] = year
            df['quarter'] = quarter
            all_data.append(df)
            time.sleep(0.7)
    if not all_data:
        st.error("유효한 재무데이터가 없습니다.")
        return None, None
    final_df = pd.concat(all_data)
    final_df = final_df.pivot_table(index=['year', 'quarter'], columns='account_nm', values='thstrm_amount', aggfunc='first').reset_index()
    return final_df, corp_name

# --- 함수: 매출액 예측 (RandomForest) ---
def sales_forecast_pipeline(df):
    """
    입력된 재무 데이터프레임(df)로 다음 분기 매출액을 예측하는 함수입니다.
    RandomForest 모델을 사용하며, 예측값, MAE, 학습된 모델, 특성 데이터(X)를 반환합니다.
    """
    # 예측에 사용할 특성(피처)와 타겟(매출액) 설정
    df = df.copy()
    df['y'] = df['매출액'].shift(-1)
    features = ['자산총계', '부채총계', '자본총계', '영업이익', '당기순이익', '매출액']
    X = df[features][:-1]
    y = df['y'][:-1]
    # 데이터 분할 (학습/테스트)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # 랜덤포레스트 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # 테스트셋 예측 및 MAE 계산
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    # 다음 분기 예측
    next_pred = model.predict([X.iloc[-1]])[0]
    return next_pred, mae, model, X

# --- 함수: 주가 예측 (LSTM & LightGBM) ---
def prepare_data_for_prediction(df, lookback=10):
    """주가 예측을 위한 데이터 전처리 및 기술적 지표 계산"""
    df = df.copy()
    
    # 기본 이동평균선
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 볼린저 밴드
    df['BB_mid'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_up'] = df['BB_mid'] + (bb_std * 2)
    df['BB_low'] = df['BB_mid'] - (bb_std * 2)
    
    # 거래량 지표
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    
    # 결측치 처리
    df = df.dropna().reset_index(drop=True)
    
    return df

def lstm_predict(df, lookback=10, epochs=50):
    df = prepare_data_for_prediction(df)
    features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI', 'BB_up', 'BB_mid', 'BB_low', 'Volume_MA5']
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # 시퀀스 데이터 생성
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 0])  # Close price
    X, y = np.array(X), np.array(y)
    
    # 학습/테스트 분할
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    # 모델 구성
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, len(features))),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # 학습
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    
    # 예측
    predictions = []
    last_sequence = scaled_data[len(scaled_data)-lookback:]
    
    for _ in range(len(X)):
        next_pred = model.predict(last_sequence.reshape(1, lookback, len(features)))
        predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1, 0] = next_pred[0, 0]
    
    # 예측값을 원래 스케일로 변환
    pred_scaled = np.zeros((len(predictions), len(features)))
    pred_scaled[:, 0] = predictions
    predictions = scaler.inverse_transform(pred_scaled)[:, 0]
    
    # 결과 데이터프레임 생성
    df_result = df.iloc[lookback:].copy()
    df_result['Predicted_LSTM'] = predictions
    
    rmse = np.sqrt(mean_squared_error(df_result['Close'], df_result['Predicted_LSTM']))
    return df_result, rmse

def lightgbm_predict(df):
    """LightGBM 모델을 사용한 주가 예측"""
    # 기술적 지표 추가
    df = prepare_data_for_prediction(df)  # 전처리 함수 호출 추가
    
    # 학습에 사용할 특성
    features = ['MA5', 'MA20', 'RSI', 'BB_up', 'BB_mid', 'BB_low', 'Volume_MA5']
    
    # 데이터 분할
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size:]
    
    # 모델 학습
    model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(train_data[features], train_data['Close'])
    
    # 예측
    df_pred = df.copy()
    df_pred['Predicted_LGBM'] = model.predict(df[features])
    
    # RMSE 계산
    rmse = np.sqrt(mean_squared_error(test_data['Close'], 
                                    df_pred.loc[test_data.index, 'Predicted_LGBM']))
    
    return df_pred, rmse

def get_naver_stock_price(code, pages=50):
    df = pd.DataFrame()
    for page in range(1, pages + 1):
        url = f"https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.select_one('table.type2')
        temp = pd.read_html(StringIO(str(table)), header=0)[0]
        df = pd.concat([df, temp], ignore_index=True)
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['날짜'])
    df['Close'] = df['종가'].astype(str).str.replace(',', '').astype(float)
    df['Volume'] = df['거래량'].astype(str).str.replace(',', '').astype(float)
    df = df[['Date', 'Close', 'Volume']].sort_values(by='Date')
    return df.reset_index(drop=True)

def analyze_technical_indicators(df):
    """
    기술적 지표를 분석하여 매수/매도/관망 신호를 반환합니다.
    """
    # 이동평균선 분석
    current_price = df['Close'].iloc[-1]
    ma_20 = df['MA20'].iloc[-1]
    ma_5 = df['MA5'].iloc[-1]
    
    if current_price > ma_5 and ma_5 > ma_20:
        ma_signal = "상승추세"
    elif current_price < ma_5 and ma_5 < ma_20:
        ma_signal = "하락추세"
    else:
        ma_signal = "횡보추세"
    
    # RSI 분석
    current_rsi = df['RSI'].iloc[-1]
    
    if current_rsi > 70:
        rsi_signal = "과매수"
    elif current_rsi < 30:
        rsi_signal = "과매도"
    else:
        rsi_signal = "중립"
    
    # 볼린저 밴드 분석
    current_bb_up = df['BB_up'].iloc[-1]
    current_bb_low = df['BB_low'].iloc[-1]
    
    if current_price > current_bb_up:
        bb_signal = "과매수"
    elif current_price < current_bb_low:
        bb_signal = "과매도"
    else:
        bb_signal = "중립"
    
    # 종합 신호
    signals = {
        'MA_Signal': ma_signal,
        'RSI': rsi_signal,
        'BB': bb_signal
    }
    
    # 매수/매도/관망 결정
    buy_signals = sum(1 for signal in [ma_signal, rsi_signal, bb_signal] 
                     if signal in ["상승추세", "과매도"])
    sell_signals = sum(1 for signal in [ma_signal, rsi_signal, bb_signal] 
                      if signal in ["하락추세", "과매수"])
    
    if buy_signals > sell_signals:
        final_signal = "매수"
    elif sell_signals > buy_signals:
        final_signal = "매도"
    else:
        final_signal = "관망"
    
    return signals, final_signal

def generate_gpt_summary(trends, signals, model_summary):
    """
    GPT를 이용한 투자 분석 요약 생성
    """
    max_retries = 3  # 최대 재시도 횟수
    
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=st.secrets["openai_api_key"])
            
            # 프롬프트 구성 (더 간결하고 명확하게)
            prompt = f"""
주식 시장 전문가로서 다음 기술적 지표를 분석해주세요:

이동평균선: {trends['MA_Signal']}
RSI: {trends['RSI']}
볼린저밴드: {trends['BB']}
종합신호: {signals['final']}

다음 순서로 분석해주세요:
1. 현재 시장 상황 (2문장)
2. 기술적 지표 분석 (3문장)
3. 투자 제안 (2문장)

쉽게 설명하고 완전한 문장으로 끝내주세요.
"""
            
            # GPT API 호출
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "주식 시장 전문가입니다. 명확하고 완전한 분석을 제공하세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800,  # 토큰 수를 500에서 800으로 더 증가
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            # 응답 반환
            result = response.choices[0].message.content.strip()
            
            # 응답이 중간에 끊겼는지 확인하고 처리
            if result and not result.endswith(('.', '!', '?')):
                # 문장이 완전하지 않으면 재시도
                if attempt < max_retries - 1:
                    time.sleep(1)  # 1초 대기 후 재시도
                    continue
                else:
                    # 마지막 시도에서도 실패하면 기본 메시지 반환
                    return f"""
현재 기술적 지표들을 종합적으로 분석한 결과입니다:

이동평균선은 {trends['MA_Signal']}를 보이고 있으며, RSI는 {trends['RSI']} 상태입니다. 
볼린저밴드 기준으로는 {trends['BB']} 구간에 위치해 있습니다.

종합적으로 {signals['final']} 포지션이 권장됩니다.
"""
            
            return result
            
        except Exception as e:
            # API 호출 실패 시 재시도
            if attempt < max_retries - 1:
                time.sleep(1)  # 1초 대기 후 재시도
                continue
            else:
                # 마지막 시도에서도 실패하면 기본 메시지 반환
                st.warning(f"GPT API 호출 중 오류가 발생했습니다: {str(e)}")
                return f"""
현재 기술적 지표들을 종합적으로 분석한 결과입니다:

이동평균선은 {trends['MA_Signal']}를 보이고 있으며, RSI는 {trends['RSI']} 상태입니다. 
볼린저밴드 기준으로는 {trends['BB']} 구간에 위치해 있습니다.

종합적으로 {signals['final']} 포지션이 권장됩니다.
"""

# --- 캐시 설정 ---
@st.cache_data(ttl=3600)  # 1시간 캐시
def get_stock_code_by_name_cached(name):
    return get_stock_code_by_name(name)

@st.cache_data(ttl=3600)
def get_naver_stock_price_cached(code, pages=50):
    return get_naver_stock_price(code, pages)

@st.cache_data(ttl=86400)  # 24시간 캐시
def get_financials_by_stock_code_cached(stock_code, start_year=2017, end_year=None):
    return get_financials_by_stock_code(stock_code, start_year, end_year)

def get_company_info(corp_code, stock_code=None):
    """DART API를 통해 기업의 기본 정보를 가져옵니다."""
    api_key = st.secrets["dart_api_key"]
    url = f"https://opendart.fss.or.kr/api/company.json?crtfc_key={api_key}&corp_code={corp_code}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == '000':
            # 업종 정보가 비어있을 경우 KRX에서 가져오기
            industry = data.get('induty', '').strip()
            if not industry and stock_code:
                try:
                    krx_list = fdr.StockListing('KRX')
                    # stock_code로 검색 (corp_code가 아님)
                    company_data = krx_list[krx_list['Code'] == stock_code]
                    if not company_data.empty:
                        industry = company_data['Industry'].iloc[0]
                    else:
                        industry = '정보없음'
                except:
                    industry = '정보없음'
            elif not industry:
                industry = '정보없음'
            
            return {
                '회사명': data.get('corp_name', '정보없음').strip(),
                '영문명': data.get('corp_name_eng', '정보없음').strip(),
                '대표자': data.get('ceo_nm', '정보없음').strip(),
                '설립일': data.get('est_dt', '정보없음').strip(),
                '본사주소': data.get('adres', '정보없음').strip(),
                '홈페이지': data.get('hm_url', '정보없음').strip(),
                '업종': industry,
                '결산월': data.get('acc_mt', '정보없음').strip()
            }
    return None

@st.cache_data(ttl=86400)  # 24시간 캐시
def get_company_info_cached(corp_code, stock_code=None):
    return get_company_info(corp_code, stock_code)

# --- 뉴스 분석 최적화 ---
def truncate_text(text, max_length=1000):
    """텍스트를 최대 길이로 제한"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def analyze_news_batch(news_items, batch_size=3):  # 배치 크기를 3으로 줄임
    results = []
    
    def process_batch(batch):
        client = openai.OpenAI(api_key=st.secrets["openai_api_key"])
        messages = []
        for item in batch:
            # 본문을 1000자로 제한
            content = truncate_text(item['content'], 1000)
            prompt = f"""
뉴스 기사 감정 분석:
제목: {item['제목']}
본문 요약: {content}

형식:
감정: (긍정/중립/부정)
이유: (한 문장으로)
"""
            messages.append({"role": "user", "content": prompt})
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # GPT-4 대신 GPT-3.5-turbo 사용
                messages=[
                    {"role": "system", "content": "뉴스 기사의 감정을 분석하는 전문가입니다."},
                    *messages
                ],
                temperature=0.2
            )
            return [choice.message.content.strip() for choice in response.choices]
        except Exception as e:
            st.error(f"GPT API 호출 중 오류 발생: {str(e)}")
            return ["감정: 중립\n이유: API 오류로 인한 기본값"] * len(batch)

    # 병렬로 뉴스 내용 수집
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(crawl_naver_article, item['URL']): item for item in news_items}
        for future in concurrent.futures.as_completed(future_to_url):
            item = future_to_url[future]
            try:
                content = future.result() or item['제목']
                item['content'] = content
            except Exception as e:
                item['content'] = item['제목']

    # 배치 처리
    for i in range(0, len(news_items), batch_size):
        batch = news_items[i:i + batch_size]
        gpt_outputs = process_batch(batch)
        
        for item, gpt_output in zip(batch, gpt_outputs):
            sentiment, reason = parse_gpt_result(gpt_output)
            results.append({
                '제목': item['제목'],
                '감정': sentiment,
                '이유': reason,
                '날짜': item['날짜']
            })
            time.sleep(0.5)  # API 호출 간 간격 추가
            
    return results

# --- 주가 데이터 최적화 ---
def get_naver_stock_price(code, pages=50):
    df = pd.DataFrame()
    
    # 최근 6개월치 데이터만 가져오기
    recent_start = pd.Timestamp.today() - pd.DateOffset(months=6)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_page = {
            executor.submit(fetch_page, code, page): page 
            for page in range(1, pages + 1)
        }
        
        for future in concurrent.futures.as_completed(future_to_page):
            try:
                temp = future.result()
                if temp is not None:
                    df = pd.concat([df, temp], ignore_index=True)
            except Exception as e:
                continue
                
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['날짜'])
    df['Close'] = df['종가'].astype(str).str.replace(',', '').astype(float)
    df['Volume'] = df['거래량'].astype(str).str.replace(',', '').astype(float)
    
    # 최근 6개월 데이터만 필터링
    df = df[df['Date'] >= recent_start]
    df = df[['Date', 'Close', 'Volume']].sort_values(by='Date')
    
    return df.reset_index(drop=True)

def fetch_page(code, page):
    url = f"https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.select_one('table.type2')
        if table:
            return pd.read_html(StringIO(str(table)), header=0)[0]
    except Exception:
        return None

def generate_pdf_report(company_name, company_info, df_news_summary, financial_summary, stock_prediction_summary):
    """PDF 보고서를 생성합니다."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # 폰트 설정
    if platform.system() == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    else:
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    
    try:
        pdfmetrics.registerFont(TTFont('Malgun', font_path))
        font_name = 'Malgun'
    except:
        font_name = 'Helvetica'

    # 제목
    c.setFont(font_name, 20)
    c.drawString(50, height - 50, f"{company_name} 기업 분석 리포트")
    c.setFont(font_name, 10)
    c.drawString(50, height - 70, f"생성일자: {datetime.datetime.now().strftime('%Y-%m-%d')}")

    # 구분선
    c.line(50, height - 80, width - 50, height - 80)

    # 기업 정보
    y = height - 120
    c.setFont(font_name, 14)
    c.drawString(50, y, "1. 기업 정보")
    y -= 30
    c.setFont(font_name, 10)
    for key, value in company_info.items():
        if key not in ['영문명', '결산월']:  # 일부 정보는 제외
            c.drawString(70, y, f"{key}: {value}")
            y -= 20

    # 뉴스 분석 요약
    y -= 30
    c.setFont(font_name, 14)
    c.drawString(50, y, "2. 최근 뉴스 동향")
    y -= 30
    c.setFont(font_name, 10)
    c.drawString(70, y, f"분석된 뉴스 기사 수: {df_news_summary['total']}건")
    y -= 20
    c.drawString(70, y, f"긍정: {df_news_summary['positive']}건 ({df_news_summary['positive_ratio']}%)")
    y -= 20
    c.drawString(70, y, f"중립: {df_news_summary['neutral']}건 ({df_news_summary['neutral_ratio']}%)")
    y -= 20
    c.drawString(70, y, f"부정: {df_news_summary['negative']}건 ({df_news_summary['negative_ratio']}%)")

    # 재무 정보
    y -= 40
    c.setFont(font_name, 14)
    c.drawString(50, y, "3. 재무 분석")
    y -= 30
    c.setFont(font_name, 10)
    c.drawString(70, y, f"다음 분기 매출액 예측: {financial_summary['next_prediction']:.2f}조원")
    y -= 20
    c.drawString(70, y, f"전분기 대비: {financial_summary['change_rate']}")

    # 주가 예측
    y -= 40
    c.setFont(font_name, 14)
    c.drawString(50, y, "4. 주가 분석")
    y -= 30
    c.setFont(font_name, 10)
    c.drawString(70, y, f"예측 모델: {stock_prediction_summary['better_model']}")
    y -= 20
    c.drawString(70, y, f"RMSE: {stock_prediction_summary['rmse']:.2f}")

    # AI 투자 분석
    y -= 40
    c.setFont(font_name, 14)
    c.drawString(50, y, "5. AI 투자 분석")
    y -= 30
    c.setFont(font_name, 10)
    
    # 문자열을 적절한 길이로 나누어 여러 줄로 표시
    lines = [stock_prediction_summary['gpt_summary'][i:i+80] for i in range(0, len(stock_prediction_summary['gpt_summary']), 80)]
    for line in lines:
        c.drawString(70, y, line)
        y -= 20

    c.save()
    buffer.seek(0)
    return buffer

# --- 클래스 정의 ---
@dataclass
class JobPosting:
    """채용 공고 데이터 구조"""
    title: str
    company: str
    location: str
    experience: str
    source: str
    url: str
    posted_date: str
    summary: str = ""

@dataclass
class CompanyInfo:
    """상장사 기본 정보"""
    company_name: str
    stock_code: str
    market: str
    sector: str
    market_cap: Optional[int]
    employee_count: Optional[int]
    revenue: Optional[int]
    listing_date: Optional[str]
    official_website: Optional[str] = None

class KoreanListedCompanies:
    """한국 상장사 정보 관리"""
    def __init__(self):
        self.companies_db = self._load_listed_companies()

    def _load_listed_companies(self) -> Dict[str, CompanyInfo]:
        """상장사 정보 로드"""
        companies = {
            "삼성전자": CompanyInfo(
                "삼성전자", "005930", "KOSPI", "반도체",
                400000000, 267937, 279651000, "1975-06-11",
                "https://www.samsung.com/sec/about-us/careers/"
            ),
            "SK하이닉스": CompanyInfo(
                "SK하이닉스", "000660", "KOSPI", "반도체",
                80000000, 29415, 44819000, "1996-12-26",
                "https://careers.skhynix.com/"
            ),
            "현대자동차": CompanyInfo(
                "현대자동차", "005380", "KOSPI", "자동차",
                30000000, 70439, 117611000, "1974-10-02",
                "https://careers.hyundai.com/"
            ),
            "기아": CompanyInfo(
                "기아", "000270", "KOSPI", "자동차",
                25000000, 52713, 89094000, "1973-07-10",
                "https://careers.kia.com/"
            ),
            "LG에너지솔루션": CompanyInfo(
                "LG에너지솔루션", "373220", "KOSPI", "배터리",
                70000000, 26586, 27307000, "2022-01-27",
                "https://www.lgensol.com/careers"
            ),
            "NAVER": CompanyInfo(
                "NAVER", "035420", "KOSPI", "인터넷",
                35000000, 3793, 8487000, "2002-10-29",
                "https://career.navercorp.com/"
            ),
            "카카오": CompanyInfo(
                "카카오", "035720", "KOSPI", "인터넷",
                25000000, 4479, 6671000, "2017-07-10",
                "https://careers.kakao.com/"
            )
        }
        return companies

    def get_company_info(self, company_name: str) -> Optional[CompanyInfo]:
        return self.companies_db.get(company_name)

def generate_job_postings(company_name: str) -> List[JobPosting]:
    """기본 채용 정보 생성"""
    jobs = []
    basic_positions = [
        ("신입사원 공개채용", "신입", "본사"),
        ("경력직 채용", "3년 이상", "전국"),
        ("연구개발직", "석사 이상", "연구소"),
        ("영업/마케팅", "2년 이상", "전국"),
        ("IT/개발직", "경력무관", "본사")
    ]

    for title, exp, loc in basic_positions:
        job = JobPosting(
            title=f"{company_name} {title}",
            company=company_name,
            location=loc,
            experience=exp,
            source="추정정보",
            url=f"https://careers.{company_name.lower()}.com/",
            posted_date=datetime.now().strftime('%Y-%m-%d'),
            summary=f"{company_name}의 일반적인 채용 형태"
        )
        jobs.append(job)

    return jobs

# --- Streamlit 대시보드 ---
st.title('📊 기업 종합 대시보드')
st.markdown('<div class="toss-card"><span class="toss-title">기업 리포트</span><br>아래에 기업명을 입력하고 분석을 시작하세요.</div>', unsafe_allow_html=True)

with st.form(key='search_form'):
    company_name = st.text_input('기업명 입력', value='삼성전자')
    submitted = st.form_submit_button('분석 시작')

if submitted:
    code = get_stock_code_by_name_cached(company_name)
    if not code:
        st.error('해당 기업명을 찾을 수 없습니다.')
        st.stop()

    # 기업 기본 정보 표시
    st.markdown('<div class="toss-card"><span class="toss-title">🏢 기업 정보</span>', unsafe_allow_html=True)
    
    # corp_code 가져오기
    corp_dict = get_corp_code_dict(st.secrets["dart_api_key"])
    if code in corp_dict:
        corp_code = corp_dict[code][0]
        company_info = get_company_info_cached(corp_code, code)
        
        if company_info:
            # 기업 정보를 3개의 열로 표시
            col1, col2, col3 = st.columns(3)
            
            card_style = '''
                padding: 15px;
                background: white;
                border-radius: 8px;
                margin: 5px;
                height: 85px;
                overflow: hidden;
            '''
            
            label_style = '''
                color: #666;
                font-size: 0.8rem;
                margin-bottom: 5px;
            '''
            
            value_style = '''
                color: #333;
                font-size: 0.95rem;
                font-weight: 500;
                overflow: hidden;
                text-overflow: ellipsis;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
            '''
            
            with col1:
                st.markdown(f'''
                    <div style="{card_style}">
                        <div style="{label_style}">회사명</div>
                        <div style="{value_style}">{company_info['회사명']}</div>
            </div>
                    <div style="{card_style}">
                        <div style="{label_style}">대표자</div>
                        <div style="{value_style}">{company_info['대표자']}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                    <div style="{card_style}">
                        <div style="{label_style}">설립일</div>
                        <div style="{value_style}">{company_info['설립일']}</div>
                    </div>
                    <div style="{card_style}">
                        <div style="{label_style}">업종</div>
                        <div style="{value_style}">{company_info['업종']}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                    <div style="{card_style}">
                        <div style="{label_style}">본사주소</div>
                        <div style="{value_style}">{company_info['본사주소']}</div>
                    </div>
                    <div style="{card_style}">
                        <div style="{label_style}">홈페이지</div>
                        <div style="{value_style}"><a href="{company_info['홈페이지']}" target="_blank" style="color: #0064ff; text-decoration: none;">{company_info['홈페이지'].replace('http://', '').replace('https://', '')}</a></div>
                    </div>
                ''', unsafe_allow_html=True)
        else:
            st.warning("기업 기본 정보를 가져오는데 실패했습니다.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 채용 정보 추가 ---
    st.markdown("### 👥 채용 정보")
    
    # 기업 정보 가져오기
    listed_companies = KoreanListedCompanies()
    company_info = listed_companies.get_company_info(company_name)

    if company_info:
        # 기업 기본 정보 표시
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            #### 기업 현황
            - 임직원 수: {company_info.employee_count:,}명
            - 상장시장: {company_info.market}
            - 업종: {company_info.sector}
            - 상장일: {company_info.listing_date}
            """)
        
        with col2:
            if company_info.official_website:
                st.markdown(f"""
                #### 채용 사이트
                🔗 [공식 채용 페이지 바로가기]({company_info.official_website})
                """)

        # 채용 공고 표시
        st.markdown("#### 최근 채용 공고")
        job_postings = generate_job_postings(company_name)
        
        # 채용 공고를 카드 형태로 표시
        for i in range(0, len(job_postings), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                job = job_postings[i]
                st.markdown(f"""
                <div style="padding: 15px; background: white; border-radius: 8px; margin: 5px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #333;">{job.title}</div>
                    <div style="color: #666; margin: 8px 0;">
                        📍 {job.location} | 💼 {job.experience}
                    </div>
                    <div style="color: #888; font-size: 0.9rem;">{job.posted_date}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if i + 1 < len(job_postings):
                with col2:
                    job = job_postings[i + 1]
                    st.markdown(f"""
                    <div style="padding: 15px; background: white; border-radius: 8px; margin: 5px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                        <div style="font-size: 1.1rem; font-weight: 600; color: #333;">{job.title}</div>
                        <div style="color: #666; margin: 8px 0;">
                            📍 {job.location} | 💼 {job.experience}
                        </div>
                        <div style="color: #888; font-size: 0.9rem;">{job.posted_date}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # 채용 플랫폼 링크
        st.markdown("""
        #### 채용 플랫폼 바로가기
        - [사람인](https://www.saramin.co.kr/)
        - [잡코리아](https://www.jobkorea.co.kr/)
        - [원티드](https://www.wanted.co.kr/)
        """)

    else:
        st.warning("해당 기업의 채용 정보를 찾을 수 없습니다.")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 재무정보 표시 수정 ---
    st.markdown('<div class="toss-card"><span class="toss-title">📋 최근 재무정보</span>', unsafe_allow_html=True)
    with st.spinner('재무정보를 가져오는 중입니다...'):
        df_fin, corp_name = get_financials_by_stock_code_cached(code)
        if df_fin is not None and not df_fin.empty:
            df_fin_sorted = df_fin.sort_values(['year', 'quarter'], ascending=[False, False]).copy()
            df_fin_sorted['year'] = df_fin_sorted['year'].astype(str).str.replace(',', '').str.extract(r'(\d{4})')[0]
            for col in df_fin_sorted.columns:
                if col not in ['year', 'quarter']:
                    df_fin_sorted[col] = df_fin_sorted[col].apply(
                        lambda x: f"{int(float(str(x).replace(',', ''))):,}" 
                        if pd.notnull(x) and str(x).replace(',', '').replace('.', '', 1).isdigit() 
                        else x
                    )
            st.write(df_fin_sorted.head(4).sort_values(['year', 'quarter'], ascending=[True, True]))
        else:
            st.warning("재무정보를 가져오는데 실패했습니다. 잠시 후 다시 시도해주세요.")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 매출액 예측 ---
    st.markdown('<div class="toss-card"><span class="toss-title">📈 매출액 예측</span>', unsafe_allow_html=True)

    # Q4 매출 누계 보정
    quarter_order = ['Q1', 'Q2', 'Q3', 'Q4']
    df_fin = df_fin.sort_values(['year', 'quarter']).reset_index(drop=True)
    df_corrected = df_fin.copy()

    # 예측에 사용되는 모든 컬럼을 float으로 변환합니다.
    # (콤마가 포함된 문자열을 숫자로 변환)
    feature_cols = ['자산총계', '부채총계', '자본총계', '영업이익', '당기순이익', '매출액']
    for col in feature_cols:
        df_corrected[col] = df_corrected[col].apply(lambda x: float(str(x).replace(',', '')) if pd.notnull(x) else x)

    # 다음 분기 정보 계산
    last_row = df_corrected.iloc[-1]
    last_year = int(last_row['year'])
    last_quarter = str(last_row['quarter'])
    if last_quarter == 'Q4':
        next_year = last_year + 1
        next_quarter = 'Q1'
    else:
        next_year = last_year
        next_quarter = quarter_order[quarter_order.index(last_quarter) + 1]

    # 매출액 예측을 위한 데이터 준비
    X = df_corrected[feature_cols].copy()
    y = df_corrected['매출액'].shift(-1)  # 다음 분기 매출액

    # NaN 제거
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 다음 분기 예측
    next_pred = model.predict([X.iloc[-1]])[0]
    next_pred_jo = next_pred / 1e12  # 예측값을 조원 단위로 변환
    prev_sales = df_corrected['매출액'].iloc[-1] / 1e12  # 이전 분기 매출액도 조원 단위

    # 증감률 계산
    change_rate = ((next_pred_jo - prev_sales) / prev_sales) * 100
    change_text = f"{abs(change_rate):.1f}% {'증가' if change_rate > 0 else '감소'}"
    trend_emoji = '▲' if change_rate > 0 else '▼'

    # 예측 결과 표시
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'''
            <div style="color: #666; font-size: 0.9rem;">다음분기 매출액 예측 ({next_year} {next_quarter})</div>
            <div style="color: #333; font-size: 1.4rem; font-weight: 600; margin-top: 5px;">{next_pred_jo:.2f}조원</div>
            <div style="color: {('#0064ff' if change_rate > 0 else '#ff6b6b')}; font-size: 0.9rem; margin-top: 5px;">
                {trend_emoji} 전분기 대비 {change_text}
            </div>
        ''', unsafe_allow_html=True)

    # 특성 중요도 분석 및 시각화
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    })
    feature_importance_df['Importance (%)'] = (feature_importance_df['Importance'] * 100).round(2)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # 표와 그래프를 같은 줄에 표시 (col1: 표, col2: 그래프)
    col1, col2 = st.columns([1,2])
    with col1:
        st.dataframe(feature_importance_df[['Feature', 'Importance (%)']])
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance (%)'], color='cornflowerblue')
        ax.invert_yaxis()  # 가장 중요한 항목이 위로 오도록 뒤집기
        ax.set_title('매출 예측에 기여한 중요도 (%)')
        ax.set_xlabel('중요도 (%)')
        ax.set_ylabel('재무 항목')
        st.pyplot(fig)

    # 가장 중요한 항목 설명 문구 추가 (토스 스타일 적용)
    most_important = feature_importance_df.iloc[0]
    st.markdown(f'''
        <div class="analysis-result">
            이번 분기 매출 예측에 가장 큰 영향을 준 항목은 <span class="analysis-highlight">'{most_important['Feature']}'</span>입니다.
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 주가 예측 ---
    st.markdown('<div class="toss-card"><span class="toss-title">💹 주가 예측</span>', unsafe_allow_html=True)

    with st.spinner('주가 예측 모델을 학습하고 있습니다...'):
        # 데이터 준비
        df_stock = get_naver_stock_price(code)
        
        # 기술적 지표 계산
        df_stock['MA5'] = df_stock['Close'].rolling(window=5).mean()
        df_stock['MA20'] = df_stock['Close'].rolling(window=20).mean()
        
        # RSI 계산
        delta = df_stock['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_stock['RSI'] = 100 - (100 / (1 + rs))
        
        # 볼린저 밴드 계산
        df_stock['BB_mid'] = df_stock['Close'].rolling(window=20).mean()
        bb_std = df_stock['Close'].rolling(window=20).std()
        df_stock['BB_up'] = df_stock['BB_mid'] + (bb_std * 2)
        df_stock['BB_low'] = df_stock['BB_mid'] - (bb_std * 2)
        
        # 거래량 이동평균
        df_stock['Volume_MA5'] = df_stock['Volume'].rolling(window=5).mean()
        
        # 결측치 제거
        df_stock = df_stock.dropna().reset_index(drop=True)
        
        # LightGBM 예측
        df_lgbm, rmse_lgbm = lightgbm_predict(df_stock.copy())
        
        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 볼린저 밴드 (회색 영역으로 표시)
        ax.fill_between(df_stock['Date'], df_stock['BB_up'], df_stock['BB_low'], 
                       color='gray', alpha=0.2, label='볼린저밴드')
        
        # 이동평균선
        ax.plot(df_stock['Date'], df_stock['MA5'], 
                label='5일 이동평균', color='#ffd700', linewidth=1, linestyle='--')
        ax.plot(df_stock['Date'], df_stock['MA20'], 
                label='20일 이동평균', color='#ff8c00', linewidth=1, linestyle='--')
        
        # 실제 주가
        ax.plot(df_stock['Date'], df_stock['Close'], 
                label='실제 주가', color='#0064ff', linewidth=2)
        
        # LightGBM 예측
        ax.plot(df_lgbm['Date'], df_lgbm['Predicted_LGBM'],
                label='예측 주가', linestyle='--', color='#ff6b6b', linewidth=1.5)
        
        # x축 날짜 포맷 설정
        plt.gcf().autofmt_xdate()
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        
        # 그래프 스타일링
        ax.set_facecolor('#f7fafd')
        fig.patch.set_facecolor('#f7fafd')
        ax.tick_params(colors='#222')
        ax.grid(True, alpha=0.3)
        
        # 범례 스타일링 (2줄로 표시)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.05), 
                 ncol=2, frameon=False, fontsize=8)
        
        # 축 레이블
        ax.set_xlabel('날짜', fontsize=10, color='#666')
        ax.set_ylabel('주가 (원)', fontsize=10, color='#666')
        
        # 여백 조정
        plt.tight_layout()
        st.pyplot(fig)
        
        # 기술적 지표 분석
        signals, final_signal = analyze_technical_indicators(df_stock)
        
        # 기술적 지표 카드 표시 (한 줄에 3개)
        st.markdown('''
<div style="display: flex; justify-content: space-between; gap: 20px; margin: 20px 0;">
    <div style="flex: 1; padding: 15px; background: white; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">이동평균선</div>
        <div style="font-size: 1.2rem; color: #0064ff; font-weight: 600;">{}</div>
    </div>
    <div style="flex: 1; padding: 15px; background: white; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">RSI</div>
        <div style="font-size: 1.2rem; color: #0064ff; font-weight: 600;">{}</div>
    </div>
    <div style="flex: 1; padding: 15px; background: white; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">볼린저밴드</div>
        <div style="font-size: 1.2rem; color: #0064ff; font-weight: 600;">{}</div>
    </div>
</div>
'''.format(signals['MA_Signal'], signals['RSI'], signals['BB']), unsafe_allow_html=True)
        
        # AI 투자 분석 요약 (모델 성능 제외)
        summary = generate_gpt_summary(
            trends={'MA_Signal': signals['MA_Signal'], 'RSI': signals['RSI'], 'BB': signals['BB']},
            signals={'final': final_signal},
            model_summary=""  # 모델 성능 정보 제외
        )
        st.markdown(f'''
<div class="analysis-result" style="margin-top: 20px; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 12px 0 rgba(0,0,0,0.04);">
    <div style="font-size: 1.3rem; color: #0064ff; font-weight: 700; margin-bottom: 15px;">💡 AI 투자 분석</div>
    <div style="color: #333; line-height: 1.6; font-size: 1rem;">{summary}</div>
</div>
''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)