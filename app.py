import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

@st.cache_resource
def load_model():
    data = pd.read_csv('seoul_daily.csv')
    model = LinearRegression()
    model.fit(data[FEATURES], data['target'])
    return model, data

try:
    model, data = load_model()
except FileNotFoundError:
    st.error('seoul_daily.csv 파일이 없습니다. GitHub 저장소에 함께 올려 주세요.')
    st.stop()

pm25 = st.number_input('오늘 PM2.5 (㎍/㎥)', 0.0, 500.0, 30.0)
month = st.slider('월', 1, 12, 1)
weekday = st.selectbox('요일', range(7), format_func=lambda d: '월화수목금토일'[d])

if st.button('내일 미세먼지 예측하기', type='primary'):
    x = pd.DataFrame([[so2, no2, o3, co, pm10, pm25, pm25_y, month, weekday]], columns=FEATURES)
    pred = float(model.predict(x)[0])
    st.metric('내일 예상 PM2.5', f'{pred:.1f} ㎍/㎥', grade(pred))
