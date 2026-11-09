import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title='서울 내일 미세먼지 예측', page_icon='🌫️')
FEATURES = ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM25', 'PM25_yesterday', 'month', 'weekday']

def grade(v):
    return '좋음' if v <= 15 else '보통' if v <= 35 else '나쁨' if v <= 75 else '매우나쁨'

@st.cache_resource
def load_model():
    data = pd.read_csv('seoul_daily.csv')
    model = LinearRegression()          # 노트북 비교에서 오차가 가장 작았던 모델
    model.fit(data[FEATURES], data['target'])
    return model, data

try:
    model, data = load_model()
except FileNotFoundError:
    st.error('seoul_daily.csv 파일이 없습니다. GitHub 저장소에 함께 올려 주세요.')
    st.stop()

st.title('🌫️ 서울 내일 초미세먼지 예측기')
st.caption('Kaggle Air Pollution in Seoul(2017~2019) 데이터로 학습한 선형회귀 모델 (평균오차 약 7.5㎍/㎥)')

c1, c2 = st.columns(2)
with c1:
    pm25 = st.number_input('오늘 PM2.5 (㎍/㎥)', 0.0, 500.0, 30.0)
    pm25_y = st.number_input('어제 PM2.5 (㎍/㎥)', 0.0, 500.0, 25.0)
    pm10 = st.number_input('오늘 PM10 (㎍/㎥)', 0.0, 1000.0, 45.0)
    month = st.slider('월', 1, 12, 1)
with c2:
    no2 = st.number_input('NO2 (ppm)', 0.0, 1.0, 0.03, format='%.3f')
    o3 = st.number_input('O3 (ppm)', 0.0, 1.0, 0.02, format='%.3f')
    co = st.number_input('CO (ppm)', 0.0, 50.0, 0.5, format='%.2f')
    so2 = st.number_input('SO2 (ppm)', 0.0, 1.0, 0.004, format='%.3f')
weekday = st.selectbox('요일', range(7), format_func=lambda d: '월화수목금토일'[d])

if st.button('내일 미세먼지 예측하기', type='primary'):
    x = pd.DataFrame([[so2, no2, o3, co, pm10, pm25, pm25_y, month, weekday]], columns=FEATURES)
    pred = float(model.predict(x)[0])
    g = grade(pred)
    st.metric('내일 예상 PM2.5', f'{pred:.1f} ㎍/㎥', g)
    if g in ('나쁨', '매우나쁨'):
        st.warning('😷 마스크를 챙기고 야외 활동을 줄이세요!')
    else:
        st.success('🙂 비교적 괜찮은 공기가 예상돼요.')

st.subheader('📈 학습 데이터: 일별 서울 평균 PM2.5')
st.line_chart(data.set_index('date')['PM25'])
