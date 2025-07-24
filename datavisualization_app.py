import streamlit as st
st.set_page_config(page_title="학교도서관 이용자수 영향 요인 분석", layout="wide")

st.title("📚 전국 및 서울시 학교도서관 이용자수 영향 요인 분석")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# ✅ 한글 폰트 설정
# ---------------------------
font_path = os.path.join(os.getcwd(), "fonts", "NanumGothicCoding.ttf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['font.family'] = font_prop.get_name()
    mpl.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None

# ---------------------------
# ✅ 데이터 불러오기
# ---------------------------
@st.cache_data
def load_data():
    df1 = pd.read_csv("문화체육관광부_국가도서관통계_전국학교도서관통계_20231231.csv", encoding="cp949")
    df2 = pd.read_csv("서울시 학교별 학교도서관 현황.csv", encoding="cp949")
    df3 = pd.read_csv("학교도서관현황_20250717223352.csv", encoding="cp949")
    return df1, df2, df3

df1, df2, df3 = load_data()

# ---------------------------
# ✅ 데이터 전처리 (학교 단위)
# ---------------------------
st.subheader("✅ 데이터 전처리 상태")
st.markdown("전국 및 서울시 학교 단위 데이터를 이용자수 중심으로 정리합니다.")

# df1: 주요 변수 추출 및 결측치 제거
df1_clean = df1[['도서관명', '장서수(인쇄)', '사서수', '대출자수', '대출권수', '도서예산(자료구입비)']].copy()
df1_clean.dropna(inplace=True)

# df2: 주요 변수 추출 및 결측치 제거
df2_clean = df2[['학교명', '도서관대여학생수', '1인당대출자료수']].copy()
df2_clean = df2_clean.dropna(subset=['도서관대여학생수'])

# 병합 전 이름 통일 (학교명 기준)
df_merge = pd.merge(
    df1_clean, df2_clean,
    left_on='도서관명', right_on='학교명',
    how='inner'
)

st.write("📄 병합된 데이터 샘플", df_merge.head())

# ---------------------------
# ✅ 학교 단위 분석: 변수 중요도
# ---------------------------
st.subheader("🔍 학교 단위: 변수 중요도 분석")

st.markdown("학교 단위에서 **대출자수(이용자수)**에 영향을 주는 주요 요인을 분석합니다.")

# ✅ 운영비예산액, 자료구입비예산액 제거 후 변수 선택
X = df_merge[['장서수(인쇄)', '사서수', '도서예산(자료구입비)', '1인당대출자료수']].copy()
y = df_merge['대출자수']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.markdown(f"✅ **예측 오차(MSE)**: `{mse:,.0f}` | **정확도(R²)**: `{r2:.4f}`")

# 변수 중요도 시각화
importance = pd.Series(model.feature_importances_, index=X.columns)
fig, ax = plt.subplots(figsize=(8, 5))
importance.sort_values().plot.barh(ax=ax, color='skyblue')
ax.set_title("학교 단위: RandomForest 변수 중요도", fontproperties=font_prop)
ax.set_xlabel("중요도", fontproperties=font_prop)
ax.set_ylabel("변수", fontproperties=font_prop)
ax.set_yticklabels(importance.sort_values().index, fontproperties=font_prop)
st.pyplot(fig)

# ---------------------------
# ✅ 전국 추세 분석(df3 활용)
# ---------------------------
st.subheader("📈 전국 학교도서관 연도별 추세 분석")

st.markdown("전국 학교도서관의 **연도별 1관당 방문자수** 변화를 학교급(초,중,고)별로 비교합니다.")

# ✅ df3 전처리: 연도별로 1관당 방문자수 컬럼만 추출
df3_clean = df3[df3['학교급별(1)'].isin(['초등학교', '중학교', '고등학교'])].copy()

# 연도별 방문자수 데이터만 선택
visit_cols = [col for col in df3_clean.columns if ".3" in col]  # e.g., 2011.3, 2023.3
df3_visit = df3_clean[['학교급별(1)'] + visit_cols].copy()
df3_visit = df3_visit.melt(id_vars='학교급별(1)', var_name='연도', value_name='1관당 방문자수')

# 연도 정리
df3_visit['연도'] = df3_visit['연도'].str.replace('.3', '', regex=False).astype(int)
df3_visit['1관당 방문자수'] = df3_visit['1관당 방문자수'].astype(float)

# 꺾은선 그래프
fig2, ax2 = plt.subplots(figsize=(10, 6))
for school_type in df3_visit['학교급별(1)'].unique():
    data = df3_visit[df3_visit['학교급별(1)'] == school_type]
    ax2.plot(data['연도'], data['1관당 방문자수'], marker='o', label=school_type)

ax2.set_title("연도별 학교급별 1관당 방문자수 추세", fontproperties=font_prop)
ax2.set_xlabel("연도", fontproperties=font_prop)
ax2.set_ylabel("1관당 방문자수", fontproperties=font_prop)
ax2.legend(prop=font_prop)
st.pyplot(fig2)

# ---------------------------
# ✅ 데이터 테이블 출력
# ---------------------------
st.subheader("📄 분석 데이터 테이블")
st.markdown("학교 단위 분석 및 전국 추세 분석에 사용된 원천 데이터입니다.")
st.dataframe(df_merge)
