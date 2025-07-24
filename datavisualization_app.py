import streamlit as st
st.set_page_config(page_title="학교도서관 이용자수 영향 요인 분석", layout="wide")

st.title("📚 전국 및 서울시 학교도서관 이용자수 분석 및 예측")

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
# 📊 전체 연도 확률 분포표 및 기댓값·분산·표준편차
# ---------------------------
st.subheader("📊 전체 연도 확률 분포표 및 기댓값·분산·표준편차")
st.markdown("""
2011년부터 2023년까지의 학교급별 1관당 방문자수를 **하나의 확률 분포**로 보고 계산했습니다.  
아래 표는 각 연도·학교급의 방문자수와 그 비율(확률 P)을 나타냅니다.
""")

visit_cols = [col for col in df3.columns if ".3" in col]
df_all_visit = df3[df3['학교급별(1)'].isin(['초등학교', '중학교', '고등학교'])][['학교급별(1)'] + visit_cols].copy()
df_all_visit = df_all_visit.melt(id_vars='학교급별(1)', var_name='연도', value_name='1관당 방문자수')

df_all_visit['연도'] = df_all_visit['연도'].str.replace('.3', '', regex=False).astype(int)
df_all_visit['1관당 방문자수'] = df_all_visit['1관당 방문자수'].astype(float)

total_all = df_all_visit['1관당 방문자수'].sum()
df_all_visit['확률(P)'] = df_all_visit['1관당 방문자수'] / total_all

E_X_all = (df_all_visit['1관당 방문자수'] * df_all_visit['확률(P)']).sum()
E_X2_all = ((df_all_visit['1관당 방문자수']**2) * df_all_visit['확률(P)']).sum()
V_X_all = E_X2_all - (E_X_all**2)
Std_X_all = np.sqrt(V_X_all)

st.dataframe(df_all_visit.head(), use_container_width=True, height=200)

with st.expander("📐 풀이 자세히 보기"):
    st.markdown("""
    **✔ 기댓값(E[X])**  
    각 방문자수 × 확률을 모두 더한 값입니다.
    """)
    E_steps = [f"({row['1관당 방문자수']:,.0f}×{row['확률(P)']:.4f})" for _, row in df_all_visit.iterrows()]
    st.code("E[X] = " + " + ".join(E_steps) + f"\n= {E_X_all:,.2f}")

    st.markdown("""
    **✔ 분산(V[X])**  
    각 방문자수의 제곱 × 확률을 모두 더한 값에서, (E[X])²을 뺀 값입니다.
    """)
    Var_steps = [f"({row['1관당 방문자수']:,.0f}²×{row['확률(P)']:.4f})" for _, row in df_all_visit.iterrows()]
    st.code("V[X] = " + " + ".join(Var_steps) +
            f"\n- (E[X])²\n= {E_X2_all:,.2f} - ({E_X_all:,.2f})²\n= {V_X_all:,.2f}")

    st.markdown("""
    **✔ 표준편차(σ[X])**  
    분산의 양의 제곱근입니다.
    """)
    st.code(f"σ[X] = √V[X] = √{V_X_all:,.2f} ≈ {Std_X_all:,.2f}")

st.success(f"✅ **기댓값(E[X]) ≈ {E_X_all:,.2f}명**")
st.info(f"✅ **분산(V[X]) ≈ {V_X_all:,.2f}**")
st.warning(f"✅ **표준편차(σ[X]) ≈ {Std_X_all:,.2f}명**")

# ---------------------------
# ✅ 학교 단위 데이터 전처리
# ---------------------------
st.subheader("✅ 데이터 전처리 및 병합 상태")
st.markdown("전국 및 서울시 학교 단위 데이터를 이용자수 중심으로 정리하여 분석합니다.")

df1_clean = df1[['도서관명', '장서수(인쇄)', '사서수', '대출자수', '대출권수', '도서예산(자료구입비)']].copy()
df1_clean.dropna(inplace=True)

df2_clean = df2[['학교명', '도서관대여학생수', '1인당대출자료수']].copy()
df2_clean = df2_clean.dropna(subset=['도서관대여학생수'])

df_merge = pd.merge(
    df1_clean, df2_clean,
    left_on='도서관명', right_on='학교명',
    how='inner'
)

st.dataframe(df_merge.head(), use_container_width=True, height=200)

# ---------------------------
# 🔍 학교 단위: 변수 중요도 분석
# ---------------------------
st.subheader("🔍 학교 단위: 변수 중요도 분석")
st.markdown("학교 단위에서 **대출자수(이용자수)**에 영향을 주는 주요 요인을 분석했습니다.")

df_merge_renamed = df_merge.rename(columns={
    '1인당대출자료수': '1인당\n대출자료수',
    '장서수(인쇄)': '장서수\n(인쇄)',
    '도서예산(자료구입비)': '도서예산\n(자료구입비)'
})

X = df_merge_renamed[['장서수\n(인쇄)', '사서수', '도서예산\n(자료구입비)', '1인당\n대출자료수']].copy()
y = df_merge_renamed['대출자수']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.success(f"✅ 예측 오차(MSE): **{mse:,.0f}** | 정확도(R²): **{r2:.4f}**")

importance = pd.Series(model.feature_importances_, index=X.columns)

fig, ax = plt.subplots(figsize=(6, 3.5))
importance.sort_values().plot.barh(ax=ax, color='skyblue')
ax.set_title("학교 단위: RandomForest 변수 중요도", fontproperties=font_prop, fontsize=12)
ax.set_xlabel("중요도", fontproperties=font_prop, fontsize=10)
ax.set_ylabel("변수", fontproperties=font_prop, fontsize=10)
ax.set_yticklabels(importance.sort_values().index, fontproperties=font_prop, fontsize=10)
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# ---------------------------
# 📈 전국 학교도서관 연도별 추세 분석
# ---------------------------
st.subheader("📈 전국 학교도서관 연도별 추세 분석")
st.markdown("전국 학교도서관의 **연도별 1관당 방문자수** 변화를 학교급(초,중,고)별로 비교했습니다.")

df3_clean = df3[df3['학교급별(1)'].isin(['초등학교', '중학교', '고등학교'])].copy()
visit_cols = [col for col in df3_clean.columns if ".3" in col]
df3_visit = df3_clean[['학교급별(1)'] + visit_cols].copy()
df3_visit = df3_visit.melt(id_vars='학교급별(1)', var_name='연도', value_name='1관당 방문자수')

df3_visit['연도'] = df3_visit['연도'].str.replace('.3', '', regex=False).astype(int)
df3_visit['1관당 방문자수'] = df3_visit['1관당 방문자수'].astype(float)

color_map = {'초등학교': 'green', '중학교': 'orange', '고등학교': 'blue'}

fig2, ax2 = plt.subplots(figsize=(6, 3.5))
for school_type in ['초등학교', '중학교', '고등학교']:
    data = df3_visit[df3_visit['학교급별(1)'] == school_type]
    ax2.plot(data['연도'], data['1관당 방문자수'],
             color=color_map[school_type],
             linestyle='-',
             marker='o',
             linewidth=2,
             label=school_type)

ax2.set_title("연도별 학교급별 1관당 방문자수 추세", fontproperties=font_prop)
ax2.set_xlabel("연도", fontproperties=font_prop)
ax2.set_ylabel("1관당 방문자수", fontproperties=font_prop)
ax2.legend(prop=font_prop, loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
st.pyplot(fig2, use_container_width=False)

# ---------------------------
# 🔍 중·고등학교 확대 비교
# ---------------------------
st.subheader("🔍 중·고등학교 확대 비교")
st.markdown("중학교와 고등학교의 **세부 추세 비교**를 위해 별도의 확대 그래프를 추가했습니다.")

df_middle_high = df3_visit[df3_visit['학교급별(1)'].isin(['중학교', '고등학교'])]

fig3, ax3 = plt.subplots(figsize=(6, 3.5))
for school_type in ['중학교', '고등학교']:
    data = df_middle_high[df_middle_high['학교급별(1)'] == school_type]
    ax3.plot(data['연도'], data['1관당 방문자수'],
             color=color_map[school_type],
             linestyle='-',
             marker='o',
             linewidth=2,
             label=school_type)

ax3.set_title("중·고등학교 연도별 1관당 방문자수 추세 (확대)", fontproperties=font_prop)
ax3.set_xlabel("연도", fontproperties=font_prop)
ax3.set_ylabel("1관당 방문자수", fontproperties=font_prop)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(prop=font_prop, loc='upper right')
plt.tight_layout()
st.pyplot(fig3, use_container_width=False)

# ---------------------------
# 📝 결론 추가
# ---------------------------
st.subheader("📝 결론: 학교도서관 이용자수에 영향을 미치는 요인")
st.markdown("""
학교도서관 이용자수 분석 결과 및 기존 연구를 종합하면 다음과 같습니다.

### 1. 머신러닝(RandomForest) 분석 결과
- **1인당 대출자료수**가 가장 큰 영향을 미쳤으며,  
  그다음으로 **장서수(인쇄)**, **도서예산(자료구입비)**, **사서수** 순으로 중요도가 나타났습니다.

### 2. 기존 연구(인터넷 조사) 결과
- **하드웨어 요인**: 최신 자료 확보, IC 공간, 좌석 및 e-book 접근성
- **소프트웨어 요인**: 독서 프로그램, 학습 워크숍, 정보 리터러시 교육
- **인적 서비스 요인**: 상시 사서 교사 배치, 질 높은 안내 및 지원
- **심리적 요인**: 학생의 내적 동기 고취, 도서관 불안감 해소

### ✅ 시사점
학교도서관의 이용자수를 높이기 위해서는 **자료와 예산 확충**, **독서·학습 프로그램 운영**,  
**사서 교사의 적극적인 참여**, **학생 친화적 공간 조성** 등이 함께 이루어져야 합니다.
""")

# ---------------------------
# 📂 추가 기능: 원본 CSV 파일 보기 & 출처 정리
# ---------------------------
st.subheader("📂 분석에 사용된 원본 CSV 파일")
with st.expander("원본 CSV 파일 직접 보기"):
    st.markdown("**① 전국 학교도서관 통계 (문화체육관광부)**")
    st.dataframe(df1.head(20), use_container_width=True, height=250)
    st.markdown("**② 서울시 학교별 학교도서관 현황**")
    st.dataframe(df2.head(20), use_container_width=True, height=250)
    st.markdown("**③ 전국 학교도서관 연도별 현황**")
    st.dataframe(df3.head(20), use_container_width=True, height=250)

st.subheader("🔗 사용된 출처 정리")
st.markdown("""
**1. CSV 데이터 출처**
- 문화체육관광부: [국가도서관통계시스템](https://www.libsta.go.kr/)
- 서울특별시교육청: [서울교육통계](https://sts.sen.go.kr/)
- KOSIS 국가통계포털: 학교도서관 연도별 통계

**2. 인터넷 조사 출처**
- 한국교육학회 논문: "학교도서관 이용 활성화 요인 분석"
- 한국도서관정보학회지: "학생의 도서관 이용 행동과 서비스 만족도 연구"
- 해외 사례: 미국 AASL(American Association of School Librarians) 보고서
""")
