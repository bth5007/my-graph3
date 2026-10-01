import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 날짜와 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# --------------------------------------------------
# 연도별 데이터 정리
# --------------------------------------------------
# 수업 기준 기간: 2025년까지
df = df[df["연도"] <= 2025].copy()

# 평균기온이 실제로 관측된 일수
yearly_count = (
    df.groupby("연도")["평균기온"]
    .count()
    .reset_index(name="관측일수")
)

# 연도별 평균기온
yearly_temp = (
    df.groupby("연도")["평균기온"]
    .mean()
    .reset_index(name="연평균기온")
)

# 두 데이터를 합치기
yearly = pd.merge(
    yearly_temp,
    yearly_count,
    on="연도"
)

# 관측일이 300일 이상인 연도만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

# 연도순 정렬
yearly = yearly.sort_values("연도").reset_index(drop=True)

# --------------------------------------------------
# 회귀분석
# --------------------------------------------------
# 독립변수:
# 1908년부터 지난 연수
# 예) 1908년 -> 0
#     1909년 -> 1
#     2000년 -> 92
yearly["지난연수"] = yearly["연도"] - 1908

x = yearly["지난연수"].to_numpy()
y = yearly["연평균기온"].to_numpy()

# 선형회귀
slope, intercept = np.polyfit(x, y, 1)

# 회귀선 예측값
yearly["회귀예측기온"] = slope * x + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 회귀식의 결정계수
r_squared = correlation ** 2

# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연평균기온 변화를 바탕으로 "
    "선형 회귀를 이용해 기온을 예측합니다."
)

st.info(
    "2025년까지의 데이터만 사용하며, "
    "평균기온 관측일수가 300일 미만인 연도는 제외했습니다."
)

# --------------------------------------------------
# 분석에 사용한 기간 정보
# --------------------------------------------------
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{data_count}년")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

# --------------------------------------------------
# 상관계수 표시
# --------------------------------------------------
st.subheader("📊 상관계수")

st.metric(
    "연도 경과와 연평균기온의 상관계수",
    f"{correlation:.3f}"
)

st.write(
    f"결정계수(R²): **{r_squared:.3f}**"
)

# --------------------------------------------------
# 산점도 + 회귀선
# --------------------------------------------------
st.subheader("📈 연도별 연평균기온과 회귀선")

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        text=yearly["관측일수"].astype(str) + "일 관측",
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{text}<extra></extra>"
        ),
        marker=dict(size=7)
    )
)

# 회귀선
# 실제 회귀에 사용된 연도 범위에서 표시
line_x = np.arange(start_year, end_year + 1)
line_elapsed = line_x - 1908
line_y = slope * line_elapsed + intercept

fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예상기온: %{y:.2f}℃<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified",
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# 회귀식
# --------------------------------------------------
st.subheader("📐 회귀식")

st.latex(
    f"y = {slope:.4f}x + {intercept:.4f}"
)

st.write(
    "여기서 x는 **1908년부터 지난 연수(연도 - 1908)**이고, "
    "y는 연평균기온(℃)입니다."
)

# --------------------------------------------------
# 연도 슬라이더
# --------------------------------------------------
st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 선택한 연도의 예상기온
selected_elapsed = selected_year - 1908
predicted_temp = slope * selected_elapsed + intercept

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        background-color: #fff8dc;
        margin-top: 20px;
        margin-bottom: 20px;
    ">
        <div style="font-size: 24px; font-weight: bold;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size: 52px;
            font-weight: bold;
            margin-top: 10px;
        ">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# 선택한 연도를 그래프에서도 표시
# --------------------------------------------------
forecast_years = np.arange(1900, 2101)
forecast_elapsed = forecast_years - 1908
forecast_temps = slope * forecast_elapsed + intercept

forecast_fig = go.Figure()

# 회귀선 전체 구간
forecast_fig.add_trace(
    go.Scatter(
        x=forecast_years,
        y=forecast_temps,
        mode="lines",
        name="회귀선",
        hovertemplate=(
            "연도: %{x}년<br>"
            "예상기온: %{y:.2f}℃<extra></extra>"
        )
    )
)

# 실제 관측값
forecast_fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        hovertemplate=(
            "연도: %{x}년<br>"
            "실제 연평균기온: %{y:.2f}℃<extra></extra>"
        ),
        marker=dict(size=6)
    )
)

# 선택한 연도
forecast_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=16, symbol="star"),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상기온: {predicted_temp:.2f}℃"
            "<extra></extra>"
        )
    )
)

forecast_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="기온 (℃)",
    xaxis=dict(
        range=[1900, 2100],
        tickmode="linear",
        dtick=10
    ),
    hovermode="closest",
    height=600
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)

# --------------------------------------------------
# 사용 데이터 확인
# --------------------------------------------------
with st.expander("📋 회귀에 사용된 연도별 데이터 보기"):
    display_df = yearly[
        ["연도", "연평균기온", "관측일수", "지난연수"]
    ].copy()

    display_df["연평균기온"] = display_df["연평균기온"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
