import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# 1. 1906~2025년 연평균 기온 데이터 생성 (현실적 기후변화 추세 반영)
np.random.seed(42)
years = np.arange(1906, 2026)

# 장기 온난화 추세 + 최근 50년 가속화 + 변동성(노이즈)
trend = 10.5 + 0.012 * (years - 1906) + 0.00012 * np.maximum(0, years - 1970) ** 2
noise = np.random.normal(0, 0.45, len(years))
temp = trend + noise

df = pd.DataFrame({"Year": years, "Temp": temp})

# 2. 데이터셋 분할
# 공통 테스트 데이터 (2006~2025)
test_mask = (df["Year"] >= 2006) & (df["Year"] <= 2025)
X_test = df.loc[test_mask, ["Year"]]
y_test = df.loc[test_mask, "Temp"]

# 학습 데이터 A: 최근 50년 (1956~2005)
train50_mask = (df["Year"] >= 1956) & (df["Year"] <= 2005)
X_train50 = df.loc[train50_mask, ["Year"]]
y_train50 = df.loc[train50_mask, "Temp"]

# 학습 데이터 B: 최근 100년 (1906~2005)
train100_mask = (df["Year"] >= 1906) & (df["Year"] <= 2005)
X_train100 = df.loc[train100_mask, ["Year"]]
y_train100 = df.loc[train100_mask, "Temp"]

# 전체 데이터 (1906~2025)
X_all = df[["Year"]]
y_all = df["Temp"]

# 3. 모델 학습 및 예측
# 전체 데이터 모델
model_all = LinearRegression().fit(X_all, y_all)

# 모델 A (최근 50년 학습)
model_50 = LinearRegression().fit(X_train50, y_train50)
pred_50_test = model_50.predict(X_test)

# 모델 B (최근 100년 학습)
model_100 = LinearRegression().fit(X_train100, y_train100)
pred_100_test = model_100.predict(X_test)


# 4. 성능 평가 함수
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2


mae_50, mse_50, r2_50 = evaluate(y_test, pred_50_test)
mae_100, mse_100, r2_100 = evaluate(y_test, pred_100_test)

# 결과 출력
print("=== [회귀선 방정식] ===")
print(
    f"전체 모델 (1906~2025): Temp = {model_all.coef_[0]:.4f} * Year + ({model_all.intercept_:.2f})"
)
print(
    f"모델 A (1956~2005, 50년): Temp = {model_50.coef_[0]:.4f} * Year + ({model_50.intercept_:.2f})"
)
print(
    f"모델 B (1906~2005, 100년): Temp = {model_100.coef_[0]:.4f} * Year + ({model_100.intercept_:.2f})"
)

print("\n=== [테스트 데이터(2006~2025) 예측 평가] ===")
print(
    f"모델 A (최근 50년 학습)  -> MAE: {mae_50:.4f} | MSE: {mse_50:.4f} | R²: {r2_50:.4f}"
)
print(
    f"모델 B (최근 100년 학습) -> MAE: {mae_100:.4f} | MSE: {mse_100:.4f} | R²: {r2_100:.4f}"
)
