import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import mean_squared_error, r2_score, adjusted_r2_score


# 데이터 로드
dataset = fetch_california_housing()
X, y = dataset.data, dataset.target  # X: 집의 특징들, y: 집값

# 데이터 정규화 (선형 회귀는 입력 데이터 스케일링이 중요함)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 성능 평가 (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Boston Housing 데이터셋 MSE: {mse:.4f}")

# 실제 vs 예측 시각화
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Boston Housing Prediction")
plt.show()
