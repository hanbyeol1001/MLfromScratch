import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegression 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import mean_squared_error, r2_score, adjusted_r2_score


# 데이터 로드 (MNIST)
mnist = fetch_openml("mnist_784", version=1, cache=True)
X, y = mnist.data, mnist.target.astype(np.float32)  # y를 float으로 변환

# 특정 픽셀(중앙 픽셀)의 값을 예측하는 문제로 변환
target_pixel_idx = 392  # 392번 픽셀 (28x28 이미지의 중간쯤)
y = X[:, target_pixel_idx]  # 타겟: 중앙 픽셀 값
X = np.delete(X, target_pixel_idx, axis=1)  # 입력 데이터에서 해당 픽셀 제거

# 데이터 정규화 (0~255 → 0~1 범위로 변환)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y / 255.0  

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression(learning_rate=0.01, n_iters=500)
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 성능 평가 (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"MNIST 데이터셋 MSE: {mse:.4f}")

# 실제 vs 예측 시각화
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Pixel Value")
plt.ylabel("Predicted Pixel Value")
plt.title("MNIST Pixel Value Prediction")
plt.show()