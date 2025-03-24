import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_regression import LinearRegression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import mean_squared_error, r2_score, adjusted_r2_score

def main():
    parser = argparse.ArgumentParser(description="Linear Regression Model")
    parser.add_argument("--dataset", type=str, choices=["diabetes", "housing", "regression"], default="diabetes", help="Dataset choice: 'diabetes' for diabetes progression prediction, 'housing' for California housing prices, or 'regression' for synthetic regression dataset")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for gradient descent (default: 0.01)")
    parser.add_argument("--n_iters", type=int, default=500, help="Number of iterations for training (default: 500)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # 데이터 로드
    print("📥 데이터셋 로드 중...")
    if args.dataset == "diabetes":
        dataset = load_diabetes()
        X, y = dataset.data, dataset.target
        target_label = "Diabetes Progression"
    elif args.dataset == "housing":
        dataset = fetch_california_housing()
        X, y = dataset.data, dataset.target
        target_label = "House Price"
    else:
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=args.random_state)
        target_label = "Synthetic Regression Target"
    print(f"✅ 데이터 로드 완료! 선택한 데이터셋: {args.dataset}, 데이터 크기: X={X.shape}, y={y.shape}")
    
    # 데이터 정규화
    print("📊 데이터 정규화 중...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("✅ 데이터 정규화 완료!")
    
    # 데이터셋 분할
    print("📊 데이터셋 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"✅ 데이터 분할 완료! 훈련 데이터 크기: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"✅ 테스트 데이터 크기: X_test={X_test.shape}, y_test={y_test.shape}")
    
    # 모델 학습
    print(f"🔧 Linear Regression 모델 생성 중... (learning_rate={args.learning_rate}, n_iters={args.n_iters})")
    model = LinearRegression(learning_rate=args.learning_rate, n_iters=args.n_iters)
    print("📚 모델 훈련 중...")
    model.fit(X_train, y_train)
    
    # 예측 수행
    print("🤖 예측 중...")
    y_pred = model.predict(X_test)
    print("✅ 예측 완료!")
    
    # 성능 평가
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2_score(y_test, y_pred, X_test.shape[1])
    
    print(f"📊 Mean Squared Error (MSE): {mse:.4f}")
    print(f"📊 R2 Score: {r2:.4f}")
    print(f"📊 Adjusted R2 Score: {adj_r2:.4f}")

    return y_test, y_pred, target_label


if __name__ == "__main__":
    y_test, y_pred, target_label = main()

    # 실제 vs 예측 시각화
    plt.scatter(y_test, y_pred, alpha=0.5, label="Predictions")
    plt.xlabel(f"Actual {target_label}")
    plt.ylabel(f"Predicted {target_label}")
    plt.title(f"{target_label} Prediction")
    
    # y=x 직선 추가 (이상적인 예측 값)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Prediction (y=x)")
    
    plt.legend()
    plt.show()