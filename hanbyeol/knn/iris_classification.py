import sys
import os
import argparse
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.metrics import accuracy


def main():
    '''
    터미널에서 입력을 받아 k-NN 분류를 수행하는 메인 함수
    '''
    # Argument Parser 설정
    parser = argparse.ArgumentParser(description="k-NN 분류 모델 실행")
    parser.add_argument("--k", type=int, default=3, help="최근접 이웃 개수 (기본값: 3)")
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["euclidean", "manhattan", "cosine"],
                        help="거리 측정 방법 ('euclidean', 'manhattan', 'cosine')")
    parser.add_argument("--test_size", type=float, default=0.2, 
                        help="테스트 데이터 비율 (0~1 사이, 기본값: 0.2)")
    parser.add_argument("--random_state", type=int, default=1234,
                        help="랜덤 시드 (기본값: 1234)")

    # 입력값 파싱
    args = parser.parse_args()

    # Iris 데이터셋 로드
    print("📥 데이터셋 로드 중...")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print(f"✅ 데이터 로드 완료! 데이터 크기: X={X.shape}, y={y.shape}\n")

    # 데이터셋 분할
    print("📊 데이터셋 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(f"✅ 데이터 분할 완료! 훈련 데이터 크기: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"✅ 테스트 데이터 크기: X_test={X_test.shape}, y_test={y_test.shape}\n")

    # k-NN 모델 생성 및 학습
    print(f"🔧 k-NN 모델 생성 중... (k={args.k}, metric={args.metric})")
    clf = KNN(k=args.k, metric=args.metric)
    print("📚 모델 훈련 중...")
    clf.fit(X_train, y_train)
    print("🤖 예측 중...")
    predictions = clf.predict(X_test)

    # 정확도 출력
    print("✅ 예측 완료!\n")
    acc = accuracy(y_test, predictions)
    print(f"KNN classification accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
