import os
import sys
import numpy as np


class LogisticRegression:
    """
    summary:
        로지스틱 회귀는 입력 변수(X)와 출력 변수(y) 간의 관계를 확률적으로 모델링하는 지도 학습 분류 모델.
        시그모이드 함수를 사용하여 출력값을 확률(0~1)로 변환하고, 이를 바탕으로 이진 분류를 수행한다.
        경사 하강법(Gradient Descent)을 통해 가중치(weights)와 절편(bias)을 최적화한다.

    args:
        learning_rate (float): 학습률 (기본값: 0.001).
        n_iters (int): 경사 하강법 반복 횟수 (기본값: 1000)
    """
    def __init__(self, learning_rate=0.001, n_iters=1000):
        """
        args:
            learning_rate (float): 학습률 (기본값: 0.001)
            n_iters (int): 경사 하강법 반복 횟수 (기본값: 1000)
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        summary:
            로지스틱 회귀 모델 학습 (Gradient Descent 기반)

        args:
            X (numpy.ndarray): 훈련 데이터
            y (numpy.ndarray): 정답 레이블 (0 또는 1)
        """
        n_samples, n_features = X.shape
        if n_samples != len(y):
            raise ValueError("입력 데이터 X와 y의 샘플 개수가 일치해야 합니다.")

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        summary:
            입력 데이터 X에 대한 클래스 예측 수행

        args:
            X (numpy.ndarray): 예측할 입력 데이터

        return:
            numpy.ndarray: 예측된 클래스 (0 또는 1)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit()을 호출하세요.")

        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])

    def _sigmoid(self, x):
        """
        summary:
            시그모이드 함수 적용

        args:
            x (numpy.ndarray or float): 입력 값

        return:
            numpy.ndarray or float: 시그모이드 함수 출력
        """
        return 1 / (1 + np.exp(-x))