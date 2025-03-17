import os
import sys
import numpy as np


class LinearRegression:
    '''
    summary:
        선형 회귀는 입력 변수(X)와 출력 변수(y) 간의 선형 관계를 학습하는 지도 학습 모델.
        경사 하강법(Gradient Descent)으로 최적의 가중치(weights)와 절편(bias)을 학습한다.

    args:
        learning_rate (float): 학습률 (기본값: 0.001). 
        n_iters (int): 경사 하강법 반복 횟수 (기본값: 1000)
    '''
    def __init__(self, learning_rate=0.001, n_iters=1000):
        '''
        args:
            learning_rate (float): 학습률 (기본값: 0.001)
            n_iters (int): 경사 하강법 반복 횟수 (기본값: 1000)
        '''
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        summary:
            선형 회귀 모델 학습 (Gradient Descent 기반)

        args:
            X (numpy.ndarray): 훈련 데이터
            y (numpy.ndarray): 정답 레이블
        """
        n_samples, n_features = X.shape
        if n_samples != len(y):
            raise ValueError("입력 데이터 X와 y의 샘플 개수가 일치해야 합니다.")

        # 가중치와 절편 초기화
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 경사 하강법을 사용한 가중치 업데이트
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias 

            # 가중치 및 절편에 대한 그래디언트(기울기) 계산
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 가중치 및 절편 업데이트
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        summary:
            입력 데이터 X에 대한 예측 수행

        args:
            X (numpy.ndarray): 예측할 입력 데이터

        return:
            numpy.ndarray: 예측된 값 벡터
        """
        if self.weights is None or self.bias is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit()을 호출하세요.")

        return np.dot(X, self.weights) + self.bias