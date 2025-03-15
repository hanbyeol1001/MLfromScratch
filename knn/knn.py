from collections import Counter
import numpy as np
from distance import euclidean_distance, manhattan_distance, cosine_similarity


class KNN:
    """
    k-NN (k-Nearest Neighbors) 알고리즘을 구현한 클래스

    summary:
        입력된 데이터셋을 저장하고, 거리 기반으로 최근접 이웃을 찾아 분류 또는 회귀 예측을 수행하는 모델.
        거리 측정 방법으로 유클리드 거리, 맨해튼 거리, 코사인 유사도를 선택할 수 있다.

    args:
        k (int): 최근접 이웃 개수 (기본값: 3)
        metric (str): 거리 측정 방법 ('euclidean', 'manhattan', 'cosine' 중 선택, 기본값: 'euclidean')
    """
    def __init__(self, k=3, metric="euclidean"):
        """
        KNN 모델을 초기화.

        args:
            k (int): 최근접 이웃 개수 (기본값: 3)
            metric (str): 거리 측정 방법 ('euclidean', 'manhattan', 'cosine' 중 선택, 기본값: 'euclidean')

        raises:
            ValueError: 지원하지 않는 거리 측정 방법이 입력되었을 경우 발생
        """
        self.k = k
        self.metric = metric
        self.distance_func = self._get_distance_function(metric)

    def _get_distance_function(self, metric):
        """
        거리 측정 방법을 설정.

        args:
            metric (str): 거리 측정 방법 ('euclidean', 'manhattan', 'cosine')

        return:
            function: 선택된 거리 측정 함수

        raises:
            ValueError: 지원하지 않는 거리 측정 방법이 입력된 경우 발생
        """
        if metric == "euclidean":
            return euclidean_distance
        elif metric == "manhattan":
            return manhattan_distance
        elif metric == "cosine":
            # 코사인 유사도는 거리가 아니라 유사도를 측정하므로, 1 - 유사도를 반환해야 함
            return lambda x1, x2: 1 - cosine_similarity(x1, x2)
        else:
            raise ValueError("지원하지 않는 거리 측정 방법입니다. (euclidean, manhattan, cosine 중 선택)")

    def fit(self, X, y):
        """
        훈련 데이터를 저장.

        args:
            X (numpy.ndarray): 입력 데이터 (특징 벡터)
            y (numpy.ndarray): 해당 데이터의 레이블

        return:
            None
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        입력 데이터 X에 대한 예측을 수행.

        args:
            X (numpy.ndarray): 예측할 데이터 (샘플 여러 개)

        return:
            numpy.ndarray: 예측된 클래스 배열
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        """
        단일 샘플 x에 대한 예측을 수행.

        args:
            x (numpy.ndarray): 예측할 단일 샘플

        return:
            int or float: 예측된 클래스 또는 회귀값
        """
        # 모든 훈련 데이터와의 거리 계산
        distances = [self.distance_func(x, x_train) for x_train in self.X_train]

        # 거리 기준으로 정렬하여 k개의 최근접 이웃 선택
        k_idx = np.argsort(distances)[: self.k]

        # 최근접 이웃들의 레이블 가져오기
        k_neighbor_labels = [self.y_train[i] for i in k_idx]

        # 가장 많이 등장한 클래스를 예측값으로 반환
        most_common = Counter(k_neighbor_labels).most_common(1)

        return most_common[0][0]