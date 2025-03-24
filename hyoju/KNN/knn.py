import numpy as np
from collections import Counter


# step2. 유클리디안 거리 정의하기
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


# step 1. KNN부터 정의하기

class KNN :
    #step 1-1
    def __init__(self, k=10 ) : #k는 고려하고 싶은 이웃의 갯수, 기본값=3
        self.k = k
    
    #step 1-2
    def fit(self, X, y):
        # KNN은 학습단계가 포함되어 있지 않음. 따라서 훈련 샘플을 저장하고 나중에 사용하기
        
        self.X_train = X
        self.y_train = y
    #step 1-3
    def predict(self, X):
        # 여러개의 샘플을 사용할 수 있음.그래서 인자도 large X
        # predicted_labels를 정의하여 리스트에 넣음
        
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
        
    #step 1-4   
    def _predict(self ,x):
        # 하나의 예측 샘플에 대해 계산
        
        # 유클리안거리를 사용하여 거리를 계산하기
        distaces = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # k 최근접 이웃과 라벨을 얻기.
        k_indices = np.argsort(distaces)[:self.k] #슬라이싱을 이용하여 k 개의 라벨을 찾기
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common class label 
        most_common = Counter(k_nearest_labels).most_common(1) #1을 넣어서 가장 흔한 값을 찾기
        
        return most_common[0][0]
    
