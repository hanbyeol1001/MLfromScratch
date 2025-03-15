import numpy as np


def accuracy(y_true, y_pred):
    """
    정확도를 계산하는 함수

    args:
        y_true (numpy.ndarray): 실제 레이블
        y_pred (numpy.ndarray): 예측된 레이블

    return:
        float: 정확도 값 (0~1 사이)
    """
    return np.sum(y_true == y_pred) / len(y_true)


