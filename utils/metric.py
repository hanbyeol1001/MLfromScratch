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


def r2_score(y_true, y_pred):
    """
    결정 계수 (R², R-squared) 계산

    summary:
        예측 모델이 데이터를 얼마나 잘 설명하는지를 나타내는 지표.
        1에 가까울수록 모델이 데이터를 잘 설명하며, 0에 가까울수록 설명력이 낮다.

    args:
        y_true (numpy.ndarray): 실제 값 (타겟 변수)
        y_pred (numpy.ndarray): 예측된 값

    return:
        float: 결정 계수 (R²) 값 (0~1 사이)
    """
    corr_matrix = np.corrcoef(y_true, y_pred)  # 상관 계수 행렬 계산
    corr = corr_matrix[0, 1]  # 실제 값과 예측 값의 상관 계수 추출
    return corr ** 2  # R² 값 반환


def adjusted_r2_score(y_true, y_pred, n_features):
    """
    summary:
        R² 값에서 독립 변수 개수를 고려하여 조정된 값으로, 
        불필요한 변수를 추가하면 감소할 수 있음.

    args:
        y_true (numpy.ndarray): 실제 값 (타겟 변수)
        y_pred (numpy.ndarray): 예측된 값
        n_features (int): 독립 변수 (특징) 개수

    return:
        float: Adjusted R² 값 (0~1 사이)
    """
    n = len(y_true)  # 샘플 개수
    r2 = r2_score(y_true, y_pred)  # 기존 R² 계산

    # Adjusted R² 계산
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    
    return adj_r2
