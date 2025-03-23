import numpy as np


def euclidean_distance(x1, x2):
    # 유클리드 거리(Euclidean Distance) 계산
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    # 맨해튼 거리(Manhattan Distance) 계산
    return np.sum(np.abs(x1 - x2))


def cosine_similarity(x1, x2):
    # 코사인 유사도(Cosine Similarity) 계산
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    if norm_x1 == 0 or norm_x2 == 0:
        return 0  # 벡터가 0일 경우 유사도 0으로 설정
    return dot_product / (norm_x1 * norm_x2)