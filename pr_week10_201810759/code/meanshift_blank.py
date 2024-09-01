import numpy as np
import sample_data
import matplotlib.pyplot as plt

# 점과 점 사이 거리 구하는 함수
def calc_euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calc_weight(x, kernel='flat'):
    # 문제부분
    if np.all(x <= 1):
        if kernel.lower() == 'flat':
            return 1
        elif kernel.lower() == 'gaussian':
            return np.exp(-1 * (x ** 2))
        else:
            raise Exception("'%s' is invalid kernel" % kernel)
    else:
        return 0

#epsilon => 이전 점의 위치와 새로 도출된 점의 위치의 거리 차이
    
def mean_shift(X, bandwidth, n_iteration=20, epsilon=0.001):
    centroids = np.zeros_like(X)   

    for i in range(len(X)):        
        centroid = X[i].copy()  # 초기 중심점(t_0) 설정 -> 각 datapoint를 초기 중심점으로 할당
        prev = centroid.copy()
        
        t = 0   # => 무한루프 방지를 위해 개수 증가, 최대개수는 n_iteration=20
        while True:
            # 코드 작성 시작
            '''
            a = 0
            b = 0
            for j in range(len(X)):
                a += X[j] * calc_weight((X[j] - centroid) / bandwidth, kernel='flat')
                b += calc_weight((X[j] - centroid) / bandwidth, kernel='flat')

            centroid = a / b

            if (calc_euclidean_distance(centroid, prev) <= epsilon) or (t > n_iteration):
                break
            '''
            # 코드 작성 완료

            # 조교님 코드 시작
            # 종료 조건 1. 반복 횟수가 n_iteration을 초과하면 stop
            if t > n_iteration:
                break

            # 현재 중심점으로부터 bandwidth 내에 있는 샘플들을 기반으로 새로운 군집 중심점 계산
            numerator = 0   # a
            denominator = 0     # b
            for sample in X:    # for j in range(len(X)):
                distance = calc_euclidean_distance(centroid, sample)
                weight = calc_weight(distance / bandwidth, 'flat')
                numerator += ((sample - centroid) * weight)     # 조교님은 5.19 사용
                denominator += weight

            if denominator == 0:
                shift = 0
            else:
                shift = numerator / denominator

            centroid += shift

            # 종료 조건 2. 수렴했으면 stop
            if calc_euclidean_distance(centroid, prev) < epsilon:
                break

            # 조교님 코드 완료

            prev = centroid.copy()
            t += 1
        
        centroids[i] = centroid.copy()

    return centroids

    
def mean_shift_with_history(X, bandwidth, n_iteration=20, epsilon=0.001):
    history = {}
    for i in range(len(X)):
        history[i] = []
    centroids = np.zeros_like(X)   

    for i in range(len(X)):
        centroid = X[i].copy()  # 초기 중심점(t_0) 설정 -> 각 datapoint를 초기 중심점으로 할당
        prev = centroid.copy()
        history[i].append(centroid.copy())
        
        t = 0
        while True:
            # 코드 작성 시작
            a = 0.0
            b = 0.0
            for j in range(len(X)):
                a += X[j] * calc_weight((X[j] - prev) / bandwidth, kernel='flat')
                b += calc_weight((X[j] - prev) / bandwidth, kernel='flat')

            centroid = a / b

            if (calc_euclidean_distance(centroid, prev) <= epsilon) or (t > n_iteration):
                break

            # 코드 작성 완료

            prev = centroid.copy()
            t += 1

            history[i].append(centroid.copy())
        
        centroids[i] = centroid.copy()

    return centroids, history

SD = sample_data.sample1

centeroids1 = mean_shift(SD, bandwidth=4, n_iteration=20, epsilon=0.001)

