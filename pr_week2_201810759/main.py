import cv2
import numpy as np
import decimal

# 모델 영상의 2차원 H-S 히스토그램 계산
img_m = cv2.imread('model.jpg')
hsv_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2HSV)
hist_m = cv2.calcHist([hsv_m], [0, 1], None, [180, 256], [0, 180, 0, 256])  # 파라미터(이미지, 채널(색상, 채도), 마스크(none), 색상과 채도 bin개수, 색상과 채도 각각의 범위)

# 입력 영상의 2차원 H-S 히스토그램 계산
img_i = cv2.imread('hand.jpg')
hsv_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2HSV)
hist_i = cv2.calcHist([hsv_i], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 히스토그램 정규화
hist_m = hist_m / (img_m.shape[0] * img_m.shape[1])     # 히스토그램을 영상사이즈 만큼 나눠 정규화
hist_i = hist_i / img_i.size
print("maximum of hist_m: % f" % hist_m.max()) # 값 범위 체크 1.0 이하
print("maximum of hist_i: % f" % hist_i.max()) # 값 범위 체크 1.0 이하

# 비율 히스토그램 계산
hist_r = hist_m / (hist_i + 1e-7)
hist_r = np.minimum(hist_r, 1.0)
print("range of hist_r: [%.1f, %.1f]" % (hist_r.min(), hist_r.max()))   # 비율 값 범위 체크: [0.0, 1.0]

# 히스토그램 역투영 수행
height, width = img_i.shape[0], img_i.shape[1]
result = np.zeros_like(img_i, dtype='float32')
h, s, v = cv2.split(hsv_i)

for i in range(height):
    for j in range(width):
        h_value = h[i, j]   # (i, j)번 째 픽셀의 색상 값
        s_value = s[i, j]   # (i, j)번 째 픽셀의 채도 값
        confidence = hist_r[h_value, s_value]   # (i, j)번 째 픽셀의 신뢰도 점수
        result[i, j] = confidence   # 신뢰도 점수를 결과 이미지의 (i, j)번 째 픽셀에 저장

# 이진화 수행 (화소값이 임계값 0.02보다 크면 255, 그렇지 않으면 0)
ret, thresholded = cv2.threshold(result, 0.02, 255, cv2.THRESH_BINARY)     # 파라미터(이미지, 임계치, 임계치보다 크면 적용되는 값,cv2.THRESH_BINARY => 임계치보다 작으면 0으로 할당)
cv2.imwrite('result.jpg', thresholded)

# 모폴로지 연산 적용
kernel = np.ones((15, 15), np.uint8)
improved = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('morphology.jpg', improved)