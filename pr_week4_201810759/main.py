import cv2

# 이미지 파일 적용
def pipeline(img):
#img = cv2.imread('./test_images/solidWhiteRight.jpg')

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(gray_img, (15, 15), 0.0)

    edge_img = cv2.Canny(blurred_img, 70, 140) # (이미지, lowTh, highTh), lowTh와 highTh를 적절히 수동적으로 조절

    return edge_img

# 비디오 파일 적용
cap = cv2.VideoCapture('./test_videos/solidWhiteRight.mp4')

while True:
    ok, frame = cap.read()
    if not ok:
        break

    edge_img = pipeline(frame)

    cv2.imshow('edge', edge_img)
    key = cv2.waitKey(30)  # -1
    if key == ord('x'):
        break

cap.release()
