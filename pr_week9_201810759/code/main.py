import cv2
import numpy as np
import util

filepath = 'rectangle.png'
img = cv2.imread(filepath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (262, 336)
w = 262
h = 336
gray1 = cv2.resize(gray, dsize=(w, h), interpolation=cv2.INTER_AREA)
siftstart = cv2.getTickCount()
sift = cv2.xfeatures2d.SIFT_create()
kpts1 = sift.detect(image=gray1, mask=None)
siftend = cv2.getTickCount()
sifttimerequired = (siftend - siftstart) / cv2.getTickFrequency()
surfstart = cv2.getTickCount()
surf = cv2.xfeatures2d.SURF_create()
kpts2 = surf.detect(image=gray1, mask=None)
surfend = cv2.getTickCount()
surftimerequired = (surfend - surfstart) / cv2.getTickFrequency()

print(">> ", (w, h))
print("SIFT: ", sifttimerequired)
print("SURF: ", surftimerequired)
print("\n")

# (523, 672)
w = 523
h = 672
gray1 = cv2.resize(gray, dsize=(w, h), interpolation=cv2.INTER_AREA)
siftstart = cv2.getTickCount()
sift = cv2.xfeatures2d.SIFT_create()
kpts1 = sift.detect(image=gray1, mask=None)
siftend = cv2.getTickCount()
sifttimerequired = (siftend - siftstart) / cv2.getTickFrequency()
surfstart = cv2.getTickCount()
surf = cv2.xfeatures2d.SURF_create()
kpts2 = surf.detect(image=gray1, mask=None)
surfend = cv2.getTickCount()
surftimerequired = (surfend - surfstart) / cv2.getTickFrequency()

print(">> ", (w, h))
print("SIFT: ", sifttimerequired)
print("SURF: ", surftimerequired)
print("\n")

# (1046, 1344)
w = 1046
h = 1344
gray1 = cv2.resize(gray, dsize=(w, h), interpolation=cv2.INTER_AREA)
siftstart = cv2.getTickCount()
sift = cv2.xfeatures2d.SIFT_create()
kpts1 = sift.detect(image=gray1, mask=None)
siftend = cv2.getTickCount()
sifttimerequired = (siftend - siftstart) / cv2.getTickFrequency()
surfstart = cv2.getTickCount()
surf = cv2.xfeatures2d.SURF_create()
kpts2 = surf.detect(image=gray1, mask=None)
surfend = cv2.getTickCount()
surftimerequired = (surfend - surfstart) / cv2.getTickFrequency()

print(">> ", (w, h))
print("SIFT: ", sifttimerequired)
print("SURF: ", surftimerequired)
print("\n")

# (5230, 6720)
w = 5230
h = 6720
gray1 = cv2.resize(gray, dsize=(w, h), interpolation=cv2.INTER_AREA)
siftstart = cv2.getTickCount()
sift = cv2.xfeatures2d.SIFT_create()
kpts1 = sift.detect(image=gray1, mask=None)
siftend = cv2.getTickCount()
sifttimerequired = (siftend - siftstart) / cv2.getTickFrequency()
surfstart = cv2.getTickCount()
surf = cv2.xfeatures2d.SURF_create()
kpts2 = surf.detect(image=gray1, mask=None)
surfend = cv2.getTickCount()
surftimerequired = (surfend - surfstart) / cv2.getTickFrequency()

print(">> ", (w, h))
print("SIFT: ", sifttimerequired)
print("SURF: ", surftimerequired)
print("\n")
