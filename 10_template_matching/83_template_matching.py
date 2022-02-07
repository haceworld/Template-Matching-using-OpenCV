# template matching - find an image in a larger image
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg')
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

temp = cv2.imread('messi_face.jpg', 0)


res = cv2.matchTemplate(grey_img, temp, cv2.TM_CCOEFF_NORMED)
print(res)


threshold = 1.0
loc = np.where(res >= threshold)
print(loc)

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
