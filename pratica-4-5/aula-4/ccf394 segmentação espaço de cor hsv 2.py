import numpy as np
import cv2
frame = cv2.imread("blococores.png")
frame = cv2.imread("bARRAS.png")

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)
v,h,s = cv2.split(hsv)
cv2.imshow("v", v)
cv2.imshow("h", h)
cv2.imshow("s", s)
cv2.imshow("original", frame)



cv2.waitKey(0)
cv2.destroyAllWindows()

