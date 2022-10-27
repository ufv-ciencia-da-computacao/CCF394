import cv2
import numpy as np
# interessante

original = cv2.imread('impressaoDigital.JPG',0)
size = np.size(original)
skel = np.zeros(original.shape,np.uint8)
ret,img = cv2.threshold(original,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True

cv2.imshow("Original",original)

cv2.imshow("Esqueleto",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
