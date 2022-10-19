#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
cv2.destroyAllWindows()
# saturação >= 0.2) and (0.5 < luminance/Saturation < 3.0) and (h <= 28 or hue >= 330)
image = cv2.imread('pilotosPEQUENO.png')

blur = cv2.medianBlur(image ,3)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV).astype(float)

h,s,v = cv2.split(hsv)

lower = np.array([40,50,20])
upper = np.array([160,120,80])

mask = cv2.inRange(blur, lower, upper)
res = cv2.bitwise_and(image,image, mask= mask)            

cv2.imshow("mask ",mask)
cv2.imshow('stack', np.hstack([image, res]))
cv2.waitKey(0)
cv2.destroyAllWindows()


lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")


mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(image,image, mask= mask)            

cv2.imshow("mask ",mask)
cv2.imshow('stack', np.hstack([image, res]))
cv2.waitKey(0)
cv2.destroyAllWindows()
