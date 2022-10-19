import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

cap = cv2.VideoCapture(0)


# Making window size adjustable
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

while (1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prewitt_cross_v = np.array([[-1, -1, -1],
                                [0, 0, 0],
                                [1, 1, 1]])

    prewitt_cross_h = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])

    frame= np.asarray(frame, dtype="int32")
    frame=frame/255.

    vertical = ndimage.convolve(frame, prewitt_cross_v)
    horizontal = ndimage.convolve(frame, prewitt_cross_h)

    output_image = np.sqrt(np.square(horizontal) + np.square(vertical))
    output_image = np.asarray(np.clip(output_image, 0, 255))
    cv2.imshow("saida",output_image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
