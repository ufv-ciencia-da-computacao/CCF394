import cv2
import numpy as np 

video = cv2.VideoCapture("resources/chromakey.mp4")
image = cv2.imread("resources/new-york.png")

while True:
    ret, frame = video.read()

    frame = cv2.resize(frame, (720, 480))
    image = cv2.resize(image, (720, 480))
    u_green = np.array([110, 255, 110])
    l_green = np.array([0, 100, 0])
 
    mask = cv2.inRange(frame, l_green, u_green)
    res = cv2.bitwise_and(frame, frame, mask = mask)
 
    f = frame - res
    f = np.where(f == 0, image, f)
 
    cv2.imshow("mask", f)
 
    if cv2.waitKey(25) == 27:
        break
 
video.release()
cv2.destroyAllWindows()