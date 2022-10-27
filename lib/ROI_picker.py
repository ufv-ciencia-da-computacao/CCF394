import numpy as np
import numpy.linalg as la
import cv2

import math
import matplotlib.pyplot as plt
import time

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global pixels
        (b,g,r)=img[y,x]
        print ("col: %d, row: %d   - R= %d, G=%d, B=%d" % (x, y, r,g,b))
        x=np.array([[b,g,r]])
        print(x)
        pixels=np.concatenate((pixels,x))
        
        
        
pixels =np.zeros((1,3),dtype=np.int8)
print(pixels)
img=cv2.imread("jogo do cruzeiro.png")

img=cv2.medianBlur(img,3)
cv2.namedWindow('Original')

cv2.setMouseCallback("Original", on_mouse)

while True:
    # display the image and wait for a keypress
    cv2.imshow("Original", img)
    key = cv2.waitKey(1) & 0xFF
    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break

cv2.destroyAllWindows() 

media=np.mean(pixels,axis=0)
mediaB=int(media[0])
mediaG=int(media[1])
mediaR=int(media[2])

# print("Media B=%d, media G= %d, media R=%s \n" % (mediaB, mediaG, mediaR))

rows,cols,cor = img.shape
# print(rows, cols)

desvioPadrao= np.std(pixels, axis=0)/2
desvioPadrao=np.uint8(desvioPadrao)

if((mediaB-desvioPadrao[0])<0):
    lowerB=0
else:
    lowerB=mediaB-desvioPadrao[1]
if((mediaG-desvioPadrao[0])<0):
    lowerG=0
else:
    lowerG=mediaG-desvioPadrao[1]
if((mediaR-desvioPadrao[2])<0):
    lowerR=0
else:
    lowerR=mediaR-desvioPadrao[2]


lower = np.array([lowerB,lowerG,lowerR])
upper = np.array([mediaB+desvioPadrao[0],mediaG+desvioPadrao[1],mediaR+desvioPadrao[2]])

print(lower)
print(upper)

