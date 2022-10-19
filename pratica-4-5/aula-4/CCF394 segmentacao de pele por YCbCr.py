import numpy as np
import numpy.linalg as la
import cv2

import math
import matplotlib.pyplot as plt
import time


# Count runtime

img=cv2.imread("pilotosPEQUENO.png")
img=cv2.medianBlur(img,5)
HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
rows,cols,cor = img.shape
print(rows, cols)
pixels =np.zeros((rows,cols),dtype=np.int8)
contador=0
for i in range(rows):
    for j in range(cols):
        (h,l,s) = HLS[i,j]
        
        if(s>0):
            ratiols=l/s
        else:
            ratiols=0
        if ( (s>51) and   ((ratiols>.5) and (ratiols<3))  and   ((h>15) and (h<165))):          
             img[i,j]=(0,242,255)
             contador=contador+1
              
    #print(i)
cv2.imshow("frame12",img)
cv2.waitKey(0)
              

cv2.destroyAllWindows()
print(contador)

    
    

