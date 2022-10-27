import numpy as np
import numpy.linalg as la
import cv2

import math
import matplotlib.pyplot as plt
import time
"""
 0<=H<=17 and 15<=S<=170 and 0<=V<=255

			and
			
    0<=Y<=255 and 135<=Cr<=180 and 85<=Cb<=135
"""


# Count runtime

img = cv2.imread('pilotosPequeno.png')
img=cv2.medianBlur(img,5)
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
CyCbCr = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
rows,cols,cor = img.shape
print(rows, cols)
pixels =np.zeros((rows,cols),dtype=np.int8)
contador=0
for i in range(rows):
    for j in range(cols):
        (h,l,s) = HSV[i,j]
        (Y,cr,cb)= CyCbCr[i,j]
        
        
        if((h>0) and (h<17)):
           # print("h %d" %(h))
            if((s>15) and (s<170)):
           #     print("s %d" %(s))
                
                if((cr>135) and (h<180)):
           #         print("cr %d" %(cr))
                    if((cb>85) and (cb<135)):
                         img[i,j]=(0,242,255)
             
              
    #print(i)
cv2.imshow("frame12",img)
cv2.waitKey(0)
              

cv2.destroyAllWindows()
print(contador)

    
    

