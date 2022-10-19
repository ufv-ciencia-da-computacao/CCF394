import cv2 
import numpy as np
from scipy import ndimage
  
roberts_cross_v = np.array( [[0,0,0 ],
                             [0,1,0],
                             [0,0,-1]] )
  
roberts_cross_h = np.array( [[0,0,0 ],
                             [0,0,1],
                             [0,-1,0]] )
img = cv2.imread("input-300x200.webp",0)
cv2.imshow("ENTRADA",img)
cv2.waitKey(0)
  

img= np.asarray(img, dtype="int32")
img=img/255.

vertical = ndimage.convolve( img, roberts_cross_v )
horizontal = ndimage.convolve( img, roberts_cross_h )
  
output_image = np.sqrt( np.square(horizontal) + np.square(vertical))
output_image = np.asarray(np.clip(output_image,0,255))
cv2.imwrite("output.jpg",output_image)
cv2.imshow("saida",output_image)
cv2.waitKey(0)
