import cv2,time,argparse,glob
import numpy as np
original=cv2.imread("pilotos.png")
scale_percent = 60 # percent of original size
width = int(original.shape[1] * scale_percent / 100)
height = int(original.shape[0] * scale_percent / 100)
dim = (width, height)
original = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
  
            # Convert the BGR image to other color spaces
imageBGR = np.copy(original)
imageHSV = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
imageLAB = cv2.cvtColor(original,cv2.COLOR_BGR2LAB)

# resize image


# Show the results
cv2.imshow('SelectBGR',imageBGR)
cv2.imshow('SelectYCB',imageHSV)
cv2.imshow('SelectHSV',imageLAB)
cv2.waitKey(0)
cv2.destroyAllWindows()

