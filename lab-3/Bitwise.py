#import opencv
import cv2 as cv
#read the images
img1 = cv.imread('circulo.png')
img2 = cv.imread('fundo.png')
bitwise_AND = cv.bitwise_and(img1, img2)
bitwise_OR = cv.bitwise_or(img1, img2)
bitwise_NOT = cv.bitwise_not(img1)
cv.imshow('img1',img1)
cv.imshow('img2',img2)
cv.imshow('AND',bitwise_AND)
cv.imshow('OR',bitwise_OR)
cv.imshow('NOT',bitwise_NOT)
if cv.waitKey(0) & 0xff == 27: 
    cv.destroyAllWindows()
