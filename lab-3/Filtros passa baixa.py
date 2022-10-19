# filtros opencv
#https://docs.opencv.org/4.6.0/d4/d13/tutorial_py_filtering.html
import cv2
import numpy as np
import os

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
        print("Imagem 2 canais")
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

"""
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n,is uniform noise with specified mean & variance.
"""


def noisy(noise_typ,image):
  if len(image.shape) == 3:
    row,col,ch= image.shape
  else:
    row,col= image.shape
    ch=None

  if noise_typ == "gauss":
   
    mean = 0
    #var = 0.1
    #sigma = var**0.5
    gauss = np.random.normal(mean,1,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy
  
  elif noise_typ == "s&p":

    s_vs_p = 0.5
    amount = 0.04
    out = image
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    print(num_salt)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 255
   # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    print(num_pepper)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out
  
  elif noise_typ == "poisson":
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image * vals) / float(vals)
    return noisy
  elif noise_typ =="speckle":

    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch)        
    noisy = image + image * gauss
    return noisy


img=cv2.imread("aviao.jpg",0)
# Filtro de media (blur)

filter_blur=cv2.blur(img,ksize=(3,3))
cv2.imshow("Filtro Media",filter_blur)
cv2.waitKey()

#Filtro Gaussiano
gaugaussian=cv2.GaussianBlur(src=img,ksize=(5,5),sigmaX=0)
cv2.imshow("Filtro Gaussiano",gaugaussian)
cv2.waitKey()

#Filtro MedianBlur (mediana)
imagemSP = noisy("s&p",img)
cv2.imshow("imagem COM RUIDO SALT PEPPER",imagemSP)
cv2.waitKey()
median=cv2.medianBlur(imagemSP,3)
cv2.imshow("Filtro MedianBlur",median)
cv2.waitKey()

#Filtro Bilateral
img=cv2.imread("aviao.jpg",0)
bilateral=cv2.bilateralFilter(img,d=5,sigmaColor=75,sigmaSpace=75)
cv2.imshow("Filtro Bilateral",bilateral)
cv2.waitKey()
