import argparse
import numpy as np
import cv2

def HD (image):
  hdrImage = cv2.detailEnhance(image, sigma_s = 12, sigma_r = 0.15)
  
  return hdrImage

def pencil (image):
  sk_gray, skColor = cv2.pencilSketch(image, sigma_s = 60, sigma_r = 0.07, shade_factor = 0.1)

  return skColor

def sepia (image):
  # Convert to float to prevent loss
  sepiaImage = np.array(image, dtype = np.float64)
  sepiaImage = cv2.transform(sepiaImage, np.matrix([[0.272, 0.543, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]))
  sepiaImage[np.where(sepiaImage > 255)] = 255
  sepiaImage = np.array(sepiaImage, dtype = np.uint8)

  return sepiaImage

def sharpen (image):
  kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
  sharpenedImage = cv2.filter2D(image, -1, kernel)

  return sharpenedImage

def brightness (image, betaValue):
  brightImage = cv2.convertScaleAbs(image, beta = betaValue)

  return brightImage

def grayScale (image):
  grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return grayImage

def invert (image):
  invertedImage = cv2.bitwise_not(image)

  return invertedImage 

def cartoonize (image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurImage = cv2.medianBlur(image, 1)

  edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

  color = cv2.bilateralFilter(image, 9, 200, 200)

  cartoon = cv2.bitwise_and(color, color, mask = edges)

  return cartoon

webcam = cv2.VideoCapture(0) #instancia o uso da webcam
janela = "Tela de captura"
cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE) #cria uma janela
 
#faz a leitura inicial de imagens

while True:
  ret, image = webcam.read()
  cartoonImage = cartoonize(image)
  invertedImage = invert(image)
  grayImage = grayScale(image)
  brightImage = brightness(image, 60)
  darkerImage = brightness(image, -60)
  sharperImage = sharpen(image) 
  sepiaImage = sepia(image)
  pencilImage = pencil(image)
  hdrImage = HD(image)
  cv2.imshow("output.jpg", cartoonImage)
  cv2.imshow("inverted.jpg", invertedImage)
  cv2.imshow("grayscale.jpg", grayImage);
  cv2.imshow("brighter.jpg", brightImage)
  cv2.imshow("darker.jpg", darkerImage)
  cv2.imshow("sharper.jpg", sharperImage)
  cv2.imshow("sepia.jpg", sepiaImage)
  cv2.imshow("pencil.jpg", pencilImage)
  cv2.imshow("hdrImage.jpg", hdrImage)
  cv2.imshow("output", hdrImage)

  cv2.waitKey(100)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyWindow(janela)
    break
 cv2.destroyAllWindows()
