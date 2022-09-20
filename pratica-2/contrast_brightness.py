import cv2
import numpy as np

def histograma(src):
    
  ## [Separate the image in 3 places ( B, G and R )]
  bgr_planes = cv2.split(src)
  ## [Separate the image in 3 places ( B, G and R )]

  ## [Establish the number of bins]
  histSize = 256
  ## [Establish the number of bins]

  ## [Set the ranges ( for B,G,R) )]
  histRange = (0, 256) # the upper boundary is exclusive
  ## [Set the ranges ( for B,G,R) )]

  ## [Set histogram param]
  accumulate = False
  ## [Set histogram param]

  ## [Compute the histograms]
  b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
  g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
  r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
  ## [Compute the histograms]

  ## [Draw the histograms for B, G and R]
  hist_w = 512
  hist_h = 400
  bin_w = int(round( hist_w/histSize ))

  histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
  ## [Draw the histograms for B, G and R]

  ## [Normalize the result to ( 0, histImage.rows )]
  cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
  cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
  cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
  ## [Normalize the result to ( 0, histImage.rows )]
  ## [Draw for each channel]
  for i in range(1, histSize):
      cv2.line(histImage, ( bin_w*(i-1), hist_h - int((b_hist[i-1])) ),
              ( bin_w*(i), hist_h - int((b_hist[i])) ),
              ( 255, 0, 0), thickness=2)
      cv2.line(histImage, ( bin_w*(i-1), hist_h - int((g_hist[i-1])) ),
              ( bin_w*(i), hist_h - int((g_hist[i])) ),
              ( 0, 255, 0), thickness=2)
      cv2.line(histImage, ( bin_w*(i-1), hist_h - int((r_hist[i-1])) ),
              ( bin_w*(i), hist_h - int((r_hist[i])) ),
              ( 0, 0, 255), thickness=2)
  ## [Draw for each channel]
  ## [Display]
  cv2.imshow('calcHist Demo', histImage)
  
  ## [Display]

def BrilhoContraste(Brilho=0):
 
    # getTrackbarPos returns the
    # current position of the specified trackbar.
    Brilho = cv2.getTrackbarPos('Brilho','CCF394')
     
    Contraste = cv2.getTrackbarPos('Contraste','CCF394')
     
    effect = controller(img,Brilho, Contraste)
 
    cv2.imshow('Effect', effect)
    histograma(effect)

def controller(img, Brilho=255, Contraste=127):
    Brilho = int((Brilho - 0) * (255 - (-255)) / (510 - 0) + (-255))
    Contraste = int((Contraste - 0) * (127 - (-127)) / (254 - 0) + (-127))
 
    if Brilho != 0:
         if Brilho > 0:
             shadow = Brilho
             max = 255
         else:
            shadow = 0
            max = 255 + Brilho
         al_pha = (max - shadow) / 255
         ga_mma = shadow
 
        # The function addWeighted calculates the weighted sum of two arrays
         cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma)
 
    else:
        cal = img
 
    if Contraste != 0:
        Alpha = float(131 * (Contraste + 127)) / (127 * (131 - Contraste))
        Gamma = 127 * (1 - Alpha)
 

        cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma)

    
 
    return cal
  
if __name__ == '__main__':
    # The function imread loads an
    # image from the specified file and returns it.
    original = cv2.imread("lena.jpg")
 
    # Making another copy of an image.
    img = original.copy()
 
    # The function namedWindow creates
    # a window that can be used as
    # a placeholder for images.
    cv2.namedWindow('CCF394')
 
  
    cv2.imshow('CCF394', original)
 
    # createTrackbar(trackbarName, windowName, value, count, onChange)
    # Brilho range -255 to 255
    cv2.createTrackbar('Brilho', 'CCF394',255, 2 * 255, BrilhoContraste)
     
    # Contraste range -127 to 127
    cv2.createTrackbar('Contraste', 'CCF394',127, 2 * 127, BrilhoContraste) 
     
    BrilhoContraste(0)
 

cv2.waitKey(0)
