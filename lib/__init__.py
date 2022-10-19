import cv2
import numpy as np
from matplotlib import pyplot as plt
from __future__ import print_function
from __future__ import division
from skimage.filters import threshold_local
import argparse

def rotate_image(img, angle):
    lin, col, dim = img.shape
    (cX, cY) = (lin//2, col//2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    return cv2.warpAffine(img, M, (lin, col))

def change_pixel_by_pixel(img, g_bounds: tuple, b_bounds: tuple, r_bounds: tuple, new_b, new_g, new_r):
    [col, lin, dim] = img.shape
    img2 = img.copy()

    for j in range(0, col-1):
        for i in range(0, lin-1):
            (b, g, r) = img[j, i]
            if (b>b_bounds[0] and b<b_bounds[1]) and (g>g_bounds[0] and g<g_bounds[1]) and (r>r_bounds[0] and r<r_bounds[1]):
                b=new_b; g=new_g; r=new_r

                img2[j,i,] = np.array([b,g,r])
            else:
                img2[j,i,] = np.array([0,0,0])
    return img2



def capture_video(output_file_path):
    camera = cv2.VideoCapture(0)
    outputFile = output_file_path

    (sucesso, frame) = camera.read()

    vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (frame.shape[1], frame.shape[0]))

    while True:
        (sucesso, frame) = camera.read()
        if not sucesso:
            break

        cv2.imshow("Exibindo video", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    vid_writer.release()
    cv2.destroyAllWindows()


def adaptative_thresholding_cv2(img, neighbourhood_size, constant_c, filepath):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                neighbourhood_size, constant_c)
    cv2.imshow("OpenCV Mean Threshold", thresh)
    cv2.waitkey(0)
    return thresh

def adaptative_thresholding_scikit(img, neighbourhood_size, constant_c, filepath):
    threshold_value = threshold_local(gray, neighbourhood_size, offset=constant_c)
    # np.uint8 devolve a matriz para a faixa de 8 bits
    thresh = (gray < threshold_value).astype(np.uint8) * 255
    cv2.imshow("Scikit-image Mean Threshold", thresh)
    cv2.waitKey(0)
    return thresh


def histogram_colored(img, bins, hist_range, accumulate=False):
    bgr_planes = img
    b_hist = cv2.calcHist(bgr_planes, [0], None, [bins], hist_range, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [bins], hist_range, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [bins], hist_range, accumulate=accumulate)
    
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/bins ))

    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    ## [Normalize the result to ( 0, histImage.rows )]
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

    for i in range(1, bins):
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int((b_hist[i-1])) ),
                ( bin_w*(i), hist_h - int((b_hist[i])) ),
                ( 255, 0, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int((g_hist[i-1])) ),
                ( bin_w*(i), hist_h - int((g_hist[i])) ),
                ( 0, 255, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int((r_hist[i-1])) ),
                ( bin_w*(i), hist_h - int((r_hist[i])) ),
                ( 0, 0, 255), thickness=2)

    return histImage

def hist_cdf_by_flattening(img, equalized=False, bins=256, range_ = [0,256]):
    if equalized:
        img = cv2.equalizeHist(img)
    hist, bins = np.histogram(img.flatten(), bins, range_)

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    return hist, cdf, cdf_normalized



def set_britghtness_and_contrast(image, brightnesss, contrast):
    effect = controller(image, brightnesss, contrast)
    return effect 

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

def thresholding(image, threshold, max_value, comb=[]): 
    """
    image: imagem a ser binarizada
    threshold: valor de limiar
    max_value: valor máximo de pixel
    comb: combinação de canais de cores
        - cv2.THRESH_BINARY
        - cv2.THRESH_BINARY_INV
        - cv2.THRESH_TRUNC
        - cv2.THRESH_TOZERO
        - cv2.THRESH_TOZERO_INV
        - cv2.THRESH_OTSU
        - cv2.THRESH_TRIANGLE
        - cv2.THRESH_MASK
    """
    if not comb:
        _, output = cv2.threshold(image, threshold, max_value,cv2.THRESH_OTSU)
    else:
        _, output = cv2.threshold(image, threshold, max_value, sum(comb))
    return output


def linear_transform(img, alpha, beta):
    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_image

def plot_image(images, plots, titles, cmap=None):
    fig = plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = fig.add_subplot(plots[0], plots[1], i+1)
        ax.imshow(images[i], cmap=cmap)
        ax.set_title(titles[i])
        ax.axis('off')
    plt.show()
    
