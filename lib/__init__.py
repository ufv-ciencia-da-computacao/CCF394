import cv2
import numpy as np
from matplotlib import pyplot as plt
from __future__ import print_function
from __future__ import division
from skimage.filters import threshold_local
import argparse

## Transformations
def rotate_image(img, angle):
    lin, col, dim = img.shape
    (cX, cY) = (lin//2, col//2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    return cv2.warpAffine(img, M, (lin, col))

def linear_transform(img, alpha, beta):
    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return new_image

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


## Thresholding and Binarizing
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


def adaptative_thresholding_cv2(img, neighbourhood_size, constant_c, filepath):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                neighbourhood_size, constant_c)

    return thresh

def adaptative_thresholding_scikit(img, neighbourhood_size, constant_c, filepath):
    threshold_value = threshold_local(img, neighbourhood_size, offset=constant_c)
    # np.uint8 devolve a matriz para a faixa de 8 bits
    thresh = (img < threshold_value).astype(np.uint8) * 255

    return thresh

## Noise
def gaussian_noise(size, mean=0, std=0.01):
    noise = np.multiply(np.random.normal(mean, std, size), 255)
    return noise

def impulsive_noise(image, prob=0.1, mode='salt_and_pepper'):
    noise = np.array(image, copy=True)
    for x in np.arange(image.shape[0]):
        for y in np.arange(image.shape[1]):
            rnd = np.random.random()
            if rnd < prob:
                rnd = np.random.random()
                if rnd > 0.5:
                    noise[x,y] = 255
                else:
                    noise[x,y] = 0
    
    return noise


## Filters
def sobel_filter(img):
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    filtered_image_xy = cv2.convertScaleAbs(sobelxy)
    
    return filtered_image_xy

def roberts_filter(img):
    roberts_cross_v = np.array([[1, 0 ],
                           [0,-1 ]] )
  
    roberts_cross_h = np.array([[ 0, 1 ],
                                 [ -1, 0 ]])
    img = np.float64(img)
    img /= 255.
    
    from scipy import ndimage
    
    vertical = ndimage.convolve( img, roberts_cross_v )
    horizontal = ndimage.convolve( img, roberts_cross_h )

    edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
    edged_img*=255
    return edged_img

def prewitt_filter(img):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(img, -1, kernelx)
    img_prewitty = cv2.filter2D(img, -1, kernely)
    img_prewitt=img_prewittx + img_prewitty
    return img_prewitt


## Plotting
def plot_images(imgs, titles, x, y, figsize, cmap="viridis"):
    images_list_w_titles = list(zip(imgs, titles))
    
    f, axarr = plt.subplots(x,y, figsize=figsize)
    for i in range(x):
        for j in range(y):
            if x > 1:
                axarr[i, j].imshow(np.uint8(images_list_w_titles[(i*y)+j][0]), cmap=cmap)
                axarr[i, j].set_title(images_list_w_titles[(i*y)+j][1])
            else:
                axarr[j].imshow(np.uint8(images_list_w_titles[(i*y)+j][0]), cmap=cmap)
                axarr[j].set_title(images_list_w_titles[(i*y)+j][1])

## SpaceColor
def get_spacecolor(original):
    scale_percent = 60 # percent of original size
    width = int(original.shape[1] * scale_percent / 100)
    height = int(original.shape[0] * scale_percent / 100)
    dim = (width, height)

    original = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
    
    # Convert the BGR image to other color spaces
    imageBGR = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
    imageHSV = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
    imageLAB = cv2.cvtColor(original,cv2.COLOR_BGR2LAB)
    return imageBGR, imageHSV, imageLAB


# Video
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


# Histograma
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

def gera_histograma(image,title):
    hist_img,_ = np.histogram(image, bins=256, range=(0,255))
    plt.figure(figsize=(14,4))
    plt.subplot(121)
    plt.title(title)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.axis('off')
    plt.subplot(122)
    plt.bar(np.arange(256), hist_img) 
    
