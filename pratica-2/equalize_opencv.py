import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('wiki.png',0)
equ = cv2.equalizeHist(img)  #retorna a imagem equalizada
#histograma e cdf  da imagem original
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.subplot(221)
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.title('Histograma imagem original'), plt.xticks([]), plt.yticks([])
#histograma e cdf da imagem equalizada
hist,bins = np.histogram(equ.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
plt.subplot(222)
plt.plot(cdf_normalized, color = 'b')
plt.hist(equ.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.title('Histograma imagem equalizada '), plt.xticks([]), plt.yticks([])
plt.subplot(223)
# o formato opencv é BGR, e o plot usa RGB, entao usa COLOR_BGR2RGB
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(224)
plt.imshow(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
plt.show()


