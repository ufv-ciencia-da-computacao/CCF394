


"""
Comandos Básicos para o Procedimento de filtragem
img=double(img) --> Converte a imagem para a classe double
fft2 ---> Transformada de fourier do domínio espacial para o domínio das frequencias
fftshift ---> Inverte os quadrantes da imagem no domínio das frequencias.
img .* filtro---> Multiplica pixel por pixel pontualmente (.) pelo filtro (Operação de Filtragem)
fftshift ---> Re-inverte os quadrantes da imagem no domínio das frequencias.
ifft2 ---> Transformada inversa de fourier do domínio das frequencias para o domínio espacial
img=uint8(img) --> Converte a imagem para a classe double
imshow(img) ---> Mostra a imagem na tela
figure ---> Abre mais uma janela para novas figuras
Dica: Sempre que tiver dúvias sobre o funcionamento de determinada função, digite na linha de
comando help "função"
(ex: help fft2, help imshow)




"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("lena_impulsiva.png",0)
cv2.imshow("t",img)
cv2.waitKey(0)

rows, cols = img.shape
crow = int(rows/2)
ccol = int(cols/2)

img_float32 = np.float32(img)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()




dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)


# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title("Imagem FIltrada"), plt.xticks([]), plt.yticks([])

plt.show() 

