import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
image = cv.imread("lena.jpg",cv.IMREAD_GRAYSCALE)
print(image.shape)
try:
    x,y,z=image.shape
except ValueError:
    print("imagem 2d")
    img2d=True

if image is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
min=np.min(image)
max=np.max(image)
print(max,min)
# um deslocamento de 20 na imagem
nova=(image+20)*(255/(max-min))
nova=np.uint8(nova)
print(np.max(nova),np.min(nova))



plt.subplot(221), plt.imshow(image,cmap='gray')
plt.subplot(222), plt.imshow(nova,cmap='gray')

plt.subplot(223), plt.hist(image.ravel(),256,[0,256]),plt.title('Histograma para uma imagem em tons de cinza')
plt.subplot(224), plt.hist(nova.ravel(),256,[0,256]),plt.title('aplicando uma transformacao linear')

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
# agora, lendo os valores de contraste e brilho do teclado g(x) =alfa*f(x)  + betaa
new_image = np.zeros(image.shape, image.dtype)
alpha = 1.0 # Simple contrast control
beta = 0    # Simple brightness control
# Initialize values
print(' Basic Linear Transforms ')
print('-------------------------')
try:
    alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
    beta = int(input('* Enter the beta value [0-100]: '))
except ValueError:
    print('Error, not a number')
# Do the operation new_image(i,j) = alpha*image(i,j) + beta
# Instead of these 'for' loops we could have used simply:
# new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
# but we wanted to show you how to access the pixels :)
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        if (img2d):
            new_image[y,x] = np.clip(alpha*image[y,x] + beta, 0, 255)
        else:
           
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

plt.subplot(221), plt.imshow(image,cmap='gray')
plt.subplot(222), plt.imshow(new_image,cmap='gray')

plt.subplot(223), plt.hist(image.ravel(),256,[0,256]),plt.title('Histograma para uma imagem em tons de cinza')
plt.subplot(224), plt.hist(new_image.ravel(),256,[0,256]),plt.title('aplicando uma transformacao linear')

plt.show()
cv.waitKey(0)

#cv.imshow('Original Image', image)
#cv.imshow('New Image', new_image)
# Wait until user press some key
cv.waitKey()
cv.destroyAllWindows()
