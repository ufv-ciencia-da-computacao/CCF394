#https://colab.research.google.com/drive/1LW2aCbeGpDT-aE356pkl4BURJbC16WUl?usp=sharing#scrollTo=yH76Rjr34-4a

import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

def uniform_noise(size, prob=0.1):
    
    levels = int((prob * 255) // 2)
    noise = np.random.randint(-levels, levels, size)
    
    return noise


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


img = cv2.imread("input-300x200.webp",0)
np.unique(img)

uni_noise = uniform_noise(img.shape, prob=0.1)
img_uni = np.clip(img.astype(int)+uni_noise, 0, 255)

hist_img,_ = np.histogram(img, bins=256, range=(0,255))
hist_uni,_ = np.histogram(img_uni, bins=256, range=(0,255))

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title("Noise Free")
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(122)
plt.bar(np.arange(256), hist_img)

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('Uniform Noise')
plt.imshow(img_uni, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(122)
plt.bar(np.arange(256), hist_uni)

#print(np.unique(img_uni))

##########################################

# creating the noise matrix to be added
gau_noise = gaussian_noise(img.shape, mean=0, std=0.05)

# adding and clipping values below 0 or above 255
img_gau = np.clip(img.astype(int)+gau_noise, 0, 255)

hist_gau,_ = np.histogram(img_gau, bins=256, range=(0,255))

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('Gaussian Noise')
plt.imshow(img_gau, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(122)
plt.bar(np.arange(256), hist_gau)

#############################################

img_imp = impulsive_noise(img, prob=0.1)

hist_imp,_ = np.histogram(img_imp, bins=256, range=(0,255))

plt.figure(figsize=(10,4))
plt.subplot(121)
plt.title('Impulsive Noise')
plt.imshow(img_imp, cmap="gray", vmin=0, vmax=255)
plt.axis('off')
plt.subplot(122)
plt.bar(np.arange(256), hist_imp)
plt.show()
