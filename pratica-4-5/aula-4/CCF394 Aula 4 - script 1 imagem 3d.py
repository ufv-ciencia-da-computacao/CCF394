"""
Apresentando pixels 3d
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
#To make the plot, you will need a few more Matplotlib libraries:

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
#Those libraries provide the functionalities you need for the plot. You want to place each pixel in its location based on its components and color it by its color. OpenCV split() is very handy here; it splits an image into its component channels. These few lines of code split the image and set up the 3D plot:
imagem=cv2.imread("pilotosPequeno.png")
scale_percent = 25 # percent of original size
width = int(imagem.shape[1] * scale_percent / 100)
height = int(imagem.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
imagem = cv2.resize(imagem, dim, interpolation = cv2.INTER_AREA)
r, g, b = cv2.split(imagem)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
#Now that you have set up the plot, you need to set up the pixel colors. In order to color each pixel according to its true color, there’s a bit of reshaping and normalization required. It looks messy, but essentially you need the colors corresponding to every pixel in the image to be flattened into a list and normalized, so that they can be passed to the facecolors parameter of Matplotlib scatter().

#Normalizing just means condensing the range of colors from 0-255 to 0-1 as required for the facecolors parameter. Lastly, facecolors wants a list, not an NumPy array:

pixel_colors = imagem.reshape((np.shape(imagem)[0]*np.shape(imagem)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
#Now we have all the components ready for plotting: the pixel positions for each axis and their corresponding colors, in the format facecolors expects. You can build the scatter plot and view it:

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.title("espaço RGB")
plt.show()
#fig.clear()
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(imagem_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.title("espaço HSV")
plt.show()
