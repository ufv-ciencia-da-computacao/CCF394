import numpy as np
import numpy.linalg as la
import cv2
#https://www.lcg.ufrj.br/marroquim/courses/cos756/trabalhos/2012/igor-ramos-taisa-martins/igor-ramos-taisa-martins-report.pdf
import math
import matplotlib.pyplot as plt
import time


# Count runtime



def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global pixels
        (H,S,V)=imghsv[y,x]
        print ("col: %d, row: %d   - H= %d, S=%d, V=%d" % (x, y, H,S,V))
        x=np.array([[H,S,V]])
        print(x)
        pixels=np.concatenate((pixels,x))
        
        
        
pixels =np.zeros((1,3),dtype=np.int8)
print(pixels)
img=cv2.imread("pilotosPEQUENO.png")
imghsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.namedWindow('frame1')
cv2.setMouseCallback("frame1", on_mouse)
CONTADOR=0
while True:
    # display the image and wait for a keypress
    cv2.imshow("frame1", imghsv)
    key = cv2.waitKey(1) & 0xFF
    # if the 'c' key is pressed, break from the loop
    if key == ord("c"):
        break
#    CONTADOR=CONTADOR+1
#    if CONTADOR>10:
#        break
pixels = pixels[1:] #remove 000 inicial
media=np.mean(pixels,axis=0)
mediaH=media[0]
mediaS=media[1]
mediaV=media[2]
print("Media H=%d, media S= %d, media V=%d \n " % (mediaH, mediaS, mediaV))
rows,cols,cor = img.shape
print(rows, cols)

for i in range(rows):
    for j in range(cols):
        #print(i,j)
        (H,S,V) = imghsv[i,j]
        #print(H,S,V)
        #if (((S<(mediaS+20)) and (S>(mediaS-20))) and ((V<(mediaV+20)) and (V>(mediaV-20))) and (H<(mediaH+13)) and (H>(mediaH-13))  ):
# para chapeu do senha        if ((H<(105) and (H>98))):
        if (   (((H>9) and (H<20)))    and   (((S>100) and (S<150)))) :          

             img[i,j]=(0,0,0)
              
    #print(i)
cv2.imshow("frame12",img)
cv2.waitKey(0)
              

cv2.destroyAllWindows()
csv_rows = ["{},{},{}".format(i, j, k) for i, j, k in pixels]
csv_text = "\n".join(csv_rows)

with open('pilotospequenos.csv', 'w') as f:
    f.write(csv_text)


    
    

