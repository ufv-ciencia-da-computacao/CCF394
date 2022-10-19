import cv2
import numpy as np

cap = cv2.VideoCapture(0)



# Making window size adjustable
cv2.namedWindow('image', cv2.WINDOW_NORMAL)






while(1):
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    th2=cv2.bitwise_not(th2, mask = None)
    b,g,r = cv2.split(frame)
    dest_andb = cv2.bitwise_and(b, th2, mask = None)
    dest_andg = cv2.bitwise_and(g, th2, mask = None)
    dest_andr = cv2.bitwise_and(r, th2, mask = None)
    frame=cv2.merge([dest_andb,dest_andg,dest_andr])
    cv2.imshow('image',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
""" 
  
    # Now this piece of code is just for smooth drawing. (Optional)
    _ , mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 
    255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
    background = cv2.bitwise_and(frame, frame,
    mask = cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)
"""
cv2.destroyAllWindows()
cap.release()
