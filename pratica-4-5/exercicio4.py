import cv2
import numpy as np

cap = cv2.VideoCapture("images/cruXinter.mp4")
# ret , frame = cap.read()
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# print(frame_width, frame_height)

while(True):

    # Take each frame
    ret , frame = cap.read()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    vid_writer = cv2.VideoWriter('saida.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))
        
    if ret == True:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        kernel = np.ones((5, 5), np.uint8)

        # lower_blue = np.array([177, 76, 52])
        # upper_blue = np.array([218, 114, 93])

        lower_blue = np.array([241,150,70])
        upper_blue = np.array([300,200,200])
        mask = cv2.inRange(frame, lower_blue, upper_blue)
        mask2=cv2.dilate(mask, kernel, 3)

        lower_red = np.array([0,0,70])
        upper_red = np.array([60,70,255])
        

        mask_red = cv2.inRange(frame, lower_red, upper_red)
        mask2_red=cv2.dilate(mask_red, kernel, 3)


        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask= (mask2 | mask2_red))


        cv2.imshow('frame', frame)
        # cv2.imshow('mask', mask)
        # cv2.imshow('res', res)
        cv2.imshow('erode/dilate', res)
        
        vid_writer.write((res).astype(np.uint8))
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cap.release()
            break
vid_writer.release()
cv2.destroyAllWindows()