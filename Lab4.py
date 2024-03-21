import cv2
import numpy as np

cam = cv2.VideoCapture("Lane_Detection_Test_Video_01.mp4")

NEW_WIDTH = 480
NEW_HEIGHT = 270




upper_left = (NEW_WIDTH*0.55 - 45, NEW_HEIGHT*0.55 + 50)
bottom_right = (NEW_WIDTH, NEW_HEIGHT)
upper_right = (NEW_WIDTH*0.55 -10, NEW_HEIGHT*0.55 + 50)
bottom_left = (0, NEW_HEIGHT)

trapez = np.array([upper_left, upper_right, bottom_right, bottom_left], dtype=np.int32)

empty_frame = np.zeros((NEW_HEIGHT, NEW_WIDTH), dtype=np.uint8)

cv2.fillConvexPoly(empty_frame, trapez, 1)

while True:
    ret, frame = cam.read()
    
    if ret is False:
        break
    width, height, _ = frame.shape
    
    
    frame = cv2.resize(frame,(NEW_WIDTH, NEW_HEIGHT))
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    road_frame = frame * empty_frame

    cv2.imshow("Road", road_frame)
   
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()