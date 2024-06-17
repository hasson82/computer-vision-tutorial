import numpy as np
import cv2

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(blank_img,(x,y),100,(0,0,255),-1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(blank_img,(x,y),100,(255,0,0),-1)

# Global variables
drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def draw_line(event,x,y,flags,param):
    global ix,iy,drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(blank_img,(ix,iy),(x,y),(0,255,0),10)
            ix,iy = x,y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(blank_img,(ix,iy),(x,y),(0,255,0),10)


cv2.namedWindow(winname='my_drawing')
cv2.setMouseCallback('my_drawing',draw_line)

blank_img = np.zeros((512,512,3))
while True:
    cv2.imshow('my_drawing',blank_img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()