import cv2 as cv
import numpy as np
from numba import cuda

cam = cv.VideoCapture(0)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

filter = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

bfilter = np.array([
    [-5,0,5],
    [-1,0,1],
    [-5,0,5]
])

@cuda.jit
def lbp(img,imgLBP):
    x,y = cuda.grid(2)
    stridex, stridey = cuda.gridsize(2)
    
    for i in range(x,img.shape[0]-3,stridex):
        for j in range(y,img.shape[1]-3,stridey):
            pattern = 0
            center = img[i+1,j+1]
            pattern += (img[i +2 , j +2] >= center) << 7
            pattern += (img[i +2, j+1] >= center) << 6
            pattern += (img[i +2, j ] >= center) << 5
            pattern += (img[i+1 ,j ] >= center) << 4
            pattern += (img[i , j ] >= center) << 3
            pattern += (img[i , j+1] >= center) << 2
            pattern += (img[i , j+2 ] >= center) << 1
            pattern += (img[i + 1, j + 2] >= center) << 0
            imgLBP[i+1,j+1] = pattern

def track(pos,previous):
    min = []
    for i in previous:
        min.append()
       

previous_img = []
count = 0

font = cv.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2

while True:
    ret, frame = cam.read()
    frame = cv.flip(frame,1)
    ori = np.copy(frame)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
    
    # imgLBP = np.zeros_like(gray)
    # gray = cuda.to_device(gray)
    # imgLBP = cuda.to_device(imgLBP)

    # lbp[128,128](gray,imgLBP)

    # imgLBP = imgLBP.copy_to_host()
    # imgLBP = imgLBP.astype(np.uint8)
    
    
    # frame1 = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame1 = (frame + cv.filter2D(frame,-1,bfilter)).astype(np.uint8)
    
    for (x,y,w,h) in faces:        
        frame1 = cv.rectangle(frame1,(x,y),(x+w,y+h),color=(0,200,255),thickness=5)
        face = cv.resize(gray[y:y+h,x:x+w],(170,170))
        
        lbpface = cuda.to_device(np.zeros_like(face))
        face = cuda.to_device(face)
        lbp[128,128](face,lbpface)
        
        lbpface = lbpface.copy_to_host()
        
        if count != 0:
            track((x,y,w,h),previous_img)
        else:
            previous_img = faces
            count+=1
            
        frame1[y+1:y+h-1,x+1:x+w-1] = cv.resize(lbpface,(h-2,w-2)).reshape(h-2,w-2,1)
        
    cv.imshow("original",ori)
    cv.imshow('Face Hidden',frame1)
    # cv.imshow('LBP_filter',imgLBP)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()

    
    