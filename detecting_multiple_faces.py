import cv2 as cv
import numpy as np
from numba import cuda
from filters import *



filter_list = [increase_brightness,decrease_brightness,sharpened,fft,frame_mixing,frame_mixing2,frame_edges,
               fft2,frame_mixing21,frame_mixing22,fft3,frame_mixing3,frame_mixing32]


# Real-time Tracking algorithm using LBP filter

##  making a LBP filter for real time tracking of faces
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

## Function to make a LBP histogram 
def get_histogram(lbpface):
    histogram = []
    for i in range(256):
        histogram.append(np.count_nonzero(lbpface == i))
    return histogram

## Function to get the object tracking by returning object id 
def get_id(curr,previous,count):
    mini_id = -1
    mini = 2500000
    for id,img in previous.items():
        eclu = 0
        for j in range(255):
            eclu += (img[j]-curr[j])**2
        if mini > np.sqrt(eclu):
            mini = np.sqrt(eclu)
            mini_id = id
        
    if mini != 2500000 and mini > 800:
        previous[count] = curr
        count+=1
        return count, count
    else:
        previous[mini_id] = curr
        return mini_id+1, count   


if __name__ == '__main__':
    cam = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    fps = cam.get(cv.CAP_PROP_FPS)
    previous_img = dict()
    count = 0

    font = cv.FONT_HERSHEY_SIMPLEX
    
    MAX_FRAME_TO_DETECT = fps*5

    ids_count = dict()
    ids_filter = dict()
    frame_count = 0
    while True:
        ret, frame = cam.read()
        frame = cv.flip(frame,1)
        ori = np.copy(frame)
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
        
        frame1 = frame
        
        idx = []
        for (x,y,w,h) in faces:        
            frame1 = cv.rectangle(frame1,(x,y),(x+w,y+h),color=(0,200,255),thickness=5)
            face = cv.resize(gray[y:y+h,x:x+w],(170,170))
            
            lbpface = cuda.to_device(np.zeros_like(face))
            face = cuda.to_device(face)
            lbp[128,128](face,lbpface)
            
            lbpface = lbpface.copy_to_host()
            
            if len(previous_img) != 0:
                id, count = get_id(get_histogram(lbpface),previous_img,count)
                frame1 = cv.putText(frame1, f'{id}', (x,y-20), font,  1, (255, 0, 0), 2)    
                idx.append(id)   
            else:
                previous_img[count] = get_histogram(lbpface) 
                count+=1
            
            ffit = 0
            try:
                ffit = ids_filter[id](frame[y:y+h,x:x+w])
            except:
                ids_filter[id] = filter_list[np.random.choice(len(filter_list),1)[0]]
            frame1[y:y+h,x:x+w] = ffit
                
                
                
        delete_arr = []
        for id in previous_img.keys():
            if id not in idx:
                try:
                    ids_count[id] += 1
                except:
                    ids_count[id] = 1
                if ids_count[id] >= MAX_FRAME_TO_DETECT:
                    delete_arr.append(id) 
            else:
                ids_count[id] = 0
        
        # Deletion Loop
        for i in delete_arr:
            del previous_img[i]
            del ids_count[i]
         
        cv.imshow("original",ori)
        cv.imshow('Face Hidden',frame1)
        # cv.imshow('LBP_filter',imgLBP)
        k = cv.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord(' '):
            cv.imwrite("output/out.jpg", frame1)
    cam.release()

    
    