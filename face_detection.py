import numpy as np
import cv2

#new_path = '/Users/.../Anaconda/Library/etc/haarcascades/'

#face_cas = cv2.CascadeClassifier(new_path + 'haarcascade_frontalface_default.xml')
#face_cas = cv2.CascadeClassifier('/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
#face_cas = getClass().getResource("/haarcascade_frontalface_alt.xml").getPath()
cam = cv2.VideoCapture(0)
#face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')   
face_cas = cv2.CascadeClassifier('/Users/DBRSJ/anaconda/pkgs/opencv-3.2.0-np112py36_0/share/OpenCV/haarcascades/haarcascade_frontalcatface_extended.xml') 
data = []
ix = 0

while True:
    ret, frame = cam.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray,1.3,5)
        
        for(x,y,w,h) in faces :
            face_component = frame[y:y+h,x:x+w,:]
            
            fc = cv2.resize(face_component,(50,50))
            
            if ix%10 == 0 and len(data) <20:
                data.append(fc)
                
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)
        ix += 1
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == 27 or len(data) >= 20:
            break
    else :
        print("error")

cv2.destroyAllWindows()
data = np.asarray(data)

print(data.shape)
np.save('face_01', data)

