import numpy as np
import cv2

import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')


cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('/Users/DBRSJ/anaconda/pkgs/opencv-3.2.0-np112py36_0/share/OpenCV/haarcascades/haarcascade_frontalcatface_extended.xml') 

font = cv2.FONT_HERSHEY_SIMPLEX


f_01 = np.load('./face_01.npy').reshape((20,50*50*3))

print (f_01.shape)

names = {
	0 : 'Jasneet'
}

labels = np.zeros((20,1))
labels[:,:] = 0.0

data = np.concatenate([f_01])

print(data.shape , labels.shape)


def distance(x1,x2):
	return np.sqrt(((x1-x2)**2).sum())

def knn(x,train, targets, k=5):
	m =train.shape[0]
	dist = []

	for ix in range(m):
		dist.append(distance(x,train[ix]))
	dist = np.asarray(dist)
	indx = np.argsort(dist)
	sorted_labels = labels[indx][:k]
	counts = np.unique(sorted_labels, return_counts = True)
	return counts[0][np.argmax(counts[1])]

while True:
	ret, frame = cam.read()

	if ret == True: 
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cas.detectMultiScale(gray,1.3,5)

		for(x,y,w,h) in faces :
			face_component = frame[y:y+h,x:x+w,:]
			fc = cv2.resize(face_component, (50,50))

			lab = knn(fc.flatten(),data ,labels)
			text = names[int(lab)]
			cv2.putText(frame,text, (x,y), font, 1, (255,255,0),2)       	

			cv2.rectangle(frame , (x,y),(x+w, y+h),(0,0,255),2)
		cv2.imshow('frame',frame)

		if cv2.waitKey(1) == 27:
			break

	else :
		print("Error")


cv2.destroyAllWindows()



