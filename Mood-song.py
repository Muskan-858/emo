import tensorflow
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import load_model
import webbrowser
import random
import sys
a=str(random.randint(1,5))
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('EmotionDetectionModel.h5',compile= False)
arr = []
arrcount = []
class_labels=['Angry','Happy','Neutral','Sad','Surprise']
cap=cv2.VideoCapture(0)
for counter in range(0,50):
    ret,frame=cap.read()
    print(frame.shape)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            preds=classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            for i in range(0, 12):
            	arr.append(label)
            	break
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for i in class_labels:
	arrcount.append(arr.count(i))
emo_dom = class_labels[arrcount.index(max(arrcount))] #dominant emotion after itteration
print("Emotion:" + emo_dom)
def sad():
	inp = int(input("Enter the choice:\n1.Sad songs\n2.Happy songs\nchoice: "))
	if inp==1:
		webbrowser.open("https://www.youtube.com/watch?v=lN1m7zLBbSU&list=PL3ca10hBnzZsXa1Ovgifsomg34hR3HtTn&index="+a)
	elif inp==2:
		webbrowser.open("https://www.youtube.com/watch?v=k4yXQkG2s1E&list=PL3ca10hBnzZsYK5dOlCpmCX7yIp4HjsYz&index="+a)
	else:
		print("wrong choice")
	sys.exit()
def happy():
	webbrowser.open("https://www.youtube.com/watch?v=k4yXQkG2s1E&list=PL3ca10hBnzZsYK5dOlCpmCX7yIp4HjsYz&index="+a)
	sys.exit()
def surprise():
	inp = int(input("Enter the choice:\n1.Surprise songs\n2.Happy songs\nchoice: "))
	if inp==1:
		webbrowser.open("https://www.youtube.com/watch?v=2Vv-BfVoq4g&list=PL3ca10hBnzZsF-awd7crn-uzhK8djjfpv&index="+a)
	elif inp==2:
		webbrowser.open("https://www.youtube.com/watch?v=k4yXQkG2s1E&list=PL3ca10hBnzZsYK5dOlCpmCX7yIp4HjsYz&index="+a)
	else:
		print("wrong choice")
	sys.exit()
def neutral():
	inp = int(input("Enter the choice:\n1.Neutral songs\n2.Happy songs\n3.Surprise songs\nchoice: "))
	if inp==1:
		webbrowser.open("https://www.youtube.com/watch?v=SxTYjptEzZs&list=PL3ca10hBnzZtfFg_iP7uk5pd0kmmfjL6e&index="+a)
	elif inp==2:
		webbrowser.open("https://www.youtube.com/watch?v=k4yXQkG2s1E&list=PL3ca10hBnzZsYK5dOlCpmCX7yIp4HjsYz&index="+a)
	elif inp==3:
		webbrowser.open("https://www.youtube.com/watch?v=2Vv-BfVoq4g&list=PL3ca10hBnzZsF-awd7crn-uzhK8djjfpv&index="+a)
	else:
		print("wrong choice")
	sys.exit()
def angry():
	inp = int(input("Enter the choice:\n1.Angry songs\n2.Happy songs\nchoice: "))
	if inp==1:
		webbrowser.open("https://www.youtube.com/watch?v=vIs4pNy8hzI&list=PL3ca10hBnzZs326cWrGAggZkHbGaOL0r8&index="+a)
	elif inp==2:
		webbrowser.open("https://www.youtube.com/watch?v=k4yXQkG2s1E&list=PL3ca10hBnzZsYK5dOlCpmCX7yIp4HjsYz&index="+a)
	else:
		print("wrong choice")
	sys.exit()
cap.release()
cv2.destroyAllWindows()
if __name__ == '__main__':
	if emo_dom == 'Angry':
	    angry()
	elif emo_dom == 'Neutral':
	    neutral()
	elif emo_dom == 'Sad':
		sad()
	elif emo_dom == 'Happy':
	    happy()
	elif emo_dom == 'Surprise':
		surprise()
	else : 
	    print("something went wrong")