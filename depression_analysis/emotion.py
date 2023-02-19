import cv2
import time
from deepface import DeepFace
import webcam as wb
# Load the pre-trained face detection model
net_dep=0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
roshan2=wb.roshan1-1
print(roshan2)
l2=['angry', 'disgust','fear', 'happy', 'sad', 'surprise', 'neutral','No_face']
l={'angry': 0, 'disgust': 0,'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0,'No_face':0}
l1={'angry':0.7,'disgust':0.4,'fear':0.8,'happy':0.1,'sad':0.9,'surprise':0.2,'neutral':0.3,'No_face':0.5}
while (roshan2!=0):
    # Load the input image and convert it to grayscale
    img = cv2.imread('face_'+str(roshan2)+'.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over the detected faces and detect the emotion using DeepFace
    if faces!=():
        try:
            for (x, y, w, h) in faces:
                detected_face = img[int(y):int(y+h), int(x):int(x+w)]
                emotions = DeepFace.analyze(detected_face, actions=['emotion'])
                # Print the emotion with the highest probability
                emotion=emotions[0]['dominant_emotion']
                #emotion = max(emotions['emotion'], key=emotions['emotion'].get)
                print('Detected emotion:', emotion)
                l[emotion]+=1
                

                # Draw a rectangle around the detected face and label it with the detected emotion
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('output', img)
            roshan2-=1
        except:
            print("NO face detected:")
            l['No_face']+=1
            roshan2-=1

    # Display the output image
       
    else:
        print("NO face detected:")
        l['No_face']+=1
        roshan2-=1
for i in l2:
    net_dep+=l[i]*l1[i]
print("depression percentage is :",(net_dep/(wb.roshan1-1))*100,"%")
cv2.waitKey(0)
cv2.destroyAllWindows()
