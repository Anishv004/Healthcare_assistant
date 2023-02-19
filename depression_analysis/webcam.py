import cv2
import time
from deepface import DeepFace
roshan=0
# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the default camera
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Set the start time
start_time = time.time()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around each face and capture the image once every 10 seconds
    if time.time() - start_time >= 2: # 10 denotes the time in seconds
        for (x, y, w, h) in faces:
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Save the captured image as a JPEG file
            #img_name = "face_" + str(time.time()) + ".jpg"
            img_name = "face_"+str(roshan)+".jpg"#
            roshan+=1
            cv2.imwrite(img_name, frame)

            start_time = time.time()

    # Display the resulting frame
    roshan1=roshan
    cv2.imshow('Face Detection', frame)
    cc=cv2.waitKey(1)
    # Wait for 'q' key to quit
    if cc==27:
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows

cap.release()
cv2.destroyAllWindows()

