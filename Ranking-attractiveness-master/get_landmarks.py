import numpy as np
import cv2
import dlib


cascade_path = "D:/Software Setups/Python package/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
predictor_path= "D:/Software Setups/dlib-master/python_examples/shape_predictor_68_face_landmarks.dat"


# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascade_path)
# create the landmark predictor
predictor = dlib.shape_predictor(predictor_path)
landmark_text=np.array([])
for i in range(1,501):
        image_path = "Z:/IIITB/MP/Attractiveness/SCUT/Data_Collection/SCUT-FBP-"+str(i)+".jpg"
        # Read the image
        image = cv2.imread(image_path)
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        #print "Found {0} faces!".format(len(faces))
        #print faces
        x=faces[0,0]
        y=faces[0,1]
        w=faces[0,2]
        h=faces[0,3]
        # Draw a rectangle around the faces
        #for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Converting the OpenCV rectangle coordinates to Dlib rectangle
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        #print dlib_rect

        detected_landmarks = predictor(image, dlib_rect).parts()

        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        # copying the image so we can see side-by-side
        image_copy = image.copy()
        #print landmarks
        for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])
                cv2.circle(image_copy, pos, 3, color=(0, 0, 255),thickness=-1)
        np.set_printoptions(suppress=True)
        #cv2.imshow("Landmarks found", image_copy)
        with open('Z:/IIITB/MP/Attractiveness/landmarks.txt','a') as f_handle:
                np.savetxt(f_handle,landmarks,fmt='%d')
        

cv2.waitKey(0)

