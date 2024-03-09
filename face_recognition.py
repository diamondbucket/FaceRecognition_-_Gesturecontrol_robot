#Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition   
import cv2
import numpy as np
import os 
from PIL import Image

def user_dataset_creator():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # For each person, enter one numeric face id
    face_id = int(input("enter the index of user(no of registered users:"))

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    
    while(True):
    
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
    
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
    
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
    
            cv2.imshow('image', img)
    
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
             break
    
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

def new_user_adder_training():
    path = 'dataset'
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
    
    
    def getImagesAndLabels(path):
    
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
    
        for imagePath in imagePaths:
    
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
    
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
    
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
    
        return faceSamples,ids
    
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
        
def queries():
    user_input = int(input("what do you wanna do: 1 for adding a new user, 2 for running the facial recognition window again, 3 for exiting the program: "))
    if user_input==1:
        user_dataset_creator()
    elif user_input==2:
        facial_recognition()
    elif user_input==3:
        cv2.destroyAllWindows()

#__________________________________________________________________________________________________________________________________________________________________
def facial_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    #iniciate id counter
    id = 0
    
    names = ['None', 'Pranav', 'Mini', 'HariKrishnan', 'Sanjay'] 
    
    
    cam = cv2.VideoCapture(0)
    cam.set(3, 1280)
    cam.set(4, 720)
    
    
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    
    while True:
    
        ret, image =cam.read()
    
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )
    
        for(x,y,w,h) in faces:
    
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
    
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
    
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(image, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(image, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera',image) 
    
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            cv2.destroyAllWindows()
            queries()
            break
    
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


        

facial_recognition()
