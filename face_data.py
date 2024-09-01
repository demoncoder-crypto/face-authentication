import cv2
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


output_dir = 'dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


cap = cv2.VideoCapture(0)

print("Collecting images. Press 'q' to quit.")

count = 0
user_id = input("Enter user ID: ")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        color_face = frame[y:y+h, x:x+w]
        
        
        eyes = eye_cascade.detectMultiScale(face)
        
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        
        if len(eyes) >= 2:  
            face_filename = os.path.join(output_dir, f'user.{user_id}.{count}.jpg')
            cv2.imwrite(face_filename, face)
            print(f"Image {count} saved.")
            count += 1

    
    cv2.imshow('Collecting Images', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 10:
        break

cap.release()
cv2.destroyAllWindows()